"""检索层：向量（余弦）+ BM25 混合检索，RRF 融合，时间/来源过滤，文档级聚合。

规模在数千 chunk，全部载入内存暴力算即可；接口语义与 Qdrant 一致，量级上来后
可将 VectorPart 换成向量库而不动调用方。
"""
import logging
import math
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from pipeline.embed import Embedder
from pipeline.store import Store

log = logging.getLogger("retrieve")

K1, B = 1.5, 0.75          # BM25 参数
VEC_W, BM25_W = 0.65, 0.35  # 召回融合权重：跨语言召回主要靠向量，BM25 兜住关键词精确匹配
TITLE_BOOST = 1.5           # 标题块命中加权：标题与查询对齐是最强相关性信号（实测1.5最优，
                            # 过大会让中文政策标题整体膨胀、挤压英文新闻，见评测记录）
RERANK_CANDIDATES = 30      # 召回候选数（cross-encoder 可用时精排，下载失败自动退回）

# 中英矿业词表（查询扩展）：中文查询追加英文关键词到 BM25 路，弥合跨语言字面鸿沟。
# 双塔 embedding 的跨语言相似度区分度有限（中文问题 vs 英文标题 ~0.55 与中文噪声同档），
# 词表命中能直接把英文文档顶上来。规模刻意小而准，避免误扩展。
ZH_EN = {
    "宁德时代": "catl", "天齐锂业": "tianqi", "赣锋锂业": "ganfeng",
    "智利": "chile", "秘鲁": "peru", "澳大利亚": "australia", "澳洲": "australia",
    "印尼": "indonesia", "印度尼西亚": "indonesia", "刚果": "congo", "蒙古": "mongolia",
    "铜": "copper", "锂": "lithium", "镍": "nickel", "锌": "zinc", "钴": "cobalt",
    "黄金": "gold", "金": "gold", "白银": "silver", "银": "silver", "铀": "uranium",
    "煤炭": "coal", "煤矿": "coal mine", "铁矿": "iron ore", "铁矿石": "iron ore",
    "稀土": "rare earth", "铝": "aluminium", "钨": "tungsten", "钼": "molybdenum",
    "停产": "shutdown", "关停": "shutdown", "罢工": "strike", "事故": "accident",
    "产量": "output", "出口": "export", "关税": "tariff", "许可": "permit",
    "并购": "merger", "收购": "acquisition", "投资": "investment", "融资": "funding",
    "价格": "price", "行情": "market", "库存": "inventory", "供应": "supply",
}


def expand_query(query: str) -> str:
    """中文关键词 -> 追加英文等价词（仅用于检索，不影响展示）。"""
    extra = [en for zh, en in ZH_EN.items() if zh in query]
    return query + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


_reranker = None
_rerank_disabled = os.environ.get("DISABLE_RERANK") == "1"


def _get_reranker():
    """中英双语交叉编码器（进程级单例）。模型 ~1GB，当前网络下载受限时自动禁用--
    检索退化为召回排序，系统仍完整可用；换网络重跑或预下载缓存后自动启用。"""
    global _reranker, _rerank_disabled
    if _reranker is None:
        if _rerank_disabled:
            return None
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        log.info("加载精排模型 BAAI/bge-reranker-base…")
        try:
            _reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        except Exception as e:
            _rerank_disabled = True
            log.warning("精排模型不可用（%s），本进程禁用精排", e)
            return None
    return _reranker


def tokenize(text: str) -> List[str]:
    """中文 jieba 分词 + 英文按词，全部小写，丢弃标点。"""
    import jieba
    out = []
    for ch in jieba.cut(text):
        low = ch.lower().strip()
        if not low:
            continue
        if all("一" <= c <= "鿿" for c in low) or low.isalnum():
            out.append(low)
    return out


class _BM25:
    def __init__(self, corpus_tokens: List[List[str]]):
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.n, 1)
        self.df = defaultdict(int)
        for d in corpus_tokens:
            for t in set(d):
                self.df[t] += 1
        self.tfs = [defaultdict(int) for _ in corpus_tokens]
        for i, d in enumerate(corpus_tokens):
            for t in d:
                self.tfs[i][t] += 1

    def idf(self, t: str) -> float:
        return math.log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        scores = np.zeros(self.n)
        for i in range(self.n):
            tf_map, dl = self.tfs[i], len(self.docs[i])
            s = 0.0
            for t in query_tokens:
                tf = tf_map.get(t)
                if not tf:
                    continue
                s += self.idf(t) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / self.avgdl))
            scores[i] = s
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


class Retriever:
    def __init__(self, store: Store, embedder: Optional[Embedder] = None):
        self.store = store
        self.embedder = embedder or Embedder()
        rows = store.load_chunk_rows()
        if not rows:
            log.warning("库为空，请先运行 .venv/bin/python -m pipeline.run")
        self.rows = rows
        self.chunk_ids = [r["chunk_id"] for r in rows]
        self.doc_ids = [r["doc_id"] for r in rows]
        embs = [r["embedding"] for r in rows]
        self.matrix = (np.stack([e for e in embs if e is not None])
                       if any(e is not None for e in embs) else None)
        self._emb_row_of = [i for i, e in enumerate(embs) if e is not None]
        self.bm25 = _BM25([tokenize(r["text"]) for r in rows])

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, date_from: str = None,
               date_to: str = None, source_types: List[str] = None,
               exclude_types: List[str] = None) -> List[dict]:
        """返回 top_k 篇文档（每篇带最相关的 chunk）。"""
        allowed = self._allowed_rows(date_from, date_to, source_types, exclude_types)
        if not allowed:
            return []

        # 两路召回 -> 分数归一 -> 加权融合。
        # 注意不要对候选做 top-N 截断：单路排名靠后的文档可能两路融合后胜出
        # （本库 <1 万 chunk，全量计算毫秒级）
        fused = defaultdict(float)
        q_tokens = tokenize(expand_query(query))
        if q_tokens and self.bm25.n:
            hits = self.bm25.search(q_tokens, 300)
            mx = max((s for _, s in hits), default=0) or 1.0
            for i, s in hits:
                if i in allowed:
                    fused[i] += BM25_W * (s / mx)
        if self.matrix is not None:
            q_emb = self.embedder.embed_one(query)
            sims = self.matrix @ q_emb
            order = np.argsort(-sims)
            for row in order:
                i = self._emb_row_of[row]
                if i in allowed and sims[row] > 0:
                    fused[i] += VEC_W * float(sims[row])
        if not fused:
            return []

        # chunk -> 文档聚合：每篇文档取融合分最高的 chunk 作为代表；
        # 标题块（idx=0 且短）命中的文档加权--标题对齐是最强相关性信号
        best_per_doc = {}
        for i, score in fused.items():
            d = self.doc_ids[i]
            if d not in best_per_doc or score > best_per_doc[d][1]:
                best_per_doc[d] = (i, score)
        ranked = []
        for d, (i, score) in best_per_doc.items():
            if self.rows[i]["idx"] == 0 and len(self.rows[i]["text"]) < 100:
                score *= TITLE_BOOST
            ranked.append((i, score))
        ranked.sort(key=lambda x: -x[1])
        ranked = ranked[:max(top_k, RERANK_CANDIDATES)]
        if len(ranked) > top_k:
            if _get_reranker() is not None:
                ranked = self._rerank(query, ranked)
            ranked = ranked[:top_k]
        return [{**self.rows[i], "score": round(s, 5)} for i, s in ranked]

    def _rerank(self, query: str, ranked: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """cross-encoder 精排：直接对 (query, title+chunk) 打相关性分。"""
        try:
            reranker = _get_reranker()
            pairs = [(query, f"{self.rows[i]['title']}。{self.rows[i]['text'][:2000]}")
                     for i, _ in ranked]
            scores = list(reranker.rerank(pairs))
            out = [(i, float(s)) for (i, _), s in zip(ranked, scores)]
            return sorted(out, key=lambda x: -x[1])
        except Exception as e:
            log.warning("精排失败，退回召回排序: %s", e)
            return ranked

    def _allowed_rows(self, date_from, date_to, source_types, exclude_types) -> set:
        allowed = set()
        for i, r in enumerate(self.rows):
            if date_from and r["published_at"] < date_from:
                continue
            if date_to and r["published_at"] > date_to:
                continue
            st = r["source_type"]
            if source_types and st not in source_types:
                continue
            if exclude_types and st in exclude_types:
                continue
            allowed.add(i)
        return allowed
