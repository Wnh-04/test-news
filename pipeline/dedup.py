"""去重层：
第一层  URL 归一化 + doc_id 主键（fetch.normalize_url / fetch.doc_id_for）
第二层  SimHash 内容指纹，海明距离 <= 3 视为近重复（转载/通稿）
第三层  chunk 级 hash 去重在入库时做（store.ensure_chunks）
"""
from typing import List, Tuple

_WEIGHTS = 64  # 64-bit simhash


def _tokens(text: str) -> List[str]:
    """中英混合分词：英文按词、中文按 2-gram（避免外部分词依赖，SimHash 够用）。"""
    tokens = []
    buf = ""
    for ch in text:
        if "一" <= ch <= "鿿":
            if buf:
                tokens.append(buf.lower())
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf.lower())
                buf = ""
    if buf:
        tokens.append(buf.lower())
    # 中文 2-gram
    cjk = [t for t in tokens if len(t) == 1 and "一" <= t[0] <= "鿿"]
    grams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    words = [t for t in tokens if len(t) > 1]
    return words + grams


def simhash(text: str) -> int:
    """简化版 SimHash：token 哈希位投票。文本足够相似时指纹接近。"""
    bits = [0] * _WEIGHTS
    for tok in _tokens(text[:8000]):
        h = hash(tok) & 0xFFFFFFFFFFFFFFFF
        for i in range(_WEIGHTS):
            bits[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(_WEIGHTS) if bits[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimhashIndex:
    """已入库指纹的内存索引。规模 O(n) 暴力扫即可（万级以下）。"""

    def __init__(self, existing: List[Tuple[str, int]] = ()):  # [(doc_id, fingerprint)]
        self.items = list(existing)

    def find_duplicate(self, fingerprint: int, threshold: int = 3):
        """返回近重复的已有 doc_id（保留先入库者，即发布更早/先抓取者）。"""
        for doc_id, fp in self.items:
            if hamming(fp, fingerprint) <= threshold:
                return doc_id
        return None

    def add(self, doc_id: str, fingerprint: int):
        self.items.append((doc_id, fingerprint))
