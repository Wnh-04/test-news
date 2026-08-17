"""pipeline 入口：采集 -> 清洗(源内已做) -> 去重 -> 向量化 -> 入库。

用法：
  .venv/bin/python -m pipeline.run            # 全量
  .venv/bin/python -m pipeline.run --stats    # 只看库内统计

幂等：doc_id 主键 + SimHash 内容去重，重复运行不会产生重复数据。
"""
import argparse
import logging
from typing import List

from .clean import chunk_text
from .dedup import SimhashIndex, simhash
from .embed import Embedder
from .models import Document
from .store import Store
from . import sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run")


def ingest_documents(store: Store, embedder: Embedder, docs: List[Document]) -> int:
    """三层去重后入库：URL 主键 / SimHash 近重复 / chunk hash。返回实际入库数。"""
    index = SimhashIndex(store.all_fingerprints())
    kept = 0
    for n, doc in enumerate(docs, 1):
        if store.has_doc(doc.doc_id):
            continue                                    # 第一层：URL 主键
        fp = simhash(doc.text)
        if doc.source_type != "price":
            dup = index.find_duplicate(fp)              # 第二层：内容近重复
            if dup:
                log.debug("SimHash 近重复，跳过 %s (≈%s)", doc.url, dup)
                continue
        store.upsert_document(doc, fp)
        index.add(doc.doc_id, fp)
        _embed_and_chunk(store, embedder, doc)          # 第三层在 replace_chunks 内
        kept += 1
        if n % 20 == 0:
            store.commit()                              # 增量提交，中断不丢已采数据
    store.commit()
    return kept


def _embed_and_chunk(store: Store, embedder: Embedder, doc: Document):
    # 标题独立成块（跨语言召回主要靠标题语义，正文块细节多会稀释相似度）
    chunks = ([doc.title] if doc.title else []) + chunk_text(doc.text)
    if not chunks:
        return
    embs = embedder.embed(chunks)
    store.replace_chunks(doc.doc_id, chunks, embs)


def rebuild_chunks():
    """分块策略变更后重建全库 chunks（documents 不动，重新向量化）。"""
    store = Store()
    embedder = Embedder()
    n = 0
    for r in store.conn.execute("SELECT doc_id, title, text FROM documents"):
        doc = Document(doc_id=r["doc_id"], source_type="", source_name="", url="",
                       title=r["title"] or "", published_at="", lang="", text=r["text"])
        _embed_and_chunk(store, embedder, doc)
        n += 1
        if n % 50 == 0:
            store.commit()
            log.info("rechunk %d/%d", n, store.count())
    store.commit()
    store.close()
    log.info("重建完成: %d 篇", n)


def run():
    store = Store()
    embedder = Embedder()
    try:
        # ---- 源1 新闻 ----
        for name, fn in sources.news_sources():
            _run_one(store, embedder, name, lambda f=fn: f(store))

        # ---- 源2 政策 ----
        for name, fn in sources.policy_sources_():
            _run_one(store, embedder, name, lambda f=fn: f(store))

        # ---- 源3 价格 ----
        for name, fn in sources.price_source():
            _run_one(store, embedder, name, lambda f=fn: f(), is_price=True)
    finally:
        store.close()


def _run_one(store: Store, embedder: Embedder, name: str, collect_fn, is_price: bool = False):
    log.info("========== 采集源: %s ==========", name)
    try:
        result = collect_fn()
    except Exception as e:                                  # 单源失败不影响整体
        store.log_run(name, "failed", 0, 0, f"{type(e).__name__}: {e}")
        store.commit()
        log.exception("源 %s 失败", name)
        return
    if is_price:
        rows, docs = result
        store.upsert_prices(rows)
        kept = ingest_documents(store, embedder, docs)
        note = f"{len(rows)} 行情行({len(set(r.symbol for r in rows))}个品种) + {len(docs)} 日报"
    else:
        rows, docs = None, result
        kept = ingest_documents(store, embedder, docs)
        note = ""
    status = "ok" if kept else "skipped"
    store.log_run(name, status, len(docs), kept, note)
    store.commit()
    log.info("源 %s 完成: 抓取 %d, 入库 %d (%s)", name, len(docs), kept, status)


def stats():
    store = Store()
    try:
        print("=== 库内统计 ===")
        for st in ("news", "policy", "price"):
            print(f"{st:8s}: {store.count(st)} 篇")
        print(f"{'合计':8s}: {store.count()} 篇")
        print("\n=== 按来源 ===")
        for r in store.conn.execute(
                "SELECT source_name, source_type, COUNT(*) c, MIN(published_at) mn, "
                "MAX(published_at) mx FROM documents GROUP BY source_name, source_type"):
            print(f"{r['source_name']:14s} [{r['source_type']:6s}] {r['c']:4d} 篇  "
                  f"{r['mn']} ~ {r['mx']}")
        print("\n=== 行情 ===")
        for r in store.conn.execute(
                "SELECT symbol, COUNT(*) c, MIN(trade_date) mn, MAX(trade_date) mx "
                "FROM prices GROUP BY symbol"):
            print(f"{r['symbol']:10s} {r['c']:3d} 天  {r['mn']} ~ {r['mx']}")
        print("\n=== 采集日志 ===")
        for r in store.conn.execute(
                "SELECT ts, source, status, fetched, kept, note FROM ingest_log "
                "ORDER BY id DESC LIMIT 12"):
            print(f"{r['ts']} {r['source']:12s} {r['status']:8s} fetched={r['fetched']:4d} "
                  f"kept={r['kept']:4d} {r['note'] or ''}")
    finally:
        store.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="只打印库内统计")
    ap.add_argument("--rebuild-chunks", action="store_true",
                    help="分块策略变更后重建全库向量块")
    args = ap.parse_args()
    if args.stats:
        stats()
    elif args.rebuild_chunks:
        rebuild_chunks()
    else:
        run()
