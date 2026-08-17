"""FastAPI 服务：POST /query 自然语言问答。

启动：
  .venv/bin/uvicorn serve.main:app --port 8000

示例：
  curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
       -d '{"question": "近 7 天锂价格走势如何?"}'
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from pipeline.store import Store
from .answer import (QueryPlan, extractive_answer, llm_available, llm_generate,
                     price_table_answer)
from .retrieve import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("serve")

store: Optional[Store] = None
retriever: Optional[Retriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever
    store = Store()
    retriever = Retriever(store)
    log.info("库内文档 %d 篇，chunk %d 条，LLM=%s",
             store.count(), len(retriever.rows), "启用" if llm_available() else "未启用(抽取式回答)")
    yield
    store.close()


app = FastAPI(title="矿业三源聚合问答", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = 5


class Citation(BaseModel):
    rank: int
    doc_id: str
    title: str
    source_name: str
    source_type: str
    url: str
    published_at: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    plan: dict
    citations: List[Citation]
    price_summary: Optional[str] = None
    latency_ms: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.monotonic()
    plan = QueryPlan(req.question)

    # 意图 -> 检索过滤：policy 类只在政策源里找，price 意图同时拉价格日报
    kwargs = {"top_k": req.top_k}
    if plan.query_type == "policy":
        kwargs["source_types"] = ["policy"]
    if plan.date_from:
        kwargs["date_from"] = plan.date_from
    if plan.date_to:
        kwargs["date_to"] = plan.date_to
    hits = retriever.search(req.question, **kwargs)

    # 结构化价格路：prices 表精确数据（有 LLM 时作为上下文，无 LLM 时直接作答）
    price_summary = None
    if plan.query_type == "price" and plan.symbols:
        price_summary = price_table_answer(store, plan)

    contexts = []
    for h in hits:
        header = f"《{h['title']}》 来源:{h['source_name']} 日期:{h['published_at']}"
        contexts.append(f"{header}\n{h['text'][:1200]}")
    if price_summary:
        contexts.insert(0, f"结构化行情数据（权威，优先引用）:\n{price_summary}")

    if llm_available() and contexts:
        answer = llm_generate(req.question, contexts) or extractive_answer(
            req.question, hits, store, plan)
    elif price_summary:
        answer = price_summary
    else:
        answer = extractive_answer(req.question, hits, store, plan)

    citations = [Citation(
        rank=i + 1, doc_id=h["doc_id"], title=h["title"], source_name=h["source_name"],
        source_type=h["source_type"], url=h["url"], published_at=h["published_at"],
        snippet=h["text"][:200]).model_dump() for i, h in enumerate(hits)]
    return QueryResponse(
        question=req.question, answer=answer, plan=plan.as_dict(),
        citations=citations, price_summary=price_summary,
        latency_ms=int((time.monotonic() - t0) * 1000))


@app.get("/health")
def health():
    return {"ok": True, "documents": store.count() if store else 0,
            "chunks": len(retriever.rows) if retriever else 0, "llm": llm_available()}


@app.get("/stats")
def stats():
    with store._lock:
        by_source = [dict(r) for r in store.conn.execute(
            "SELECT source_name, source_type, COUNT(*) c FROM documents "
            "GROUP BY source_name, source_type")]
        prices = [dict(r) for r in store.conn.execute(
            "SELECT symbol, COUNT(*) c FROM prices GROUP BY symbol")]
    return {"documents": by_source, "price_rows": prices}
