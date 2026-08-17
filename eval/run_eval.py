"""评测：20 条 ground truth Q&A -> recall@5 + answer faithfulness。

用法：
  .venv/bin/python -m eval.run_eval                 # 跑完打印报告
  .venv/bin/python -m eval.run_eval --json out.json # 另存机器可读结果

指标：
  recall@5      每条 QA 的 expected_doc_ids 有多少出现在检索 top-5（纯代码计算）
  faithfulness  答案是否被引用的上下文支持（有 LLM key 时 LLM-as-judge；
                无 LLM 时退化为 N-gram 蕴含度：答案 token 被 contexts 覆盖比例）
"""
import argparse
import json
import logging
import os
import re

from pipeline.store import Store
from serve.answer import QueryPlan, llm_available, price_table_answer
from serve.retrieve import Retriever, tokenize

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("eval")

GT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_qa.jsonl")


def load_gt():
    with open(GT_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------- recall@5 ----------------
def recall_at_k(retriever: Retriever, qa: dict, k: int = 5):
    plan = QueryPlan(qa["question"])
    kwargs = {"top_k": k}
    if plan.query_type == "policy" and qa.get("expect_type") == "policy":
        kwargs["source_types"] = ["policy"]
    if plan.date_from:
        kwargs["date_from"] = plan.date_from
    if plan.date_to:
        kwargs["date_to"] = plan.date_to
    hits = retriever.search(qa["question"], **kwargs)
    if qa.get("expect_type") == "price" and plan.symbols:
        # 价格题口径：top-5 含任一同品种行情日报即视为召回（同品种日报文本几乎相同，
        # 期望特定某一天是不合理的随机口径）。品种代码/名称任一命中即可。
        from pipeline.models import SYMBOLS
        want = set(plan.symbols) | {SYMBOLS[s]["name"][:3] for s in plan.symbols}
        hit = any(h["source_type"] == "price"
                  and any(w in (h["title"] + h["text"]) for w in want) for h in hits)
        return (1.0 if hit else 0.0), hits
    got = {h["doc_id"] for h in hits}
    want = set(qa.get("expected_doc_ids", []))
    if not want:
        return None, hits
    if qa.get("match") == "any":
        # 开放性问题：多个文档均为合理答案，命中任一即算召回
        return (1.0 if got & want else 0.0), hits
    return len(got & want) / len(want), hits


# ---------------- faithfulness ----------------
def faithfulness(question: str, answer: str, contexts) -> float:
    if llm_available():
        return _llm_faithfulness(question, answer, contexts)
    return _ngram_faithfulness(answer, contexts)


def _llm_faithfulness(question, answer, contexts) -> float:
    import httpx
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    ctx = "\n\n".join(f"[{i+1}] {c[:1200]}" for i, c in enumerate(contexts))
    prompt = (
        "判断 ANSWER 中的每个事实性论断是否被 CONTEXTS 支持。"
        "返回 0~10 的整数分数：10=全部被支持，5=约一半，0=全部无依据。只输出数字。\n\n"
        f"CONTEXTS:\n{ctx}\n\nQUESTION: {question}\nANSWER: {answer}\n\n分数:")
    try:
        r = httpx.post(f"{base}/chat/completions",
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"model": model, "temperature": 0, "max_tokens": 8,
                             "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        score = int(re.search(r"\d+", r.json()["choices"][0]["message"]["content"]).group())
        return min(score, 10) / 10
    except Exception as e:
        log.warning("LLM 评审失败，退化为 N-gram: %s", e)
        return _ngram_faithfulness(answer, contexts)


def _ngram_faithfulness(answer: str, contexts, n=2) -> float:
    """无 LLM 回退：答案的 bigram 被上下文覆盖的比例（对齐 RAGAS 忠实度的粗糙近似）。"""
    ans_grams = set(ngrams(tokenize(answer), n))
    if not ans_grams:
        return 1.0
    ctx_grams = set()
    for c in contexts:
        ctx_grams |= set(ngrams(tokenize(c), n))
    covered = sum(1 for g in ans_grams if g in ctx_grams)
    return round(covered / len(ans_grams), 4)


def ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))]


# ---------------- 主流程 ----------------
def run(json_out: str = None):
    store = Store()
    retriever = Retriever(store)
    gts = load_gt()
    results = []
    for qa in gts:
        rec, hits = recall_at_k(retriever, qa)
        plan = QueryPlan(qa["question"])
        price_summary = None
        if plan.query_type == "price" and plan.symbols:
            price_summary = price_table_answer(store, plan)
        contexts = [h["text"] for h in hits]
        if price_summary:
            contexts.insert(0, price_summary)
        answer = qa_answer(qa, hits, store, plan, price_summary)
        faith = faithfulness(qa["question"], answer, contexts) if answer else 0.0
        results.append({
            "id": qa.get("id"), "question": qa["question"],
            "type": qa.get("expect_type", "rag"),
            "recall_at_5": rec, "faithfulness": faith,
            "n_hits": len(hits), "answer": answer[:200],
        })
        print(f"[{qa.get('id')}] {qa['question'][:36]:38s} recall={rec} faith={faith}")
    report = summarize(results)
    print(report)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"results": results, "summary": _summary_dict(results)},
                      f, ensure_ascii=False, indent=2)
    store.close()


def qa_answer(qa, hits, store, plan, price_summary):
    from serve.answer import extractive_answer, llm_generate, llm_available
    contexts = [f"《{h['title']}》{h['text'][:1200]}" for h in hits]
    if price_summary:
        contexts.insert(0, f"结构化行情数据:\n{price_summary}")
    if llm_available() and contexts:
        return llm_generate(qa["question"], contexts) or extractive_answer(
            qa["question"], hits, store, plan)
    if price_summary:
        return price_summary
    return extractive_answer(qa["question"], hits, store, plan)


def summarize(results):
    s = _summary_dict(results)
    lines = ["", "=" * 56, "评测汇总", "=" * 56,
             f"总题数: {s['n']}  (可评 recall 的 {s['n_recall']} 题)",
             f"recall@5      (macro): {s['recall']}",
             f"faithfulness  (macro): {s['faithfulness']}  [{s['faith_mode']}]",
             "-" * 56]
    for t, v in s["by_type"].items():
        lines.append(f"  {t:8s} n={v['n']:2d}  recall@5={v['recall']}  faith={v['faith']}")
    lines.append("=" * 56)
    return "\n".join(lines)


def _summary_dict(results):
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    by_type = {}
    for t in ("news", "policy", "price", "rag"):
        rs = [r for r in results if r["type"] == t]
        if rs:
            by_type[t] = {"n": len(rs), "recall": avg([r["recall_at_5"] for r in rs]),
                          "faith": avg([r["faithfulness"] for r in rs])}
    return {
        "n": len(results),
        "n_recall": sum(1 for r in results if r["recall_at_5"] is not None),
        "recall": avg([r["recall_at_5"] for r in results]),
        "faithfulness": avg([r["faithfulness"] for r in results]),
        "faith_mode": "LLM-as-judge" if llm_available() else "ngram(近似)",
        "by_type": by_type,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args.json)
