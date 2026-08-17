"""问答层：查询理解（时间/品种/意图，规则优先、LLM 可选）+ 答案生成。

LLM 可选：设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL）则走 LLM 生成；
未设置时回退为抽取式回答（引用原文关键句 + 价格表格），保证系统零外部依赖可跑。
"""
import datetime
import logging
import os
import re
from typing import List, Optional, Tuple

from pipeline.models import SYMBOLS
from pipeline.store import Store

log = logging.getLogger("answer")

TODAY = datetime.date.today()

_WEEK_NUMS = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

SYMBOL_KEYWORDS = {
    "LME_CU": ["lme铜", "铜价", "电解铜", "精炼铜", "copper"],
    "LME_ZN": ["lme锌", "锌价", "zinc"],
    "LME_NI": ["lme镍", "镍价", "nickel"],
    "SHFE_LC": ["碳酸锂", "锂价", "氯化锂", "氢氧化锂", "lithium"],
    "IRON_ORE": ["铁矿石", "铁矿", "iron ore", "pb粉"],
}
PRICE_INTENT = re.compile(r"价格|行情|走势|涨|跌|收盘|多少钱|报价|结算|均价")
POLICY_INTENT = re.compile(r"政策|法规|条例|规定|监管|管理办法|公告|战略|出口管制|许可")
NEWS_INTENT = re.compile(r"新闻|动态|消息|事件|发生了什么|资讯|报道")


class QueryPlan:
    def __init__(self, question: str):
        self.question = question
        self.date_from, self.date_to = parse_time_range(question)
        self.symbols = detect_symbols(question)
        self.query_type = detect_type(question, self.symbols)

    def as_dict(self):
        return {"date_from": self.date_from, "date_to": self.date_to,
                "symbols": self.symbols, "query_type": self.query_type}


def parse_time_range(q: str) -> Tuple[Optional[str], Optional[str]]:
    q = re.sub(r"\s+", "", q)          # "近 7 天" -> "近7天"，空格不参与匹配
    m = re.search(r"(近|最近|过去)(?:的)?(\d+)(天|日)", q)
    if m:
        return (TODAY - datetime.timedelta(days=int(m.group(2)))).isoformat(), None
    m = re.search(r"(近|最近|过去)(?:的)?([一二两三四五六七八九十\d]+)(周|个?星期)", q)
    if m:
        n = _to_int(m.group(2))
        return (TODAY - datetime.timedelta(weeks=n)).isoformat(), None
    m = re.search(r"(近|最近|过去)(?:的)?([一二两三四五六七八九十\d]+)个?月", q)
    if m:
        n = _to_int(m.group(2))
        d = TODAY - datetime.timedelta(days=30 * n)
        return f"{d.year}-{d.month:02d}-01", None
    if re.search(r"本周|这周", q):
        return (TODAY - datetime.timedelta(days=TODAY.weekday())).isoformat(), None
    if re.search(r"上周", q):
        start = TODAY - datetime.timedelta(days=TODAY.weekday() + 7)
        return start.isoformat(), (start + datetime.timedelta(days=6)).isoformat()
    if re.search(r"本月|这个月", q):
        return f"{TODAY.year}-{TODAY.month:02d}-01", None
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", q)
    if m:
        d = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        return d, d
    return None, None


def _to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    return _WEEK_NUMS.get(s, 1)


def _days_since_weekday(n: int) -> int:
    return TODAY.weekday() + n


def detect_symbols(q: str) -> List[str]:
    low = q.lower()
    return [sym for sym, kws in SYMBOL_KEYWORDS.items()
            if any(k in low for k in kws)]


def detect_type(q: str, symbols: List[str]) -> str:
    if symbols and PRICE_INTENT.search(q):
        return "price"
    if POLICY_INTENT.search(q):
        return "policy"
    return "rag"


# ---------------- LLM（可选） ----------------
def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def llm_generate(question: str, contexts: List[str]) -> Optional[str]:
    """OpenAI 兼容接口（可指向任意兼容网关）。失败返回 None 由调用方回退。"""
    import httpx
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    ctx = "\n\n".join(f"[{i+1}] {c[:1500]}" for i, c in enumerate(contexts))
    prompt = (
        "你是矿业行业研究助手。仅依据下面编号的参考资料回答问题，"
        "每个事实后标注来源编号（如[1]）；资料不足以回答时明确说明，不要编造。\n\n"
        f"参考资料:\n{ctx}\n\n问题: {question}\n回答（中文）:")
    try:
        r = httpx.post(f"{base}/chat/completions",
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"model": model, "temperature": 0.1, "max_tokens": 600,
                             "messages": [{"role": "user", "content": prompt}]},
                       timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("LLM 调用失败，回退抽取式回答: %s", e)
        return None


# ---------------- 抽取式回退（无 LLM 也可用） ----------------
def extractive_answer(question: str, hits: List[dict], store: Store,
                      plan: QueryPlan) -> str:
    parts = []
    for i, h in enumerate(hits[:5], 1):
        if h["source_type"] == "price":
            line = h["text"]
        else:
            sents = re.split(r"(?<=[。！？!?])", h["text"])
            line = max(sents, key=len)[:200] if sents else h["title"]
        parts.append(f"[{i}] {h['title']}（{h['source_name']}，{h['published_at']}）：{line}")
    if not parts:
        return "在当前数据范围内没有检索到相关内容。"
    return "\n".join(parts)


def price_table_answer(store: Store, plan: QueryPlan, limit: int = 10) -> Optional[str]:
    """品种+价格意图：直接从结构化 prices 表生成最近行情摘要。"""
    lines = []
    for sym in plan.symbols:
        rows = store.query_prices(sym, plan.date_from or _default_from(), plan.date_to or "9999-12-31")
        if not rows:
            continue
        recent = rows[-limit:]
        name = SYMBOLS[sym]["name"]
        last = recent[-1]
        seg = [f"{name}（{sym}）近{len(recent)}个交易日（{recent[0]['trade_date']}~{last['trade_date']}）："
               f"最新 {last['trade_date']} 收于 {last['close']:g} {last['unit']}"]
        if last.get("change_pct") is not None:
            seg.append(f"日涨跌 {last['change_pct']:+.2f}%")
        closes = [r["close"] for r in recent]
        if len(closes) >= 2:
            seg.append(f"区间累计 {(closes[-1]/closes[0]-1)*100:+.1f}%，最高 {max(closes):g}，最低 {min(closes):g}")
        lines.append("，".join(seg) + "。")
    return "\n".join(lines) if lines else None


def _default_from() -> str:
    return (TODAY - datetime.timedelta(days=30)).isoformat()
