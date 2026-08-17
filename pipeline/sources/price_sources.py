"""源3 价格：新浪财经公开期货日K线接口（免登录、含完整历史）。

  LME铜/锌/镍  -> GlobalFuturesService (外盘连续 CAD/ZSD/NID, 美元/吨)
  碳酸锂       -> InnerFuturesNewService (广期所主力 LC0, 元/吨)
  铁矿石       -> InnerFuturesNewService (大商所主力 I0, 元/吨)

题目原定 LME官网/SHFE/上海钢联均有登录墙或频控，此处按方案用合规公开接口替代，
口径差异记录在 DATA_NOTES.md。数据双写：
  1) prices 表（结构化，供精确数值查询）
  2) 每品种每天生成一条价格日报 Document（供自然语言检索命中）
"""
import datetime
import json
import logging
import re
from typing import List, Tuple

from .. import fetch
from ..models import Document, PriceRow, SYMBOLS

log = logging.getLogger("source.price")

GLOBAL_URL = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
              "var%20_v=/GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={sym}")
INNER_URL = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/"
             "var%20_v=/InnerFuturesNewService.getDailyKLine?symbol={sym}")
HEADERS = {"Referer": "https://finance.sina.com.cn", "Accept": "*/*"}
DAYS_WINDOW = 90          # 日历天：5品种 × ~60交易日 ≈ 300 行，满足源级 ≥200 条


def collect() -> Tuple[List[PriceRow], List[Document]]:
    cutoff = (datetime.date.today() - datetime.timedelta(days=DAYS_WINDOW)).isoformat()
    all_rows, all_docs = [], []
    for symbol, meta in SYMBOLS.items():
        rows = _fetch_symbol(symbol, meta)
        rows = [r for r in rows if r.trade_date >= cutoff]
        _fill_change_pct(rows)
        all_rows.extend(rows)
        all_docs.extend(_digest_documents(symbol, meta, rows))
        log.info("%s(%s): %d 个交易日", symbol, meta["sina"], len(rows))
    return all_rows, all_docs


def _fetch_symbol(symbol: str, meta: dict) -> List[PriceRow]:
    is_global = symbol.startswith("LME_")
    url = (GLOBAL_URL if is_global else INNER_URL).format(sym=meta["sina"])
    try:
        r = fetch.fetch(url, headers=HEADERS)
    except fetch.FetchError as e:
        log.error("%s 行情获取失败: %s", symbol, e)
        return []
    m = re.search(r"\((.*)\)\s*;?\s*$", r.text.strip(), re.S)
    if not m:
        log.error("%s JSONP 解析失败", symbol)
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    out = []
    for k in data:
        try:
            if is_global:
                out.append(PriceRow(symbol=symbol, trade_date=k["date"],
                                    open=float(k["open"]), high=float(k["high"]),
                                    low=float(k["low"]), close=float(k["close"]),
                                    unit=meta["unit"], source="sina_global"))
            else:
                out.append(PriceRow(symbol=symbol, trade_date=k["d"],
                                    open=float(k["o"]), high=float(k["h"]),
                                    low=float(k["l"]), close=float(k["c"]),
                                    unit=meta["unit"], source="sina_inner"))
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda x: x.trade_date)
    return out


def _fill_change_pct(rows: List[PriceRow]):
    prev = None
    for r in rows:
        if prev:
            r.change_pct = round((r.close - prev.close) / prev.close * 100, 2)
        prev = r


def _digest_documents(symbol: str, meta: dict, rows: List[PriceRow]) -> List[Document]:
    """每品种每天一条摘要 -> source_type=price 的 Document，主键用合成URL保证稳定。"""
    closes = [r.close for r in rows]
    docs = []
    for i, r in enumerate(rows):
        chg = f"{'上涨' if (r.change_pct or 0) > 0 else '下跌' if (r.change_pct or 0) < 0 else '持平'}"
        pct = f"{abs(r.change_pct)}%" if r.change_pct is not None else "未知"
        week = ""
        if i >= 5 and closes[i - 5]:
            wp = (r.close - closes[i - 5]) / closes[i - 5] * 100
            week = f"近5个交易日累计{'上涨' if wp > 0 else '下跌'}{abs(wp):.1f}%。"
        text = (f"{meta['name']}（{symbol}）{r.trade_date} 收于 {r.close:g} {r.unit}，"
                f"较前一交易日{chg}{pct}。当日最高 {r.high:g}，最低 {r.low:g}。{week}")
        url = f"price://{symbol}/{r.trade_date}"      # 合成URL：doc_id 稳定可重跑
        docs.append(Document(doc_id=fetch.doc_id_for(url), source_type="price",
                             source_name="price_feed", url=url, title=f"{meta['name']} {r.trade_date} 行情",
                             published_at=r.trade_date, lang="zh", text=text))
    return docs
