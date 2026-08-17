"""源1 矿业新闻：mining.com RSS（WordPress 分页）。
RSS 只给摘要 -> 逐条抓原文全文（trafilatura 抽正文），这是本源的考察点。
"""
import datetime
import logging
import re
from typing import List

import feedparser

from .. import fetch
from ..clean import extract_text, clean_document_text
from ..models import Document

log = logging.getLogger("source.news")

FEED = "https://www.mining.com/feed/"
MAX_PAGES = 15          # 每页约36条
MAX_DOCS = 250          # 题目要求近30天≥200条，取最新250条抓全文即可
DAYS_WINDOW = 35        # 近30个自然日 + 余量（时区差）


def _days_ago(n: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()


def collect(store=None) -> List[Document]:
    """抓 RSS 列表 + 每条全文。传入 store 时跳过已入库 URL（增量采集不重复抓全文）。"""
    entries = []
    for page in range(1, MAX_PAGES + 1):
        url = FEED if page == 1 else f"{FEED}?paged={page}"
        try:
            r = fetch.fetch(url, headers={"Accept": "application/rss+xml"})
        except fetch.FetchError as e:
            log.warning("RSS 第%d页失败，停止翻页: %s", page, e)
            break
        parsed = feedparser.parse(r.text)
        if not parsed.entries:
            break
        entries.extend(parsed.entries)
        log.info("RSS 第%d页: %d 条（累计 %d）", page, len(parsed.entries), len(entries))
    log.info("RSS 共 %d 条", len(entries))

    # 时间窗过滤 -> 按发布时间倒序（缺日期的排最后）-> 截取 MAX_DOCS 条抓全文
    dated, undated = [], []
    for e in entries:
        if not e.get("link"):
            continue
        published = _parse_date(e.get("published", ""))
        if published and published < _days_ago(DAYS_WINDOW):
            continue
        (dated if published else undated).append((published, e))
    dated.sort(key=lambda x: x[0], reverse=True)
    selected = dated[:MAX_DOCS] + undated[:max(MAX_DOCS - len(dated), 0)]
    log.info("时间窗内 %d 条（有日期 %d），截取 %d 条抓全文", len(dated) + len(undated), len(dated), len(selected))

    docs = []
    for published, e in selected:
        url = e.get("link", "")
        if store is not None and store.has_doc(fetch.doc_id_for(url)):
            continue  # 已入库，跳过全文抓取
        title = e.get("title", "").strip()
        try:
            html = fetch.fetch(url).text
        except fetch.FetchError as err:
            log.warning("全文抓取失败 %s: %s", url, err)
            continue
        body = extract_text(html, url)
        if len(body) < 120:      # 抽不出正文的（付费墙/JS页）跳过，计入 fetched 不计 kept
            log.info("正文过短，跳过: %s", url)
            continue
        text, lang = clean_document_text(title, body)
        certain = bool(published)
        docs.append(Document(
            doc_id=fetch.doc_id_for(url), source_type="news", source_name="mining.com",
            url=url, title=title, published_at=published or _days_ago(0),
            lang=lang, text=text, time_certain=certain,
            extra={"summary": e.get("summary", "")[:200]}))
    return docs


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_RFC822 = re.compile(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})")


def _parse_date(s: str) -> str:
    m = _RFC822.search(s or "")
    if not (m and m.group(2) in _MONTHS):
        return ""
    return f"{m.group(3)}-{_MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
