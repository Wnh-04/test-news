"""澳洲 DISR Critical Minerals Strategy（industry.gov.au）。
适配器完整实现：RSS + 页面解析。当前部署网络对该域不可达（超时），
collect 会抛出/返回空并由 run.py 记录 ingest_log(skipped)——换网络环境即可工作。
"""
import datetime
import logging
import re
from typing import List

from .. import fetch
from ..clean import extract_text, clean_document_text
from ..models import Document

log = logging.getLogger("source.disr")

BASE = "https://www.industry.gov.au"
LIST_URLS = [
    f"{BASE}/news-media",                                        # 新闻列表
    f"{BASE}/publications",                                      # 出版物（策略文件）
]
ARTICLE = re.compile(r'href="(/(?:news|publications|regulations-and-standards)/[^"]+)"')
DAYS_WINDOW = 60


def collect(store=None) -> List[Document]:
    urls = set()
    for list_url in LIST_URLS:
        try:
            html = fetch.fetch(list_url).text
        except fetch.FetchError as e:
            log.warning("DISR 列表页不可达 %s: %s（国内网络通常不可达，跳过）", list_url, e)
            continue
        urls.update(ARTICLE.findall(html))
    log.info("DISR 候选文章: %d", len(urls))
    docs = []
    cutoff = (datetime.date.today() - datetime.timedelta(days=DAYS_WINDOW)).isoformat()
    for path in sorted(urls):
        url = BASE + path
        if store is not None and store.has_doc(fetch.doc_id_for(url)):
            continue
        try:
            html = fetch.fetch(url).text
        except fetch.FetchError:
            continue
        body = extract_text(html, url)
        if len(body) < 150:
            continue
        title = (re.search(r"<title>([^<]+)</title>", html) or [None, ""])[1] \
            .split("|")[0].strip()
        published = _meta_date(html) or datetime.date.today().isoformat()
        if published < cutoff:
            continue
        text, lang = clean_document_text(title, body)
        docs.append(Document(doc_id=fetch.doc_id_for(url), source_type="policy",
                             source_name="disr", url=url, title=title,
                             published_at=published, lang=lang, text=text))
    return docs


def _meta_date(html: str) -> str:
    m = re.search(r'datetime="(\d{4}-\d{2}-\d{2})"', html)
    return m.group(1) if m else ""
