"""源2 关键矿产政策（组合策略，DATA_NOTES.md 中如实记录各子源贡献）：

  a) 国务院政策文件库 gov.cn —— 权威政策全文，多关键词轮询（稀土/矿产/锂/铜/镍…）
  b) 中国稀土集团官网 chinare.com.cn —— 公司新闻/动态
  c) SMM 上海有色资讯 —— 行业政策动态（首页文章列表，SSR）
  d) 澳洲 DISR Critical Minerals Strategy —— 适配器完整实现；
     当前部署网络不可达时自动跳过并写 ingest_log（换网络环境即可工作）

原始题目只给了 a/b/d 两个建议源，但稀土集团发文频次低（周更）、DISR 在国内
网络不可达，为保证"≥200条/源"的量级，组合了 gov.cn 与 SMM，均属于关键矿产
政策/监管范畴。
"""
import datetime
import json
import logging
import re
from typing import List

from .. import fetch
from ..clean import extract_text, clean_document_text
from ..models import Document

log = logging.getLogger("source.policy")

GOV_SEARCH = ("https://sousuo.www.gov.cn/search-gov/data?t=zhengcelibrary_gw"
              "&q={q}&timetype=timeqb&mintime=&maxtime=&sort=pubtime&sortType=1"
              "&searchfield=title&p={p}&n=20&filetype=&childtype=&subchildtype=&tsbq="
              "&pubtimeyear=&puborg=&pcodeYear=&pcodeNum=&homeflag=true&orpro=&dup=&pro=&subsite=")
GOV_KEYWORDS = ["稀土", "矿产", "锂", "铜", "镍", "铁矿石", "矿业", "有色金属",
                "钨", "钼", "锰", "煤炭", "黄金", "战略性矿产", "地质勘查"]
GOV_REFERER = ("https://sousuo.www.gov.cn/sousuo/search.shtml?code=17da70961a7"
               "&searchWord=%E7%A8%80%E5%9C%9F&dataTypeId=107&sign=")

CHINARE_LIST = "https://www.chinare.com.cn/zhzjt/441147/gsxw/index.html"
CHINARE_ARTICLE = re.compile(r'href="(/zhzjt/441147/gsxw/\d+/index\.html)"')

SMM_HOME = "https://news.smm.cn/"
SMM_ARTICLE = re.compile(r'href="(?:https://news\.smm\.cn)?(/news/\d+)"')


def _today() -> str:
    return datetime.date.today().isoformat()


# ---------------- a) gov.cn 政策库 ----------------
def collect_gov(store=None) -> List[Document]:
    docs, seen = [], set()
    for kw in GOV_KEYWORDS:
        for page in (1, 2, 3):
            try:
                r = fetch.fetch(GOV_SEARCH.format(q=_quote(kw), p=page),
                                headers={"Referer": GOV_REFERER, "Accept": "application/json"})
                data = json.loads(r.text)
            except (fetch.FetchError, json.JSONDecodeError) as e:
                log.warning("gov.cn 检索失败 kw=%s: %s", kw, e)
                continue
            items = ((data.get("searchVO") or {}).get("listVO")) or []
            log.info("gov.cn kw=%s p=%d: %d 条", kw, page, len(items))
            for it in items:
                url = it.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                docs.extend(_doc_from_page(
                    url=url, source_name="gov.cn", store=store, title=_strip_em(it.get("title", "")),
                    published=(it.get("pubtimeStr", "") or "")[:10].replace("/", "-") or None))
    return docs


# ---------------- b) 稀土集团官网 ----------------
def collect_chinare(store=None) -> List[Document]:
    try:
        html = fetch.fetch(CHINARE_LIST).text
    except fetch.FetchError as e:
        log.error("稀土集团列表页失败: %s", e)
        return []
    urls = sorted(set(CHINARE_ARTICLE.findall(html)))
    log.info("稀土集团列表: %d 篇", len(urls))
    docs = []
    for path in urls:
        url = f"https://www.chinare.com.cn{path}"
        m = re.search(r"/(\d{4})(\d{2})(\d{2})", path)   # URL 内嵌时间戳 20260424…
        published = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        docs.extend(_doc_from_page(url, "chinare", None, published, store=store))
        if published:
            # 已入库的更新日期（首次入库时日期解析失败的修正）
            _backfill_date(store, fetch.doc_id_for(url), published)
    return docs


def _backfill_date(store, doc_id: str, published: str):
    """首次抓取时日期没解析出来、用抓取日兜底的文档，补回真实发布日期。"""
    try:
        store.conn.execute("UPDATE documents SET published_at=?, time_certain=1 "
                           "WHERE doc_id=? AND time_certain=0", (published, doc_id))
    except Exception:
        pass


# ---------------- c) SMM 资讯 ----------------
def collect_smm(store=None) -> List[Document]:
    try:
        html = fetch.fetch(SMM_HOME).text
    except fetch.FetchError as e:
        log.error("SMM 列表页失败: %s", e)
        return []
    urls = sorted(set(SMM_ARTICLE.findall(html)))
    log.info("SMM 列表: %d 篇", len(urls))
    docs = []
    for path in urls:
        url = f"https://news.smm.cn{path}"
        docs.extend(_doc_from_page(url, "smm", None, None, store=store))
    return docs


# ---------------- 公共：抓详情页 -> Document ----------------
def _doc_from_page(url: str, source_name: str, title: str = None,
                   published: str = None, store=None) -> List[Document]:
    if store is not None and store.has_doc(fetch.doc_id_for(url)):
        return []          # 已入库，增量采集跳过
    try:
        html = fetch.fetch(url).text
    except fetch.FetchError as e:
        log.warning("详情页失败 %s: %s", url, e)
        return []
    body = extract_text(html, url)
    if len(body) < 100:
        return []
    title = (title or _og_title(html) or "").strip()
    text, lang = clean_document_text(title, body)
    return [Document(
        doc_id=fetch.doc_id_for(url), source_type="policy", source_name=source_name,
        url=url, title=title, published_at=published or _today(), lang=lang, text=text,
        time_certain=bool(published))]


def _og_title(html: str) -> str:
    m = re.search(r'<meta[^>]+(?:og:title|property="og:title")[^>]+content="([^"]+)"', html) \
        or re.search(r"<title>([^<]+)</title>", html)
    return (m.group(1).split("__")[0].strip() if m else "")


def _strip_em(s: str) -> str:
    return re.sub(r"</?em>", "", s).strip()


def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s)
