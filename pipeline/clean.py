"""清洗层：HTML -> 正文抽取（trafilatura，readability 思路兜底）、语言判定、分块。"""
import logging
import re
from typing import List, Tuple

import trafilatura

log = logging.getLogger("clean")

_BLOCK = re.compile(r"[一-鿿]")
_WS = re.compile(r"\s+")


def extract_text(html: str, url: str = "") -> str:
    """从 HTML 抽正文。trafilatura 主路；失败时退化为标签剥离 + 文本密度启发。"""
    text = trafilatura.extract(html, url=url, include_comments=False,
                               include_tables=True, favor_recall=True)
    if text and len(text) >= 80:
        return _normalize(text)
    return _fallback_strip(html)


def _fallback_strip(html: str) -> str:
    """极简兜底：去 script/style/标签，压缩空白。正文短于 80 字符时返回空。"""
    html = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    text = _normalize(re.sub(r"&[a-z]+;", " ", html))
    return text if len(text) >= 80 else ""


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def detect_lang(text: str, title: str = "") -> str:
    """中英判定：CJK 字符占比 > 15% 视为中文（矿业语料混有英文代码/单位）。"""
    sample = (title * 3 + " " + text)[:2000]
    if not sample:
        return "zh"
    cjk = len(_BLOCK.findall(sample))
    return "zh" if cjk / max(len(sample), 1) > 0.15 else "en"


def chunk_text(text: str, chunk_chars: int = 480, overlap: int = 80,
               min_chars: int = 60) -> List[str]:
    """按字符滑窗分块（中英混排按字符近似 token 长度）。尾部过短并入前块。"""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chars else []
    chunks, start, step = [], 0, chunk_chars - overlap
    while start < len(text):
        piece = text[start:start + chunk_chars].strip()
        if len(piece) >= min_chars:
            chunks.append(piece)
        start += step
    return chunks


def clean_document_text(title: str, body: str) -> Tuple[str, str]:
    """正文 = 标题 + 正文（embedding 时标题权重前置），返回 (final_text, lang)。"""
    body = _normalize(body)
    text = f"{title}。{body}" if title else body
    return text, detect_lang(body, title)
