"""带浏览器头、限速、重试的 HTTP 客户端；以及 URL 归一化（去重第一层）。"""
import hashlib
import logging
import random
import re
import time
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from typing import Optional

import httpx

log = logging.getLogger("fetch")

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 每域名最低请求间隔（秒），带随机抖动，尊重源站频控
_MIN_INTERVAL = {}
_DEFAULT_INTERVAL = 2.0
_last_hit = {}


def set_rate_limit(host: str, seconds: float):
    _MIN_INTERVAL[host] = seconds


class FetchError(Exception):
    pass


def _throttle(host: str):
    interval = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    last = _last_hit.get(host, 0.0)
    wait = interval - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait + random.uniform(0, 0.5))
    _last_hit[host] = time.monotonic()


def fetch(url: str, *, headers: Optional[dict] = None, max_retries: int = 3,
          timeout: float = 25.0) -> httpx.Response:
    """GET + 限速 + 指数退避重试。4xx 不重试（除非 403/429），5xx/超时重试。"""
    host = urlsplit(url).netloc
    merged = {**BROWSER_HEADERS, **(headers or {})}
    last_exc = None
    for attempt in range(max_retries):
        _throttle(host)
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=merged) as c:
                r = c.get(url)
            if r.status_code < 400:
                return r
            if r.status_code in (403, 429) or r.status_code >= 500:
                last_exc = FetchError(f"HTTP {r.status_code} {url}")
            else:
                raise FetchError(f"HTTP {r.status_code} {url}")  # 4xx（非403/429）直接放弃
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = FetchError(f"{type(e).__name__} {url}: {e}")
        backoff = (2 ** attempt) + random.uniform(0, 1)
        log.warning("第%d次失败: %s，%.1fs 后重试", attempt + 1, last_exc, backoff)
        time.sleep(backoff)
    raise last_exc or FetchError(url)


_TRACKING_PREFIX = ("utm_", "spm", "from", "share", "source", "sid", "ref", "fbclid", "gclid")


def normalize_url(url: str) -> str:
    """URL 归一化：小写 host、去锚点、去跟踪参数、去尾斜杠差异 -> 用于稳定主键。"""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    host = parts.netloc.lower()
    path = re.sub(r"/+$", "", parts.path) or "/"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not any(k.lower().startswith(p) for p in _TRACKING_PREFIX)]
    query.sort()
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))


def doc_id_for(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]
