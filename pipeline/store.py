"""存储层：SQLite + 向量 BLOB。

documents   文本元数据，主键 doc_id
chunks      分块与 embedding（float32 BLOB）
prices      结构化行情，主键 (symbol, trade_date)
ingest_log  采集运行日志

规模在数千 chunk 级，检索用 numpy 暴力余弦；量级上来后把向量部分换成
Qdrant 即可，对外接口语义不变。
"""
import functools
import json
import os
import sqlite3
import threading
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import Document, PriceRow

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mining.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_name TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT NOT NULL,
  published_at TEXT NOT NULL,
  lang TEXT NOT NULL,
  time_certain INTEGER NOT NULL DEFAULT 1,
  simhash INTEGER NOT NULL,
  text TEXT NOT NULL,
  extra TEXT,
  ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_pub ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_doc_src ON documents(source_type, source_name);

CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY,          -- doc_id:idx
  doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL,         -- chunk 级去重（第三层）
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS prices(
  symbol TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL,
  unit TEXT, source TEXT, change_pct REAL,
  PRIMARY KEY(symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS ingest_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  source TEXT NOT NULL,
  status TEXT NOT NULL,               -- ok | skipped | failed
  fetched INTEGER NOT NULL,
  kept INTEGER NOT NULL,
  note TEXT
);
"""


def _locked(fn):
    """公开方法统一加锁：服务进程多线程访问 SQLite 时串行化（采集单线程时锁无开销）。"""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class Store:
    def __init__(self, path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread=False：FastAPI 同步端点跑在线程池，连接需跨线程，
        # 由 _locked 串行化
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")

    # ---------- documents ----------
    @_locked
    def has_doc(self, doc_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    @_locked
    def upsert_document(self, doc: Document, fingerprint: int):
        self.conn.execute(
            "INSERT INTO documents(doc_id,source_type,source_name,url,title,published_at,lang,"
            "time_certain,simhash,text,extra,ingested_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(doc_id) DO UPDATE SET title=excluded.title, text=excluded.text,"
            "simhash=excluded.simhash, ingested_at=excluded.ingested_at",
            (doc.doc_id, doc.source_type, doc.source_name, doc.url, doc.title, doc.published_at,
             doc.lang, int(doc.time_certain), _to_signed64(fingerprint), doc.text,
             json.dumps(doc.extra, ensure_ascii=False), _now()))

    @_locked
    def all_fingerprints(self) -> List[Tuple[str, int]]:
        return [(r["doc_id"], _from_signed64(r["simhash"])) for r in self.conn.execute(
            "SELECT doc_id, simhash FROM documents WHERE source_type != 'price'")]

    @_locked
    def count(self, source_type: Optional[str] = None) -> int:
        if source_type:
            return self.conn.execute("SELECT COUNT(*) c FROM documents WHERE source_type=?",
                                     (source_type,)).fetchone()["c"]
        return self.conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]

    # ---------- chunks ----------
    @_locked
    def replace_chunks(self, doc_id: str, chunk_texts: Sequence[str], embeddings: Sequence):
        """chunk 级去重：同 content_hash 的块只保留一次（防样板文字重复入向量库）。"""
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        seen = set()
        for i, (txt, emb) in enumerate(zip(chunk_texts, embeddings)):
            h = _content_hash(txt)
            if h in seen:
                continue
            seen.add(h)
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks(chunk_id,doc_id,idx,text,content_hash,embedding) "
                "VALUES(?,?,?,?,?,?)",
                (f"{doc_id}:{i}", doc_id, i, txt, h,
                 emb.astype(np.float32).tobytes() if emb is not None else None))

    @_locked
    def load_chunk_rows(self, where: str = "1=1", params: tuple = ()) -> List[dict]:
        """检索层一次性载入内存（向量 + BM25 语料）。锁内物化，
        避免生成器把锁在迭代期间释放掉。"""
        q = ("SELECT c.chunk_id, c.doc_id, c.idx, c.text, c.embedding, d.source_type, d.source_name,"
             " d.title, d.url, d.published_at, d.lang FROM chunks c "
             "JOIN documents d ON d.doc_id = c.doc_id WHERE " + where)
        rows = []
        for r in self.conn.execute(q, params):
            emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
            rows.append(dict(r) | {"embedding": emb})
        return rows

    # ---------- prices ----------
    @_locked
    def upsert_prices(self, rows: Iterable[PriceRow]):
        self.conn.executemany(
            "INSERT INTO prices(symbol,trade_date,open,high,low,close,unit,source,change_pct) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,trade_date) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,"
            "change_pct=excluded.change_pct",
            [(p.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.unit, p.source, p.change_pct)
             for p in rows])

    @_locked
    def query_prices(self, symbol: str, date_from: str, date_to: str) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM prices WHERE symbol=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            (symbol, date_from, date_to))]

    @_locked
    def price_symbols(self) -> List[str]:
        return [r["symbol"] for r in self.conn.execute(
            "SELECT DISTINCT symbol FROM prices ORDER BY symbol")]

    # ---------- log ----------
    @_locked
    def log_run(self, source: str, status: str, fetched: int, kept: int, note: str = ""):
        self.conn.execute("INSERT INTO ingest_log(ts,source,status,fetched,kept,note) VALUES(?,?,?,?,?,?)",
                          (_now(), source, status, fetched, kept, note))

    @_locked
    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _content_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _to_signed64(n: int) -> int:
    """SQLite INTEGER 是有符号64位；SimHash 是无符号64位，存储前转换。"""
    return n - 2 ** 64 if n >= 2 ** 63 else n


def _from_signed64(n: int) -> int:
    return n + 2 ** 64 if n < 0 else n
