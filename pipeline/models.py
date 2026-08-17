"""统一数据模型：三类源最终都规范化为 Document（文本，入向量库）或 PriceRow（结构化行情）。"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Document:
    """文本类文档（新闻 / 政策 / 价格日报摘要）。主键 doc_id = sha256(归一化URL) 前16位。"""
    doc_id: str
    source_type: str          # news | policy | price
    source_name: str          # mining.com | smm | chinare | disr | price_feed
    url: str
    title: str
    published_at: str         # ISO 8601 日期；解析失败用抓取日兜底
    lang: str                 # zh | en
    text: str                 # 清洗后的正文（分块在入库阶段做）
    time_certain: bool = True # published_at 是否来自原文（False=抓取日兜底）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriceRow:
    """结构化行情行。主键 (symbol, trade_date)。"""
    symbol: str               # LME_CU | LME_ZN | LME_NI | SHFE_LC | IRON_ORE
    trade_date: str           # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    unit: str
    source: str               # sina_global | sina_inner
    change_pct: Optional[float] = None  # 日涨跌幅%，入库时按前收盘补算

    def to_dict(self) -> dict:
        return asdict(self)


# 品种元数据：符号 -> 名称 / 单位 / 数据源
SYMBOLS = {
    "LME_CU":   {"name": "LME铜",   "unit": "美元/吨", "sina": "CAD"},
    "LME_ZN":   {"name": "LME锌",   "unit": "美元/吨", "sina": "ZSD"},
    "LME_NI":   {"name": "LME镍",   "unit": "美元/吨", "sina": "NID"},
    "SHFE_LC":  {"name": "碳酸锂主力", "unit": "元/吨",  "sina": "LC0"},   # 广期所
    "IRON_ORE": {"name": "铁矿石主力", "unit": "元/吨",  "sina": "I0"},    # 大商所
}
