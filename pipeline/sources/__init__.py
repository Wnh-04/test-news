"""采集源注册表。run.py 按此顺序执行，每个源独立记录 ingest_log，单源失败不影响整体。"""
from . import mining_news, policy_sources, price_sources, disr


def news_sources():
    return [("mining.com", mining_news.collect)]


def policy_sources_():
    return [
        ("gov.cn", policy_sources.collect_gov),
        ("smm", policy_sources.collect_smm),
        ("disr", disr.collect),
    ]


def price_source():
    return [("price_feed", price_sources.collect)]
