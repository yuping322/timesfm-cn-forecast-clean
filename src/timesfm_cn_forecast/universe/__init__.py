"""
指数成份股宇宙模块 (Index Stock Universe)

提供从 AkShare 拉取各类指数成份股并持久化至 DuckDB 的能力。

Usage:
    from timesfm_cn_forecast.universe import get_stock_universe
    stocks = get_stock_universe('CYBZ', duckdb_path='data/index_market.duckdb')
    # -> ['300015', '300024', ...]
"""
from .fetcher import fetch_constituents, INDEX_MAP
from .storage import upsert_constituents, query_constituents

__all__ = [
    "fetch_constituents",
    "query_constituents",
    "upsert_constituents",
    "get_stock_universe",
    "INDEX_MAP",
]


def get_stock_universe(index_symbol: str, duckdb_path: str) -> list[str]:
    """
    从 DuckDB 获取指定指数最新成份股代码列表。

    Args:
        index_symbol: 逻辑指数代号，如 'CYBZ', 'HS300', 'ZZ500'。
        duckdb_path: index_market.duckdb 文件路径。

    Returns:
        股票代码列表（6位纯数字）。
    """
    return query_constituents(index_symbol, duckdb_path)
