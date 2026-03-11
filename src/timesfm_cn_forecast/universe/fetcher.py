"""
指数成份股拉取模块。

内置逻辑指数名 → AkShare 查询 code(s) 的映射，并使用 AkShare 的
index_stock_cons 接口拉取各指数的成份股，统一规范化输出格式。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# 逻辑名 → AkShare code(s) 映射
# 每个 entry 可以是单个 code str 或 list of codes（多个 code 的结果会合并）
# prefix_filter: 要过滤掉的股票代码前缀
INDEX_MAP: dict[str, dict] = {
    "HS300": {
        "codes": ["000300"],
        "prefix_filter": [],
        "description": "沪深300"
    },
    "ZZ500": {
        "codes": ["399905"],
        "prefix_filter": [],
        "description": "中证500"
    },
    "ZZ800": {
        "codes": ["399906"],
        "prefix_filter": [],
        "description": "中证800"
    },
    "CYBZ": {
        "codes": ["399006"],
        "prefix_filter": [],
        "description": "创业板指"
    },
    "ZXBZ": {
        "codes": ["399005"],
        "prefix_filter": [],
        "description": "中小板指"
    },
    "small": {
        "codes": ["399101"],
        "prefix_filter": ["68", "4", "8"],   # 过滤科创板、北交所
        "description": "中小盘综指（过滤科创/北交所）"
    },
    "A": {
        "codes": ["000002", "399107"],
        "prefix_filter": ["68", "4", "8"],   # 过滤科创板、北交所
        "description": "全A（沪深，过滤科创/北交所）"
    },
    "AA": {
        "codes": ["000985"],
        "prefix_filter": ["3", "68", "4", "8"],  # 额外过滤创业板
        "description": "全市场综指（过滤创业板/科创/北交所）"
    },
    "small_fengzhi": {
        "codes": ["000002", "399107"],
        "prefix_filter": ["3", "68", "4", "8"],  # 过滤创业板、科创、北交所
        "description": "自定义中小盘（沪深主板+中小板，过滤创业板/科创/北交所）"
    },
}


def _normalize_code(raw_code: str) -> str:
    """将股票代码规范化为 6 位纯数字字符串。"""
    return str(raw_code).strip().zfill(6)


def _fetch_single(akshare_code: str) -> pd.DataFrame:
    """从 AkShare 拉取单个指数的成份股。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError("请先安装 akshare: pip install akshare") from e

    df = ak.index_stock_cons(symbol=akshare_code)
    df = df.rename(columns={
        "品种代码": "code",
        "品种名称": "name",
        "纳入日期": "in_date",
    })
    df["code"] = df["code"].apply(_normalize_code)
    df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce").dt.date
    df["akshare_code"] = akshare_code
    return df[["akshare_code", "code", "name", "in_date"]]


def fetch_constituents(index_symbol: str) -> pd.DataFrame:
    """
    拉取指定逻辑指数的所有成份股，规范化后返回 DataFrame。

    Args:
        index_symbol: 逻辑指数代号，如 'CYBZ', 'HS300'。

    Returns:
        DataFrame with columns: [index_symbol, akshare_code, code, name, in_date, fetched_at]
    """
    if index_symbol not in INDEX_MAP:
        valid = list(INDEX_MAP.keys())
        raise ValueError(f"不支持的指数: {index_symbol}，可用选项: {valid}")

    cfg = INDEX_MAP[index_symbol]
    all_frames = []

    for akcode in cfg["codes"]:
        logger.info(f"  正在拉取 {index_symbol} -> AkShare code: {akcode}...")
        try:
            df = _fetch_single(akcode)
            all_frames.append(df)
        except Exception as err:
            logger.warning(f"  拉取 {akcode} 失败: {err}")

    if not all_frames:
        raise RuntimeError(f"所有 AkShare 接口均拉取失败：{index_symbol}")

    merged = pd.concat(all_frames, ignore_index=True)

    # 按 code 去重（多 codes 合并时可能重复）
    merged = merged.drop_duplicates(subset=["code"])

    # 过滤不需要的股票前缀
    prefix_filter = cfg.get("prefix_filter", [])
    if prefix_filter:
        mask = merged["code"].apply(
            lambda c: not any(c.startswith(p) for p in prefix_filter)
        )
        before = len(merged)
        merged = merged[mask].reset_index(drop=True)
        logger.info(f"  前缀过滤: {before} -> {len(merged)} 只（过滤前缀 {prefix_filter}）")

    merged["index_symbol"] = index_symbol
    merged["fetched_at"] = datetime.now(tz=timezone.utc)

    logger.info(f"  [{index_symbol}] 拉取完成，共 {len(merged)} 只成份股。")
    return merged[["index_symbol", "akshare_code", "code", "name", "in_date", "fetched_at"]]
