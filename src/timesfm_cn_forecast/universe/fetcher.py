"""
指数成份股拉取模块。

支持三种数据来源:
  - "akshare": 通过 AkShare 的 index_stock_cons 接口拉取
  - "industry_csv": 从本地 industry_category.csv 按申万三级行业名过滤
  - "concept_csv": 从本地 concept_category.csv 按概念名过滤
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 默认本地 CSV 路径（相对于项目根目录）
_DEFAULT_INDUSTRY_CSV = "data/industry_category.csv"
_DEFAULT_CONCEPT_CSV = "data/concept_category.csv"

# ---------------------------------------------------------------------------
# INDEX_MAP
# 每条记录必须包含:
#   source: "akshare" | "industry_csv" | "concept_csv"
#   description: 可读说明
# AkShare 类型额外需要:
#   codes: List[str]               AkShare 查询代码列表
#   prefix_filter: List[str]       要过滤的股票代码前缀
# CSV 类型额外需要:
#   category: str                  CSV 中的分类名（精确匹配）
#   prefix_filter: List[str]       要过滤的股票代码前缀
# ---------------------------------------------------------------------------
INDEX_MAP: dict[str, dict] = {
    # ── AkShare 宽基指数 ──────────────────────────────────────────────────────
    "HS300": {
        "source": "akshare",
        "codes": ["000300"],
        "prefix_filter": [],
        "description": "沪深300",
    },
    "ZZ500": {
        "source": "akshare",
        "codes": ["399905"],
        "prefix_filter": [],
        "description": "中证500",
    },
    "ZZ800": {
        "source": "akshare",
        "codes": ["399906"],
        "prefix_filter": [],
        "description": "中证800",
    },
    "CYBZ": {
        "source": "akshare",
        "codes": ["399006"],
        "prefix_filter": [],
        "description": "创业板指",
    },
    "ZXBZ": {
        "source": "akshare",
        "codes": ["399005"],
        "prefix_filter": [],
        "description": "中小板指",
    },
    "small": {
        "source": "akshare",
        "codes": ["399101"],
        "prefix_filter": ["68", "4", "8"],
        "description": "中小盘综指（过滤科创/北交所）",
    },
    "small_25": {
        "source": "akshare",
        "codes": ["000002", "399107"],
        "prefix_filter": [],
        "limit": 50,
        "description": "自定义 small_25（临时：未按市值筛选，仅取前 50）",
    },
    "A": {
        "source": "akshare",
        "codes": ["000002", "399107"],
        "prefix_filter": ["68", "4", "8"],
        "description": "全A（沪深，过滤科创/北交所）",
    },
    "AA": {
        "source": "akshare",
        "codes": ["000985"],
        "prefix_filter": ["3", "68", "4", "8"],
        "description": "全市场综指（过滤创业板/科创/北交所）",
    },
    "small_fengzhi": {
        "source": "akshare",
        "codes": ["000002", "399107"],
        "prefix_filter": ["3", "68", "4", "8"],
        "description": "自定义中小盘（沪深主板，过滤创业板/科创/北交所）",
    },
    # ── 申万三级行业分组（来自 industry_category.csv） ────────────────────────
    "ind_消费电子": {
        "source": "industry_csv",
        "category": "消费电子零部件及组装III",
        "prefix_filter": [],
        "description": "申万行业：消费电子零部件及组装",
    },
    "ind_军工电子": {
        "source": "industry_csv",
        "category": "军工电子III",
        "prefix_filter": [],
        "description": "申万行业：军工电子",
    },
    "ind_芯片": {
        "source": "industry_csv",
        "category": "数字芯片设计III",
        "prefix_filter": [],
        "description": "申万行业：数字芯片设计",
    },
    "ind_IT服务": {
        "source": "industry_csv",
        "category": "IT服务III",
        "prefix_filter": [],
        "description": "申万行业：IT服务",
    },
    "ind_医疗耗材": {
        "source": "industry_csv",
        "category": "医疗耗材III",
        "prefix_filter": [],
        "description": "申万行业：医疗耗材",
    },
    "ind_化学制剂": {
        "source": "industry_csv",
        "category": "化学制剂III",
        "prefix_filter": [],
        "description": "申万行业：化学制剂",
    },
    "ind_汽车底盘": {
        "source": "industry_csv",
        "category": "底盘与发动机系统III",
        "prefix_filter": [],
        "description": "申万行业：底盘与发动机系统",
    },
    "ind_化工": {
        "source": "industry_csv",
        "category": "其他化学制品III",
        "prefix_filter": [],
        "description": "申万行业：其他化学制品",
    },
    # ── 概念分组（来自 concept_category.csv）──────────────────────────────────
    "con_低空经济": {
        "source": "concept_csv",
        "category": "低空经济",
        "prefix_filter": [],
        "description": "概念：低空经济",
    },
    "con_比亚迪链": {
        "source": "concept_csv",
        "category": "比亚迪概念",
        "prefix_filter": [],
        "description": "概念：比亚迪产业链",
    },
    "con_汽车零部件": {
        "source": "concept_csv",
        "category": "汽车零部件概念",
        "prefix_filter": [],
        "description": "概念：汽车零部件",
    },
    "con_军民融合": {
        "source": "concept_csv",
        "category": "军民融合",
        "prefix_filter": [],
        "description": "概念：军民融合",
    },
    "con_信创": {
        "source": "concept_csv",
        "category": "信创",
        "prefix_filter": [],
        "description": "概念：信创（国产软件）",
    },
    "con_氢能源": {
        "source": "concept_csv",
        "category": "氢能源",
        "prefix_filter": [],
        "description": "概念：氢能源",
    },
    "con_新能源": {
        "source": "concept_csv",
        "category": "新能源",
        "prefix_filter": [],
        "description": "概念：新能源",
    },
}


# ---------------------------------------------------------------------------
# 数据源处理函数
# ---------------------------------------------------------------------------

def _normalize_code_6digit(raw_code: str) -> str:
    """将 AkShare 格式的 6 位纯数字代码规范化。"""
    return str(raw_code).strip().zfill(6)


def _xshare_to_dbsymbol(code: str) -> tuple[str, str]:
    """
    将 CSV 中的 '000001.XSHE' / '600001.XSHG' 格式转换为:
      - db_symbol: 'sz000001' / 'sh600001'（用于 market.duckdb 查询）
      - pure_code: '000001'（6 位纯数字）
    """
    parts = code.split(".")
    if len(parts) != 2:
        pure = str(code).strip().zfill(6)
        return pure, pure
    num, exchange = parts[0].strip(), parts[1].strip().upper()
    prefix = "sz" if exchange == "XSHE" else "sh"
    return prefix + num, num


def _fetch_from_akshare(index_symbol: str, cfg: dict) -> pd.DataFrame:
    """通过 AkShare 接口拉取宽基指数成份股。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError("请先安装 akshare: pip install akshare") from e

    all_frames = []
    for akcode in cfg["codes"]:
        logger.info(f"  [{index_symbol}] AkShare code: {akcode}...")
        try:
            df = ak.index_stock_cons(symbol=akcode)
            df = df.rename(columns={"品种代码": "code", "品种名称": "name", "纳入日期": "in_date"})
            df["code"] = df["code"].apply(_normalize_code_6digit)
            df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce").dt.date
            df["akshare_code"] = akcode
            all_frames.append(df[["akshare_code", "code", "name", "in_date"]])
        except Exception as err:
            logger.warning(f"  [{index_symbol}] 拉取 {akcode} 失败: {err}")

    if not all_frames:
        raise RuntimeError(f"所有 AkShare 接口均失败: {index_symbol}")
    return pd.concat(all_frames, ignore_index=True)


def _fetch_from_csv(
    index_symbol: str,
    cfg: dict,
    csv_path: str,
    code_col: str = "code",
    name_col: str | None = None,
    category_col: str = "category",
) -> pd.DataFrame:
    """从本地 CSV 按 category 名过滤成份股。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    cat_value = cfg["category"]
    df_all = pd.read_csv(path)
    df = df_all[df_all[category_col] == cat_value].copy()

    if df.empty:
        logger.warning(f"  [{index_symbol}] 在 {csv_path} 中未找到分类: {cat_value}")
        return pd.DataFrame(columns=["akshare_code", "code", "name", "in_date"])

    # 转换代码格式
    db_symbols, pure_codes = zip(*df[code_col].apply(_xshare_to_dbsymbol))
    df["code"] = list(pure_codes)
    df["akshare_code"] = cat_value      # 这里用 category 名作为 akshare_code 字段的占位
    df["in_date"] = None
    df["name"] = df[name_col].fillna("") if name_col and name_col in df.columns else ""

    logger.info(f"  [{index_symbol}] 从 CSV 读取 {len(df)} 条，分类: {cat_value}")
    return df[["akshare_code", "code", "name", "in_date"]]


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

def fetch_constituents(
    index_symbol: str,
    industry_csv: str = _DEFAULT_INDUSTRY_CSV,
    concept_csv: str = _DEFAULT_CONCEPT_CSV,
) -> pd.DataFrame:
    """
    拉取指定逻辑指数/分组的所有成份股，规范化后返回 DataFrame。

    Args:
        index_symbol: 逻辑分组代号，如 'CYBZ', 'ind_消费电子', 'con_低空经济'。
        industry_csv: industry_category.csv 路径。
        concept_csv: concept_category.csv 路径。

    Returns:
        DataFrame with columns: [index_symbol, akshare_code, code, name, in_date, fetched_at]
    """
    if index_symbol not in INDEX_MAP:
        valid = list(INDEX_MAP.keys())
        raise ValueError(f"不支持的分组: {index_symbol}，可用选项: {valid}")

    cfg = INDEX_MAP[index_symbol]
    source = cfg.get("source", "akshare")

    if source == "akshare":
        merged = _fetch_from_akshare(index_symbol, cfg)
    elif source == "industry_csv":
        merged = _fetch_from_csv(
            index_symbol, cfg, industry_csv,
            code_col="code", name_col=None, category_col="category"
        )
    elif source == "concept_csv":
        merged = _fetch_from_csv(
            index_symbol, cfg, concept_csv,
            code_col="code", name_col="name", category_col="category"
        )
    else:
        raise ValueError(f"未知 source 类型: {source}")

    # 去重
    merged = merged.drop_duplicates(subset=["code"]).reset_index(drop=True)

    # 前缀过滤
    prefix_filter = cfg.get("prefix_filter", [])
    if prefix_filter:
        before = len(merged)
        merged = merged[
            merged["code"].apply(lambda c: not any(c.startswith(p) for p in prefix_filter))
        ].reset_index(drop=True)
        logger.info(f"  [{index_symbol}] 前缀过滤: {before} -> {len(merged)} 只")

    # 可选：限制数量（如 small_25 临时取前 50）
    limit = cfg.get("limit")
    if isinstance(limit, int) and limit > 0:
        before = len(merged)
        merged = merged.head(limit).reset_index(drop=True)
        logger.info(f"  [{index_symbol}] 限制数量: {before} -> {len(merged)} 只")

    merged["index_symbol"] = index_symbol
    merged["fetched_at"] = datetime.now(tz=timezone.utc)
    logger.info(f"  [{index_symbol}] 完成，共 {len(merged)} 只。")
    return merged[["index_symbol", "akshare_code", "code", "name", "in_date", "fetched_at"]]
