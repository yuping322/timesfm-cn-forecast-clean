#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史数据提供方。"""

from __future__ import annotations

import io
import os
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


@dataclass
class 数据请求:
    provider: str
    symbol: str | None = None
    start: str | None = None
    end: str | None = None
    input_csv: str | None = None
    input_parquet: str | None = None
    date_column: str = "date"
    value_column: str = "close"
    auto_fetch_akshare: bool = False
    oss_file_template: str = "{symbol}.csv"
    oss_date_column: str = "日期"
    oss_value_column: str = "close"
    tushare_field: str = "close"
    akshare_adjust: str = ""
    kline: bool = False


def 标准化代码(symbol: str, provider: str) -> str:
    """根据数据源类型标准化股票代码格式。"""
    s = symbol.lower()
    # 移除前缀以便计算
    pure_code = "".join(filter(str.isdigit, s)).zfill(6)
    
    if provider == "akshare":
        if s.startswith(("sh", "sz", "bj")):
            return s
        if pure_code.startswith("6"):
            return "sh" + pure_code
        elif pure_code.startswith(("0", "3")):
            return "sz" + pure_code
        elif pure_code.startswith(("4", "8")):
            return "bj" + pure_code
        return pure_code
        
    if provider == "tushare":
        s = symbol.upper()
        if s.endswith((".SH", ".SZ", ".BJ")):
            return s
        if pure_code.startswith("6"):
            return f"{pure_code}.SH"
        if pure_code.startswith(("0", "3")):
            return f"{pure_code}.SZ"
        if pure_code.startswith(("4", "8")):
            return f"{pure_code}.BJ"
        return pure_code
        
    return pure_code


def 批量加载历史数据(
    symbols: list[str],
    provider: str,
    start: str | None = None,
    end: str | None = None,
    **kwargs
) -> pd.DataFrame:
    """
    批量加载多只股票的历史数据，返回宽表格式 (Index=date, Columns=symbol, Values=close)。
    """
    frames = []
    for s in symbols:
        try:
            # 构造临时的请求对象以复用现有逻辑
            req = 数据请求(
                provider=provider,
                symbol=s,
                start=start,
                end=end,
                input_csv=kwargs.get("input_csv"),
                input_parquet=kwargs.get("input_parquet"),
                date_column=kwargs.get("date_column", "date"),
                value_column=kwargs.get("value_column", "close" if provider == "akshare" else "value"),
                auto_fetch_akshare=False,
                oss_file_template=kwargs.get("oss_file_template", "{symbol}.csv"),
                oss_date_column=kwargs.get("oss_date_column", "日期"),
                oss_value_column=kwargs.get("oss_value_column", "close"),
                tushare_field=kwargs.get("tushare_field", "close"),
                akshare_adjust=kwargs.get("akshare_adjust", ""),
                kline=kwargs.get("kline", False),
            )
            df = 加载历史数据(req)
            if not df.empty:
                df = df.rename(columns={"value": s})
                frames.append(df.set_index("date")[s])
        except Exception as e:
            print(f"加载股票 {s} 数据失败: {e}")
            continue
            
    if not frames:
        return pd.DataFrame()
        
    return pd.concat(frames, axis=1).sort_index()


def 加载历史数据(req: 数据请求) -> pd.DataFrame:
    if req.provider == "local":
        return 从本地加载(req)
    if req.provider == "oss":
        return 从OSS加载(req)
    if req.provider == "tushare":
        return 从Tushare加载(req)
    if req.provider == "akshare":
        return 从AkShare加载(req)
    raise ValueError(f"不支持的数据源: {req.provider}")


def 从本地加载(req: 数据请求) -> pd.DataFrame:
    if req.input_csv:
        df = pd.read_csv(req.input_csv)
    elif req.input_parquet:
        df = pd.read_parquet(req.input_parquet)
    elif req.auto_fetch_akshare:
        if not req.symbol:
            raise ValueError("local 自动拉取模式需要提供 --symbol，例如 600519")
        local_path = _自动拉取到本地(req)
        df = pd.read_csv(local_path)
    else:
        raise ValueError("local 模式必须提供 --input-csv 或 --input-parquet")
    
    if req.kline:
        # 尝试保留 OHLCV 常用列
        ohlc_cols = ["open", "high", "low", "close", "volume", "开盘", "最高", "最低", "收盘", "成交量"]
        date_col = req.date_column
        val_col = req.value_column
        available_cols = [c for c in ohlc_cols if c in df.columns and c != date_col]
        return _标准化输出(df, date_col, val_col, req.symbol, extra_cols=available_cols)

    return _标准化输出(df, req.date_column, req.value_column, req.symbol)


def 从OSS加载(req: 数据请求) -> pd.DataFrame:
    try:
        import oss2
    except ImportError as exc:
        raise ImportError("使用 oss 数据源前请安装 oss2") from exc

    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")
    bucket_name = os.environ.get("OSS_BUCKET")
    prefix = os.environ.get("OSS_PREFIX", "hangqing/daily_data/")

    if not all([key_id, key_secret, endpoint, bucket_name]):
        raise ValueError("OSS 环境变量不完整，至少需要 OSS_ACCESS_KEY_ID/SECRET/ENDPOINT/BUCKET")
    if not req.symbol:
        raise ValueError("oss 模式必须提供 --symbol")

    auth = oss2.Auth(key_id, key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    
    # 支持带前缀的标准化代码，如 sh600519
    formatted_symbol = 标准化代码(req.symbol, "akshare") 
    object_name = req.oss_file_template.format(symbol=formatted_symbol)
    object_path = f"{prefix.rstrip('/')}/{object_name}".lstrip("/")
    
    try:
        content = bucket.get_object(object_path).read()
    except Exception as e:
        # 尝试不带前缀的纯数字
        pure_symbol = "".join(filter(str.isdigit, req.symbol)).zfill(6)
        object_name = req.oss_file_template.format(symbol=pure_symbol)
        object_path = f"{prefix.rstrip('/')}/{object_name}".lstrip("/")
        content = bucket.get_object(object_path).read()

    if object_name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    else:
        # 支持多种中文表头
        df = pd.read_csv(io.BytesIO(content))
        
    date_col = req.oss_date_column
    if date_col not in df.columns and "日期" in df.columns:
        date_col = "日期"
        
    val_col = req.oss_value_column
    if val_col not in df.columns and "close" in df.columns:
        val_col = "close"
    elif val_col not in df.columns and "收盘" in df.columns:
        val_col = "收盘"

    if req.kline:
        ohlc_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        extra_cols = []
        for cn_name, en_name in ohlc_map.items():
            if cn_name in df.columns and cn_name not in [date_col, val_col]:
                extra_cols.append(cn_name)
            elif en_name in df.columns and en_name not in [date_col, val_col]:
                extra_cols.append(en_name)
        return _标准化输出(df, date_col, val_col, req.symbol, extra_cols=extra_cols)

    return _标准化输出(df, date_col, val_col, req.symbol)


def 从Tushare加载(req: 数据请求) -> pd.DataFrame:
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError("使用 tushare 数据源前请安装 tushare") from exc

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("缺少环境变量 TUSHARE_TOKEN")
    if not req.symbol:
        raise ValueError("tushare 模式必须提供 --symbol，例如 600519.SH")

    ts.set_token(token)
    pro = ts.pro_api()
    ts_code = 标准化代码(req.symbol, "tushare")
    df = pro.daily(ts_code=ts_code, start_date=_tushare_date(req.start), end_date=_tushare_date(req.end))
    if df.empty:
        raise ValueError(f"Tushare 未返回数据: {req.symbol}")
    
    if req.kline:
        # Tushare 默认返回: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        extra_cols = [c for c in ["open", "high", "low", "vol"] if c in df.columns]
        return _标准化输出(df, "trade_date", req.tushare_field, req.symbol, extra_cols=extra_cols)
        
    return _标准化输出(df, "trade_date", req.tushare_field, req.symbol)


def 从AkShare加载(req: 数据请求) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("使用 akshare 数据源前请安装 akshare") from exc

    if not req.symbol:
        raise ValueError("akshare 模式必须提供 --symbol，例如 600519")

    symbol_sina = 标准化代码(req.symbol, "akshare")
    start = _akshare_date(req.start)
    end = _akshare_date(req.end)

    # 尝试使用 stock_zh_a_daily (Sina 接口，支持旧代码)
    try:
        df = ak.stock_zh_a_daily(
            symbol=symbol_sina,
            start_date=start,
            end_date=end,
            adjust=req.akshare_adjust or "qfq",
        )
    except Exception:
        # 备选使用比较通用的接口
        pure_symbol = symbol_sina[-6:]
        df = ak.stock_zh_a_hist(
            symbol=pure_symbol,
            start_date=start or "20000101",
            end_date=end or pd.Timestamp.today().strftime("%Y%m%d"),
            adjust=req.akshare_adjust or "qfq",
            period="daily"
        )

    if df.empty:
        raise ValueError(f"AkShare 未返回数据: {req.symbol}")
        
    date_col = "date" if "date" in df.columns else ("日期" if "日期" in df.columns else df.columns[0])
    val_col = "close" if "close" in df.columns else ("收盘" if "收盘" in df.columns else "收盘价")
    
    if req.kline:
        # AkShare 常见列名转换为标准英文
        # 即使接口返回的是英文，extra_cols 也会在 _标准化输出中被正确识别
        ohlc_map = {"开盘": "open", "最高": "high", "最低": "low", "成交量": "volume"}
        extra_cols = []
        for cn, en in ohlc_map.items():
            if cn in df.columns:
                extra_cols.append(cn)
            elif en in df.columns:
                extra_cols.append(en)
        return _标准化输出(df, date_col, val_col, req.symbol, extra_cols=extra_cols)

    return _标准化输出(df, date_col, val_col, req.symbol)


def _标准化输出(
    df: pd.DataFrame, date_col: str, val_col: str, symbol: str | None, extra_cols: list[str] | None = None
) -> pd.DataFrame:
    # 转换为日期格式并重命名列
    df[date_col] = pd.to_datetime(df[date_col])
    
    rename_map = {date_col: "date", val_col: "value"}
    # 如果是 K 线模式，也尝试标准化 extra_cols 的名称
    std_map = {
        "开盘": "open", "最高": "high", "最低": "low", "成交量": "volume", "vol": "volume", "收盘": "close"
    }
    if extra_cols:
        for col in extra_cols:
            if col in std_map:
                rename_map[col] = std_map[col]

    df = df.rename(columns=rename_map)
    
    # 特殊处理：如果请求了 K 线，确保 open, high, low, close 都在
    # value 列作为主数值列（对应 TimesFM 输入），close 列作为 K 线展示列
    if "close" not in df.columns and "value" in df.columns:
        df["close"] = df["value"]
    elif "value" not in df.columns and "close" in df.columns:
        df["value"] = df["close"]
    
    target_cols = ["date", "value", "open", "high", "low", "close", "volume"]
    # 只保留存在的列
    target_cols = [c for c in target_cols if c in df.columns]
    
    df = df[target_cols].sort_values("date").reset_index(drop=True)
    return df


def _tushare_date(date_str: str | None) -> str | None:
    if date_str is None:
        return None
    return pd.to_datetime(date_str).strftime("%Y%m%d")


def _akshare_date(date_str: str | None) -> str | None:
    if date_str is None:
        return None
    return pd.to_datetime(date_str).strftime("%Y%m%d")


def _自动拉取到本地(req: 数据请求) -> str:
    df = _尝试tushare(req)
    if df is None:
        df = _尝试oss(req)
    if df is None:
        df = _尝试akshare(req)

    # 修正路径寻址
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{req.symbol}.csv"
    file_path = data_dir / file_name
    df.to_csv(file_path, index=False)
    return str(file_path)


def _尝试tushare(req: 数据请求) -> pd.DataFrame | None:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        return None
    try:
        return 从Tushare加载(req)
    except Exception:
        return None


def _尝试oss(req: 数据请求) -> pd.DataFrame | None:
    if not os.environ.get("OSS_ACCESS_KEY_ID"):
        return None
    try:
        return 从OSS加载(req)
    except Exception:
        return None


def _尝试akshare(req: 数据请求) -> pd.DataFrame | None:
    try:
        return 从AkShare加载(req)
    except Exception:
        return None


def _auto_date_range(start: str | None, end: str | None) -> tuple[str, str]:
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today().normalize()
    start_dt = pd.to_datetime(start) if start else end_dt - pd.Timedelta(days=365)
    return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
