#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测流水线核心逻辑。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from .modeling import 加载模型, 运行预测, 默认模型目录, load_advanced_model, AdvancedStockModel
from .providers import 加载历史数据, 数据请求, 批量加载历史数据


def run_pipeline(args) -> None:
    请求 = 数据请求(
        provider=args.provider,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        input_csv=args.input_csv,
        input_parquet=args.input_parquet,
        date_column=args.date_column,
        value_column=args.value_column,
        auto_fetch_akshare=_是否自动拉取(args),
        oss_file_template=args.oss_file_template,
        oss_date_column=args.oss_date_column,
        oss_value_column=args.oss_value_column,
        tushare_field=args.tushare_field,
        akshare_adjust=args.akshare_adjust,
        kline=getattr(args, "kline", False),
    )

    历史数据 = 加载历史数据(请求)
    if 历史数据.empty:
        raise ValueError("历史数据为空，无法预测")

    数值序列 = 历史数据["value"].to_numpy(dtype=np.float32)
    model_dir = args.model_dir or 默认模型目录()
    输出目录 = Path(args.output_dir or (Path.cwd() / "outputs" / "cn_forecast"))
    输出目录.mkdir(parents=True, exist_ok=True)

    if getattr(args, "adapter", None):
        模型 = load_advanced_model(model_dir=model_dir, adapter_path=args.adapter)
        
        # 提取 OHLCV 数据作为额外特征
        ohlcv_inputs = None
        ohlcv_cols = ["open", "high", "low", "volume"]
        if all(c in 历史数据.columns for c in ohlcv_cols):
            ohlcv_data = 历史数据[ohlcv_cols].to_numpy(dtype=np.float32)
            ohlcv_inputs = [ohlcv_data[-args.context_length:] if ohlcv_data.shape[0] > args.context_length else ohlcv_data]
            
        # AdvancedStockModel.forecast returns (pts, qts)
        点预测_raw, 分位数预测_raw = 模型.forecast(
            inputs=[数值序列[-args.context_length:] if 数值序列.size > args.context_length else 数值序列], 
            horizon=args.horizon,
            ohlcv_inputs=ohlcv_inputs
        )
        点预测, 分位数预测 = 点预测_raw[0], 分位数预测_raw[0]
    else:
        模型 = 加载模型(model_dir)
        点预测, 分位数预测 = 运行预测(模型, 数值序列, args.context_length, args.horizon)

    保存结果(
        历史数据=历史数据,
        点预测=点预测,
        分位数预测=分位数预测,
        输出目录=输出目录,
        provider=args.provider,
        symbol=args.symbol or "local_series",
        context_length=args.context_length,
        horizon=args.horizon,
        model_dir=model_dir,
        kline=getattr(args, "kline", False),
    )


class BatchRankingPipeline:
    """批量股票预测与排名管线。"""
    def __init__(self, model: AdvancedStockModel):
        self.model = model

    def run(
        self,
        symbols: List[str],
        provider: str,
        start_date: str,
        end_date: str,
        context_len: int = 60,
        horizon_len: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """运行批量预测并按预期收益率排序。"""
        print(f"正在为 {len(symbols)} 只股票批量加载数据 ({provider})...")
        
        wide_df = 批量加载历史数据(
            symbols=symbols,
            provider=provider,
            start=start_date,
            end=end_date,
            **kwargs
        )
        
        if wide_df.empty:
            return []

        results = []
        actual_symbols = wide_df.columns
        for symbol in actual_symbols:
            try:
                series = wide_df[symbol].dropna().values
                if len(series) < context_len:
                    continue
                
                model_input = series[-context_len:].astype(np.float32)
                pts, _ = self.model.forecast(inputs=[model_input], horizon=horizon_len)
                
                pred_price = float(pts[0, 0])
                last_price = float(series[-1])
                expected_return = (pred_price - last_price) / last_price * 100
                
                results.append({
                    "symbol": symbol,
                    "last_price": last_price,
                    "predicted_price": pred_price,
                    "expected_return": expected_return,
                    "timestamp": str(wide_df.index[-1])
                })
            except Exception as e:
                print(f"预测股票 {symbol} 时出错: {e}")

        return sorted(results, key=lambda x: x["expected_return"], reverse=True)

def run_batch_ranking(
    model: AdvancedStockModel,
    symbols: List[str],
    provider: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """批量排名入口。"""
    pipeline = BatchRankingPipeline(model)
    return pipeline.run(symbols, provider, start_date, end_date, **kwargs)


def 保存结果(
    历史数据: pd.DataFrame,
    点预测: np.ndarray,
    分位数预测: np.ndarray,
    输出目录: Path,
    provider: str,
    symbol: str,
    context_length: int,
    horizon: int,
    model_dir: str,
    kline: bool = False,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(输出目录 / ".mplconfig"))
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _配置中文字体(font_manager)

    历史数据.to_csv(输出目录 / "history.csv", index=False)

    最后日期 = 历史数据["date"].iloc[-1]
    预测日期 = pd.date_range(start=最后日期, periods=horizon + 1, freq="D")[1:]
    forecast_df = pd.DataFrame(
        {
            "date": 预测日期,
            "point_forecast": 点预测,
        }
    )
    for idx in range(分位数预测.shape[1]):
        forecast_df[f"quantile_{idx}"] = 分位数预测[:, idx]
    forecast_df.to_csv(输出目录 / "forecast.csv", index=False)

    summary = {
        "provider": provider,
        "symbol": symbol,
        "history_points": int(len(历史数据)),
        "context_length": int(min(context_length, len(历史数据))),
        "horizon": int(horizon),
        "model_dir": model_dir,
        "last_history_date": str(最后日期.date()),
        "last_history_value": float(历史数据["value"].iloc[-1]),
        "forecast_head": [float(x) for x in 点预测[: min(10, len(点预测))]],
    }
    with open(输出目录 / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    plt.figure(figsize=(12, 6))
    plt.plot(历史数据["date"], 历史数据["value"], label="历史数据", linewidth=2)
    plt.plot(预测日期, 点预测, label="点预测", linestyle="--", linewidth=2)
    if 分位数预测.shape[1] >= 10:
        plt.fill_between(
            预测日期,
            分位数预测[:, 1],
            分位数预测[:, -1],
            alpha=0.15,
            label="分位区间",
        )
    plt.title(f"TimesFM 预测: {symbol}")
    plt.xlabel("日期")
    plt.ylabel("数值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(输出目录 / "forecast.png", dpi=200, bbox_inches="tight")
    plt.close()

    if kline:
        _绘制K线图(历史数据, symbol, 输出目录)

    print(
        json.dumps(
            {
                "output_dir": str(输出目录.resolve()),
                "provider": provider,
                "symbol": symbol,
                "history_points": len(历史数据),
                "forecast_points": len(点预测),
            },
            ensure_ascii=False,
        )
    )


def _绘制K线图(df: pd.DataFrame, symbol: str, 输出目录: Path) -> None:
    """绘制蜡烛图。"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    # 检查是否有 OHLC 列
    ohlc_cols = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in ohlc_cols):
        print(f"警告：数据中缺少 OHLC 列，跳过 K 线图绘制。现有列: {df.columns.tolist()}")
        return

    df = df.copy()
    df["date_num"] = mdates.date2num(df["date"])

    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 颜色设置 (符合中国股市习惯: 涨红跌绿)
    up_color = "#ef5350"    # 红色
    down_color = "#26a69a" # 绿色

    width = 0.6 # 蜡烛宽度

    for _, row in df.iterrows():
        open_p = row["open"]
        high_p = row["high"]
        low_p = row["low"]
        close_p = row["close"]
        
        color = up_color if close_p >= open_p else down_color
        
        # 1. 绘制影线 (Wick)
        ax.plot([row["date_num"], row["date_num"]], [low_p, high_p], color=color, linewidth=1.2, zorder=2)
        
        # 2. 绘制实体 (Body)
        body_bottom = min(open_p, close_p)
        body_height = abs(open_p - close_p)
        if body_height < 0.01:
            body_height = 0.01 # 极小高度确保可见
            
        rect = Rectangle(
            (row["date_num"] - width/2, body_bottom), 
            width, 
            body_height, 
            facecolor=color, 
            edgecolor=color,
            zorder=3
        )
        ax.add_patch(rect)

    ax.set_title(f"{symbol} K 线走势图", fontsize=16, fontweight="bold")
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("价格", fontsize=12)
    
    # 设置 X 轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    save_path = 输出目录 / f"forecast_kline_{symbol}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"K 线图已保存至: {save_path}")


def _配置中文字体(font_manager) -> None:
    preferred_fonts = [
        "PingFang SC",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return

def _是否自动拉取(args) -> bool:
    if args.provider != "local":
        return False
    if args.input_csv or args.input_parquet:
        return False
    return True
