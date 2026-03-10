#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测验证与参数优化脚本。"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from .modeling import 加载模型, 运行预测, 默认模型目录, load_advanced_model
from .providers import 数据请求, 加载历史数据

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标。"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 方向准确率 (Hit Rate)
    true_direction = np.sign(y_true[1:] - y_true[:-1]) if len(y_true) > 1 else np.array([0])
    pred_direction = np.sign(y_pred[1:] - y_true[:-1]) if len(y_pred) > 1 else np.array([0])
    hit_rate = np.mean(true_direction == pred_direction) * 100 if len(true_direction) > 0 else 0.0
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "HitRate": float(hit_rate)
    }

def run_backtest(
    symbol: str,
    provider: str,
    start_date: str,
    end_date: str,
    context_lengths: List[int],
    horizon: int = 5,
    test_days: int = 20,
    adapter_path: Optional[str] = None
):
    """
    运行滚动回测。
    """
    print(f"开始为 {symbol} 运行回测验证 (Provider: {provider})...")
    
    # 1. 加载足够长的数据
    # 为了保证回测最后一天也有足够的 context，需要提前多加载一些
    max_context = max(context_lengths)
    req = 数据请求(
        symbol=symbol,
        provider=provider,
        start=start_date,
        end=end_date,
        kline=True if adapter_path else False
    )
    df = 加载历史数据(req)
    
    if len(df) < max_context + test_days:
        print(f"数据量不足：需要至少 {max_context + test_days} 天，实际仅有 {len(df)} 天。")
        return

    prices = df["value"].values
    dates = df.index.values
    
    # 如果有适配器，加载高级模型
    model_dir = 默认模型目录()
    if adapter_path:
        model = load_advanced_model(model_dir, adapter_path)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv_data = df[ohlcv_cols].values if all(c in df.columns for c in ohlcv_cols) else None
    else:
        model = 加载模型(model_dir)

    all_stats = []

    for clen in context_lengths:
        print(f"测试上下文长度: {clen}...")
        errors = []
        preds = []
        actuals = []
        
        # 滚动窗口回测最后 test_days 天 (每次只测试 T+1 的预测准确性，或者 T+horizon 的平均值)
        # 这里我们关注 T+1 预测的准确性，作为指标
        for i in range(len(prices) - test_days, len(prices)):
            context = prices[i - clen : i]
            target = prices[i] # 这一天的实际收盘价
            
            if adapter_path:
                # 高级模型预测
                ohlcv_context = [ohlcv_data[i - clen : i]] if ohlcv_data is not None else None
                pts, _ = model.forecast(inputs=[context.astype(np.float32)], horizon=1, ohlcv_inputs=ohlcv_context)
                pred_val = pts[0, 0]
            else:
                # 基础模型预测
                pts, _ = 运行预测(model, context.astype(np.float32), clen, 1)
                pred_val = pts[0]
            
            preds.append(pred_val)
            actuals.append(target)
            
        metrics = calculate_metrics(np.array(actuals), np.array(preds))
        metrics["ContextLen"] = clen
        all_stats.append(metrics)

    # 打印结果表
    stats_df = pd.DataFrame(all_stats)
    print("\n回测结果汇总 (T+1 预测准确性):")
    print(stats_df.to_string(index=False))
    
    best_clen = stats_df.loc[stats_df['RMSE'].idxmin(), 'ContextLen']
    print(f"\n建议：对于 {symbol}，最优上下文长度为 {int(best_clen)} (基于最小 RMSE)。")
    
    return stats_df

def main():
    parser = argparse.ArgumentParser(description="TimesFM 滚动回测工具")
    parser.add_argument("--symbol", type=str, required=True, help="股票代码")
    parser.add_argument("--provider", type=str, default="akshare", help="数据源")
    parser.add_argument("--start", type=str, default="2023-01-01", help="开始日期")
    parser.add_argument("--test-days", type=int, default=20, help="回测天数")
    parser.add_argument("--horizon", type=int, default=5, help="预测步长")
    parser.add_argument("--adapter", type=str, help="适配器路径 (可选)")
    
    args = parser.parse_args()
    
    context_lengths = [30, 60, 90, 128, 256, 512]
    run_backtest(
        symbol=args.symbol,
        provider=args.provider,
        start_date=args.start,
        end_date="2026-03-10",
        context_lengths=context_lengths,
        horizon=args.horizon,
        test_days=args.test_days,
        adapter_path=args.adapter
    )

if __name__ == "__main__":
    main()
