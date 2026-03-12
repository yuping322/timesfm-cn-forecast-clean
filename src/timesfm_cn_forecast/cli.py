#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文 TimesFM 历史数据预测工具")
    parser.add_argument("--provider", choices=["local", "oss", "tushare", "akshare", "duckdb"], required=True)
    parser.add_argument("--symbol", default=None, help="股票代码或序列标识。local 模式可选。")
    parser.add_argument("--start", default=None, help="开始日期，例如 2025-01-01")
    parser.add_argument("--end", default=None, help="结束日期，例如 2025-12-31")
    parser.add_argument("--horizon", type=int, default=5, help="预测步数")
    parser.add_argument("--context-length", type=int, default=512, help="上下文窗口")
    parser.add_argument("--model-dir", default=None, help="本地模型目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--adapter", default=None, help="微调适配器 (.pth) 权重路径")
    parser.add_argument("--kline", action="store_true", help="是否生成 K 线（蜡烛图）")

    parser.add_argument("--input-csv", default=None, help="local 模式的 CSV 输入文件")
    parser.add_argument("--input-parquet", default=None, help="local 模式的 Parquet 输入文件")
    parser.add_argument("--date-column", default="date", help="日期列名")
    parser.add_argument("--value-column", default="close", help="数值列名")
    parser.add_argument("--duckdb-path", default=None, help="DuckDB 文件路径 (market.duckdb)")

    # 新增批量/Group参数
    parser.add_argument("--group", type=str, default=None, help="股票池分组（例如 CYBZ, small_fengzhi）")
    parser.add_argument("--index-duckdb", type=str, default="data/index_market.duckdb", help="指数 DuckDB 路径（结合 group 使用）")

    parser.add_argument("--oss-file-template", default="{symbol}.csv", help="OSS 文件模板")
    parser.add_argument("--oss-date-column", default="日期", help="OSS 数据日期列名")
    parser.add_argument("--oss-value-column", default="close", help="OSS 数据值列名")

    parser.add_argument("--tushare-field", default="close", help="Tushare 数值字段")

    parser.add_argument("--akshare-adjust", choices=["", "qfq", "hfq"], default="", help="AkShare 复权方式")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    if getattr(args, "group", None) or (getattr(args, "symbol", None) and "," in getattr(args, "symbol", "")):
        from .pipeline import run_batch_ranking
        from .modeling import load_advanced_model, 加载模型, 默认模型目录
        from .universe import get_stock_universe
        import pandas as pd
        from datetime import datetime, timezone
        
        symbols = []
        if getattr(args, "group", None):
            symbols = get_stock_universe(args.group, duckdb_path=args.index_duckdb)
        elif getattr(args, "symbol", None):
            symbols = [s.strip() for s in getattr(args, "symbol", "").split(",") if s.strip()]
            
        if not symbols:
            print("未找到任何股票代码进行批量预测。")
            return
            
        print(f"开始对 {len(symbols)} 支股票进行批量预测 (Provider: {args.provider})")
        
        model_dir = args.model_dir or 默认模型目录()
        if args.adapter:
            model = load_advanced_model(model_dir=model_dir, adapter_path=args.adapter)
        else:
            model = load_advanced_model(model_dir=model_dir)
            
        results = run_batch_ranking(
            model=model,
            symbols=symbols,
            provider=args.provider,
            start_date=args.start,
            end_date=args.end,
            context_len=args.context_length,
            horizon_len=args.horizon,
            duckdb_path=args.duckdb_path,
        )
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="expected_return", ascending=False)
            output_dir = Path(args.output_dir or (Path.cwd() / "outputs" / "cn_forecast"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
            group_name = args.group or "custom"
            out_file = output_dir / f"daily_picks_{group_name}_{date_str}.csv"
            df_res.to_csv(out_file, index=False)
            
            print(f"\n批量预测结果已保存至 {out_file}")
            print("\nTop 10 Picks:")
            print(df_res.head(10).to_string(index=False))
    else:
        run_pipeline(args)

