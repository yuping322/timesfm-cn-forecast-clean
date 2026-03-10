#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文 TimesFM 历史数据预测工具")
    parser.add_argument("--provider", choices=["local", "oss", "tushare", "akshare"], required=True)
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

    parser.add_argument("--oss-file-template", default="{symbol}.csv", help="OSS 文件模板")
    parser.add_argument("--oss-date-column", default="日期", help="OSS 数据日期列名")
    parser.add_argument("--oss-value-column", default="close", help="OSS 数据值列名")

    parser.add_argument("--tushare-field", default="close", help="Tushare 数值字段")

    parser.add_argument("--akshare-adjust", choices=["", "qfq", "hfq"], default="", help="AkShare 复权方式")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
