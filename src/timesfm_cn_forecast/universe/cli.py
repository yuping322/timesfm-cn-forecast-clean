"""
指数成份股模块 CLI 入口。

Usage:
    # 拉取所有支持的指数
    python -m timesfm_cn_forecast.universe

    # 拉取指定指数
    python -m timesfm_cn_forecast.universe --index HS300 ZZ500 CYBZ small

    # 指定 DuckDB 路径
    python -m timesfm_cn_forecast.universe --duckdb-path data/index_market.duckdb

    # 只查看已有数据，不更新
    python -m timesfm_cn_forecast.universe --list
"""
from __future__ import annotations

import argparse
import logging

from .fetcher import INDEX_MAP, fetch_constituents
from .storage import list_all_symbols, upsert_constituents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="拉取指数成份股并写入 DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        nargs="+",
        default=list(INDEX_MAP.keys()),
        help=f"要拉取的分组列表，默认全部。可选: {list(INDEX_MAP.keys())}",
    )
    parser.add_argument(
        "--duckdb-path",
        default="data/index_market.duckdb",
        help="index_market.duckdb 文件路径（默认: data/index_market.duckdb）",
    )
    parser.add_argument(
        "--industry-csv",
        default="data/industry_category.csv",
        help="申万行业 CSV 路径（默认: data/industry_category.csv）",
    )
    parser.add_argument(
        "--concept-csv",
        default="data/concept_category.csv",
        help="概念 CSV 路径（默认: data/concept_category.csv）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅列出 DuckDB 中已存储的分组及成份股数量，不执行拉取",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\n已存储于 {args.duckdb_path} 的指数成份股：")
        df = list_all_symbols(args.duckdb_path)
        if df.empty:
            print("  （暂无数据，请先运行拉取）")
        else:
            print(df.to_string(index=False))
        return

    total_ok = 0
    total_fail = 0

    for sym in args.index:
        if sym not in INDEX_MAP:
            logger.warning(f"跳过未知指数: {sym}")
            total_fail += 1
            continue

        desc = INDEX_MAP[sym]["description"]
        logger.info(f"=== 开始拉取 [{sym}] {desc} ===")
        try:
            df = fetch_constituents(
                sym,
                industry_csv=args.industry_csv,
                concept_csv=args.concept_csv,
            )
            written = upsert_constituents(df, args.duckdb_path)
            logger.info(f"  [{sym}] 写入完成，共 {written} 条。")
            total_ok += 1
        except Exception as err:
            logger.error(f"  [{sym}] 拉取/写入失败: {err}")
            total_fail += 1

    print(f"\n批量拉取结束：成功 {total_ok} / 失败 {total_fail}")

    # 最后展示写入后的汇总
    df = list_all_symbols(args.duckdb_path)
    if not df.empty:
        print("\n当前 DuckDB 中的成份股汇总：")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
