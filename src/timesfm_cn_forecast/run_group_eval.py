#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run group-level adapter training and per-stock backtests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timesfm_cn_forecast.backtest import run_backtest
from timesfm_cn_forecast.features import FeatureExtractor, get_feature_names
from timesfm_cn_forecast.finetuning import train_linear_adapter, save_adapter
from timesfm_cn_forecast.providers import DataRequest, load_historical_data, normalize_symbol
from timesfm_cn_forecast.universe import get_stock_universe


DEFAULT_CONTEXT_LENGTHS = [30, 60, 90, 128, 256, 512]


@dataclass
class TrainSampleStore:
    features: List[np.ndarray]
    targets: List[float]
    base_preds: List[float]


def _parse_context_lengths(raw: str | None) -> List[int]:
    if not raw:
        return DEFAULT_CONTEXT_LENGTHS
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in items]


def _chunked(items: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _filter_by_min_days(
    symbols: List[str],
    duckdb_path: str,
    min_days: int,
) -> List[str]:
    if min_days <= 0 or not symbols:
        return symbols

    try:
        import duckdb
    except ImportError as exc:
        raise ImportError("duckdb is required for reading market.duckdb") from exc

    raw_to_db = {sym: normalize_symbol(sym, "db") for sym in symbols}
    db_symbols = sorted(set(raw_to_db.values()))

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        valid_db: set[str] = set()
        for chunk in _chunked(db_symbols, 500):
            df = con.execute(
                """
                SELECT symbol, COUNT(*) AS n
                FROM daily_data
                WHERE symbol IN (SELECT * FROM UNNEST(?))
                GROUP BY symbol
                """,
                [chunk],
            ).fetchdf()
            if df.empty:
                continue
            valid_db.update(df.loc[df["n"] >= min_days, "symbol"].tolist())
    finally:
        con.close()

    return [sym for sym, db_sym in raw_to_db.items() if db_sym in valid_db]


def _build_training_samples(
    symbols: List[str],
    args: argparse.Namespace,
    feature_names: List[str],
) -> TrainSampleStore:
    store = TrainSampleStore(features=[], targets=[], base_preds=[])

    for symbol in symbols:
        req = DataRequest(
            provider="duckdb",
            symbol=symbol,
            start=args.start,
            end=args.end,
            kline=True,
            value_column="close",
            duckdb_path=args.market_duckdb,
        )
        df = load_historical_data(req)
        if df.empty:
            continue

        df = df.ffill().bfill()
        if args.train_days:
            df = df.tail(args.train_days + args.context_len + args.horizon)

        prices = df["value"].to_numpy(dtype=np.float32)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv = df[ohlcv_cols].to_numpy(dtype=np.float32) if all(c in df.columns for c in ohlcv_cols) else None

        n_samples = len(prices) - args.context_len - args.horizon + 1
        if n_samples <= 0:
            continue

        for i in range(n_samples):
            context = prices[i : i + args.context_len]
            target = prices[i + args.context_len + args.horizon - 1]
            base_pred = float(context[-1])
            ohlcv_context = ohlcv[i : i + args.context_len] if ohlcv is not None else None
            feats = FeatureExtractor.compute(
                context,
                base_pred,
                ohlcv_context=ohlcv_context,
                feature_names=feature_names,
            )
            store.features.append(feats)
            store.targets.append(float(target))
            store.base_preds.append(base_pred)

    return store


def _train_group_adapter(
    symbols: List[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    feature_names = get_feature_names(args.feature_set)
    samples = _build_training_samples(symbols, args, feature_names)

    if not samples.features:
        raise RuntimeError("训练样本为空，无法训练分组适配器。")

    train_X = np.array(samples.features, dtype=np.float32)
    train_y = np.array(samples.targets, dtype=np.float32)
    train_base = np.array(samples.base_preds, dtype=np.float32)

    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)

    weights = train_linear_adapter(
        train_X=train_X,
        train_y=train_y,
        train_base=train_base,
        context_len=args.context_len,
        horizon_len=args.horizon,
        feature_names=feature_names,
        stock_code=f"group:{args.group}",
    )

    adapter_path = output_dir / "adapter.pth"
    save_adapter(weights, str(adapter_path))
    return adapter_path


def _summarize_best(stats_df: pd.DataFrame) -> dict[str, float]:
    best_idx = stats_df["RMSE"].idxmin()
    row = stats_df.loc[best_idx]
    return {
        "best_context_len": int(row["ContextLen"]),
        "rmse": float(row["RMSE"]),
        "hitrate": float(row["HitRate"]),
        "mae": float(row["MAE"]),
        "mape": float(row["MAPE"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-level training + evaluation runner.")
    parser.add_argument("--group", type=str, required=True, help="Group name, e.g. ind_xxx")
    parser.add_argument("--market-duckdb", type=str, required=True, help="market.duckdb path")
    parser.add_argument("--index-duckdb", type=str, required=True, help="index_market.duckdb path")
    parser.add_argument("--feature-set", type=str, default="full", help="Feature set (basic/technical/structural/full)")
    parser.add_argument("--train-days", type=int, default=60, help="Train days window")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--context-len", type=int, default=60, help="Context length for adapter training")
    parser.add_argument("--context-lengths", type=str, default=None, help="Eval context lengths, comma-separated")
    parser.add_argument("--test-days", type=int, default=20, help="Backtest days")
    parser.add_argument("--min-days", type=int, default=1000, help="Minimum history days per symbol")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/research", help="Output root dir")

    args = parser.parse_args()
    context_lengths = _parse_context_lengths(args.context_lengths)

    output_root = Path(args.output_dir)
    group_dir = output_root / args.group
    group_dir.mkdir(parents=True, exist_ok=True)

    symbols = get_stock_universe(args.group, duckdb_path=args.index_duckdb)
    if not symbols:
        raise RuntimeError(f"No symbols found for group: {args.group}")

    symbols = _filter_by_min_days(symbols, args.market_duckdb, args.min_days)
    if not symbols:
        raise RuntimeError("No symbols left after min-days filter.")

    adapter_path = _train_group_adapter(symbols, args, group_dir)

    results = []
    for symbol in symbols:
        try:
            stats_df = run_backtest(
                symbol=symbol,
                provider="duckdb",
                start_date=args.start,
                end_date=args.end,
                context_lengths=context_lengths,
                horizon=args.horizon,
                test_days=args.test_days,
                adapter_path=str(adapter_path),
                input_csv=None,
                duckdb_path=args.market_duckdb,
            )
            if stats_df is None or stats_df.empty:
                results.append({
                    "symbol": symbol,
                    "status": "empty",
                })
                continue

            summary = _summarize_best(stats_df)
            results.append({
                "symbol": symbol,
                **summary,
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(exc),
            })

    df = pd.DataFrame(results)
    output_path = group_dir / "results.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
