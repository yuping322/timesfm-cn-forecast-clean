#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate group results into a summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize_group(group: str, df: pd.DataFrame) -> dict[str, float]:
    if "status" in df.columns:
        df_ok = df[df["status"] == "ok"].copy()
    else:
        df_ok = df.copy()

    n_stocks = int(len(df_ok))
    if n_stocks == 0:
        return {
            "group": group,
            "n_stocks": 0,
            "hitrate_mean": np.nan,
            "hitrate_std": np.nan,
            "hitrate_p65": np.nan,
            "rmse_mean": np.nan,
        }

    hitrate = df_ok["hitrate"].astype(float)
    rmse = df_ok["rmse"].astype(float)

    return {
        "group": group,
        "n_stocks": n_stocks,
        "hitrate_mean": float(hitrate.mean()),
        "hitrate_std": float(hitrate.std(ddof=0)),
        "hitrate_p65": float((hitrate >= 65.0).mean() * 100.0),
        "rmse_mean": float(rmse.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-group results.csv into a summary.")
    parser.add_argument("--input-dir", type=str, default="data/research", help="Root directory with group results")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input dir not found: {input_root}")

    results_files = list(input_root.rglob("results.csv"))
    summaries = []
    for path in results_files:
        group = path.parent.name
        df = pd.read_csv(path)
        summaries.append(_summarize_group(group, df))

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["hitrate_mean", "rmse_mean"],
            ascending=[False, True],
            na_position="last",
        )

    output_path = Path(args.output) if args.output else input_root / "group_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    # Export the best group
    if not summary_df.empty:
        best_group = summary_df.iloc[0]["group"]
        best_group_file = input_root / "best_group.txt"
        best_group_file.write_text(str(best_group))
        print(f"Exported best group '{best_group}' to {best_group_file}")

if __name__ == "__main__":
    main()
