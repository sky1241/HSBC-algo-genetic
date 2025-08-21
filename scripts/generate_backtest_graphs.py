#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère des graphiques simples (equity, drawdown) depuis un CSV (timestamp,equity[,drawdown]).

Usage:
  py -3 scripts/generate_backtest_graphs.py --csv outputs/runs_pipeline_web6_latest.csv --out-dir outputs/graphs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default=str(Path("outputs") / "graphs"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "equity" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"] if "timestamp" in df.columns else df.index, df["equity"], label="Equity")
        plt.title("Equity timeline")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_timeline.png", dpi=120)
        plt.close()

        # Compute drawdown if not provided
        if "drawdown" not in df.columns:
            peak = df["equity"].cummax()
            dd = (df["equity"] - peak) / peak
        else:
            dd = df["drawdown"]

        plt.figure(figsize=(10, 3))
        plt.plot(df["timestamp"] if "timestamp" in df.columns else df.index, dd, color="crimson", label="Drawdown")
        plt.title("Drawdown timeline")
        plt.xlabel("Time")
        plt.ylabel("DD")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dd_timeline.png", dpi=120)
        plt.close()

        print(f"Saved graphs to {out_dir}")
        return 0

    print("CSV must contain at least an 'equity' column.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


