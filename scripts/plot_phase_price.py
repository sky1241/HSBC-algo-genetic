#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot price with phase-colored bands.

Usage:
    python scripts/plot_phase_price.py --symbol BTC_USD --timeframe 1d

The script loads OHLCV data from ``data/<symbol>_<timeframe>.csv``,
computes phase features via :func:`phase_snapshot`, and saves a PNG
with the price curve and colored regions for each phase.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.phase_aware_module import phase_snapshot  # type: ignore

OHLCV_PATHS: Dict[Tuple[str, str], Path] = {
    ("BTC_USDT", "2h"): Path("data") / "BTC_USDT_2h.csv",
    ("BTC_USDT", "1d"): Path("data") / "BTC_USDT_1d.csv",
    ("BTC_USD", "2h"): Path("data") / "BTC_USD_2h.csv",
    ("BTC_USD", "1d"): Path("data") / "BTC_USD_1d.csv",
}

PHASE_COLORS = {
    "accumulation": "skyblue",
    "expansion": "limegreen",
    "euphoria": "gold",
    "distribution": "orange",
    "bear": "crimson",
    "capitulation": "purple",
}

def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]

def segment_phases(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    phases = df["phase"].astype(str)
    segments = []
    start = df.index[0]
    current = phases.iloc[0]
    for ts, ph in zip(df.index[1:], phases.iloc[1:]):
        if ph != current:
            segments.append((start, ts, current))
            start = ts
            current = ph
    segments.append((start, df.index[-1], current))
    return segments

def plot_with_phases(df: pd.DataFrame, sym: str, tf: str) -> None:
    segments = segment_phases(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], color="black", linewidth=1)
    for start, end, ph in segments:
        color = PHASE_COLORS.get(ph, "grey")
        ax.axvspan(start, end, color=color, alpha=0.3)
    handles = [mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    ax.legend(handles=handles, title="Phase", loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"{sym} {tf} — price with phases")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig.tight_layout()
    out_dir = Path("outputs") / "fourier" / "phase_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sym}_{tf}_phase_price.png"
    fig.savefig(out_file)
    print("Saved plot to", out_file)

def main() -> int:
    ap = argparse.ArgumentParser(description="Plot price with phase-colored bands.")
    ap.add_argument("--symbol", default="BTC_USD")
    ap.add_argument("--timeframe", default="1d")
    args = ap.parse_args()

    path = OHLCV_PATHS.get((args.symbol, args.timeframe))
    if path is None or not path.exists():
        raise FileNotFoundError(f"OHLCV not found for {(args.symbol, args.timeframe)}")
    df = read_ohlcv(path)
    feats = phase_snapshot(df)
    df = df.join(feats[["phase"]], how="inner")
    plot_with_phases(df, args.symbol, args.timeframe)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
