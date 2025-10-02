#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot top Optuna trials as a 3D "carte maritime" of Ichimoku params.

Input: JSONL file of trials, one JSON per line with keys:
  - params: { tenkan, kijun, senkou_b, shift, atr_mult }
  - metrics_train: { eq_ret }  # optional; if missing, compute from equity_mult
    (fallbacks: metrics_train.equity_mult -> eq_ret = equity_mult - 1)
  - optional: mdd under metrics_train for filtering

Usage:
  python scripts/plot_top_trials.py --input outputs/trial_logs/phase/K3/seed_123/trials_20250930_123456_1234.jsonl 
                                    --out docs/IMAGES/top_trials.png --top 0.10 --mdd-max 0.50

Notes:
  - Keeps only eq_ret > 0, then top X% (default 10%) by eq_ret.
  - Aggregates by (tenkan, kijun, atr_mult) → median eq_ret, IQR, robustness = median - IQR.
  - Plots 3D scatter: X=tenkan, Y=kijun, Z=atr_mult; color=median eq_ret (RdYlGn), size=robustness (normalized).
  - Right panel lists top 15 aggregated settings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot top trials heatmap-like 3D scatter from JSONL trial logs")
    p.add_argument("--input", required=True, help="Path to trials_*.jsonl file")
    p.add_argument("--out", default="docs/IMAGES/top_trials.png", help="Output PNG path")
    p.add_argument("--top", type=float, default=0.10, help="Top fraction by eq_ret to keep (0-1), default 0.10")
    p.add_argument("--mdd-max", type=float, default=0.50, help="Max MDD filter if present in metrics")
    return p.parse_args()


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def read_trials(jsonl_path: Path, mdd_max: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            params = safe_get(rec, "params", default={}) or {}
            mt = safe_get(rec, "metrics_train", default={}) or {}
            # eq_ret: try direct, else compute from equity_mult
            eq_ret = mt.get("eq_ret")
            if eq_ret is None:
                eq_mult = mt.get("equity_mult")
                if isinstance(eq_mult, (int, float)) and eq_mult > 0:
                    eq_ret = float(eq_mult) - 1.0
            # mdd if present (for filtering)
            mdd = mt.get("max_drawdown")

            try:
                row = dict(
                    tenkan=float(params.get("tenkan", np.nan)),
                    kijun=float(params.get("kijun", np.nan)),
                    senkou_b=float(params.get("senkou_b", np.nan)),
                    shift=float(params.get("shift", np.nan)),
                    atr_mult=float(params.get("atr_mult", np.nan)),
                    eq_ret=(None if eq_ret is None else float(eq_ret)),
                    mdd=(None if mdd is None else float(mdd)),
                )
            except Exception:
                continue
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop missing params or eq_ret
    df.dropna(subset=["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "eq_ret"], inplace=True)
    # Filter MDD if column exists
    if "mdd" in df.columns:
        df = df[(df["mdd"].isna()) | (df["mdd"] <= float(mdd_max))]
    # Keep only positive eq_ret
    df = df[df["eq_ret"] > 0.0]
    return df


def iqr(values: np.ndarray) -> float:
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    return float(q3 - q1)


def aggregate_top(df: pd.DataFrame, top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    # Top X% by eq_ret
    thr = np.nanpercentile(df["eq_ret"].values, 100.0 * (1.0 - top_frac))
    dft = df[df["eq_ret"] >= thr].copy()
    # Aggregate by (tenkan,kijun,atr_mult)
    grp = dft.groupby(["tenkan", "kijun", "atr_mult"], as_index=False)
    agg = grp["eq_ret"].agg(["median"]).reset_index()
    # compute IQR & robustness
    # Note: need full group to compute IQR; recompute via apply
    iqr_list = []
    for _, g in grp:
        iqr_list.append(iqr(g["eq_ret"].values))
    agg["iqr"] = iqr_list
    agg["robust"] = agg["median"] - agg["iqr"]
    # Sort for panel text
    agg_sorted = agg.sort_values(["median", "robust"], ascending=[False, False]).reset_index(drop=True)
    return dft, agg_sorted


def plot_figure(dft: pd.DataFrame, agg_sorted: pd.DataFrame, out_path: Path) -> None:
    if dft.empty or agg_sorted.empty:
        # Create a placeholder figure
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No data after filtering", ha="center", va="center")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.0])

    # Left: 3D scatter (tenkan, kijun, atr_mult)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    X = agg_sorted["tenkan"].values
    Y = agg_sorted["kijun"].values
    Z = agg_sorted["atr_mult"].values
    C = agg_sorted["median"].values * 100.0  # %
    R = agg_sorted["robust"].values
    # normalize size
    if np.nanmax(np.abs(R)) > 1e-12:
        Rn = (R - np.nanmin(R)) / max(1e-12, (np.nanmax(R) - np.nanmin(R)))
    else:
        Rn = np.zeros_like(R)
    sizes = 50.0 + 250.0 * Rn
    sc = ax3d.scatter(X, Y, Z, c=C, cmap="RdYlGn", s=sizes, alpha=0.8)
    ax3d.set_xlabel("tenkan")
    ax3d.set_ylabel("kijun")
    ax3d.set_zlabel("atr_mult")
    cbar = fig.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.6)
    cbar.set_label("eq_ret median (%)")
    ax3d.set_title("Top trials (aggregated) — 3D view")

    # Right: text panel with top 15
    axTxt = fig.add_subplot(gs[0, 1])
    axTxt.axis("off")
    top_n = min(15, len(agg_sorted))
    lines: List[str] = []
    for i in range(top_n):
        r = agg_sorted.iloc[i]
        line1 = f"T={int(r['tenkan'])} | K={int(r['kijun'])} | ATR={r['atr_mult']:.2f}"
        line2 = f"eq_ret_med={r['median']:+.4f} | Robust={r['robust']:+.4f}"
        lines.append(line1)
        lines.append(line2)
        lines.append("")
    axTxt.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main_cli() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)
    df = read_trials(in_path, mdd_max=float(args.mdd_max))
    dft, agg_sorted = aggregate_top(df, top_frac=float(args.top))
    plot_figure(dft, agg_sorted, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())


