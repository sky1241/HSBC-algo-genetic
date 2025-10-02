#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plotly live heatmaps (auto-refresh) from trial JSONL")
    p.add_argument("--k", required=True, help="K tag (e.g., K3). Use 'ALL' or 'K2,K3,K5' for multiple")
    p.add_argument(
        "--jsonl",
        help="Path to trials_from_wfa.jsonl (default under outputs/trial_logs/phase/<K>/trials_from_wfa.jsonl)",
    )
    p.add_argument(
        "--out",
        help="Output HTML path (default docs/IMAGES/heatmaps_live_<K>.html)",
    )
    p.add_argument("--bins", type=int, default=15, help="Number of bins per axis (default 15)")
    p.add_argument("--mdd-max", type=float, default=0.50, help="Filter MDD <= mdd_max (default 0.50)")
    return p.parse_args()


def load_trials(jsonl_path: Path) -> pd.DataFrame:
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
            params = rec.get("params") or {}
            mt = rec.get("metrics_train") or {}
            eq_ret = mt.get("eq_ret")
            if eq_ret is None:
                eq_mult = mt.get("equity_mult")
                if isinstance(eq_mult, (int, float)) and eq_mult > 0:
                    eq_ret = float(eq_mult) - 1.0
            mdd = mt.get("max_drawdown")
            try:
                rows.append(
                    dict(
                        tenkan=float(params.get("tenkan", np.nan)),
                        kijun=float(params.get("kijun", np.nan)),
                        senkou_b=float(params.get("senkou_b", np.nan)),
                        shift=float(params.get("shift", np.nan)),
                        atr_mult=float(params.get("atr_mult", np.nan)),
                        eq_ret=(None if eq_ret is None else float(eq_ret)),
                        mdd=(None if mdd is None else float(mdd)),
                        phase=str(rec.get("run_context", {}).get("phase_label", rec.get("phase", "?"))),
                    )
                )
            except Exception:
                continue
    df = pd.DataFrame(rows)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "eq_ret"], inplace=True)
    return df


def discover_jsonl_paths(k_arg: str, root: Path) -> List[Path]:
    paths: List[Path] = []
    k_arg_u = k_arg.strip().upper()
    # Phase JSONL selection
    if "," in k_arg_u:
        for part in [x.strip().upper() for x in k_arg_u.split(",") if x.strip()]:
            p = root / "outputs" / "trial_logs" / "phase" / part / "trials_from_wfa.jsonl"
            if p.exists():
                paths.append(p)
    elif k_arg_u == "ALL":
        phase_root = root / "outputs" / "trial_logs" / "phase"
        if phase_root.exists():
            for sub in sorted(phase_root.iterdir()):
                if sub.is_dir():
                    p = sub / "trials_from_wfa.jsonl"
                    if p.exists():
                        paths.append(p)
    else:
        p = root / "outputs" / "trial_logs" / "phase" / k_arg_u / "trials_from_wfa.jsonl"
        if p.exists():
            paths.append(p)

    # Optionally include FIXED aggregate when viewing ALL
    if k_arg_u == "ALL":
        p_fixed = root / "outputs" / "trial_logs" / "fixed" / "trials_from_wfa.jsonl"
        if p_fixed.exists():
            paths.append(p_fixed)

    return paths


def cut_series(s: pd.Series, nb: int) -> pd.Series:
    try:
        return pd.cut(s, bins=nb)
    except Exception:
        return s


def build_fig(df: pd.DataFrame, mdd_max: float, nbins: int, title: str) -> go.Figure:
    df = df[(df["mdd"].isna()) | (df["mdd"] <= float(mdd_max))].copy()
    # All parameter pairs
    params = ["tenkan", "kijun", "senkou_b", "shift", "atr_mult"]
    pairs = list(combinations(params, 2))
    phases = sorted([str(x) for x in df["phase"].dropna().unique()]) or ["?"]

    traces: List[go.BaseTraceType] = []
    trace_meta: List[Dict[str, Any]] = []

    # Precompute traces for each (pair, phase)
    for (a, b) in pairs:
        for ph in phases:
            sub = df[df["phase"] == ph][[a, b, "eq_ret"]].dropna()
            if len(sub) < 20:
                # build an empty trace placeholder to keep indexing consistent
                z = np.array([[np.nan]])
                xcats = [""]
                ycats = [""]
            else:
                A = cut_series(sub[a], nbins)
                B = cut_series(sub[b], nbins)
                piv = sub.assign(A=A, B=B).groupby(["A", "B"])["eq_ret"].median().unstack("B")
                if piv is None or piv.shape[0] == 0:
                    z = np.array([[np.nan]])
                    xcats = [""]
                    ycats = [""]
                else:
                    z = piv.values * 100.0  # % color
                    ycats = [str(x) for x in piv.index]
                    xcats = [str(x) for x in piv.columns]
            # Heatmap trace
            traces.append(
                go.Heatmap(
                    z=z,
                    x=xcats,
                    y=ycats,
                    colorscale="RdYlGn",
                    zmid=0.0,
                    colorbar=dict(title="eq_ret median (%)"),
                    visible=False,
                )
            )
            trace_meta.append(dict(pair=f"{a}×{b}", phase=ph, a=a, b=b, kind="heat"))
            # Overlay scatter points (each trial as a dot), aligned to same bins as heatmap
            pts = df[df["phase"] == ph][[a, b, "eq_ret"]].dropna()
            marker = dict(
                color=(pts["eq_ret"] * 100.0).tolist() if len(pts) else None,
                colorscale="RdYlGn",
                cmin=-max(1.0, abs(float(df["eq_ret"].quantile(0.95) * 100.0))) if len(df) else -50,
                cmax=max(1.0, abs(float(df["eq_ret"].quantile(0.95) * 100.0))) if len(df) else 50,
                size=6,
                opacity=0.75,
                showscale=False,
            )
            # Use identical binning so points land on the heatmap grid categories
            if len(pts):
                A_pts = cut_series(pts[a], nbins)
                B_pts = cut_series(pts[b], nbins)
                x_pts = [str(x) for x in B_pts]
                y_pts = [str(y) for y in A_pts]
            else:
                x_pts = []
                y_pts = []
            traces.append(
                go.Scattergl(
                    x=x_pts,
                    y=y_pts,
                    mode="markers",
                    marker=marker,
                    visible=False,
                    name="trials",
                )
            )
            trace_meta.append(dict(pair=f"{a}×{b}", phase=ph, a=a, b=b, kind="scatter"))

    # Default visible: first trace
    if traces:
        traces[0].visible = True
        if len(traces) > 1:
            traces[1].visible = True

    fig = go.Figure(data=traces)
    fig.update_layout(title=title, xaxis_title=trace_meta[0]["b"] if trace_meta else "", yaxis_title=trace_meta[0]["a"] if trace_meta else "")

    # Build dropdowns for pair and phase
    pair_names = sorted({m["pair"] for m in trace_meta})
    phase_names = phases

    def visibility_mask(select_pair: str, select_phase: str) -> List[bool]:
        mask: List[bool] = []
        for m in trace_meta:
            mask.append(m["pair"] == select_pair and m["phase"] == select_phase)
        if not any(mask) and mask:
            # fallback: show first trace
            mask = [False] * len(mask)
            mask[0] = True
        return mask

    # Updatemenus (two dropdowns)
    updatemenus = []
    # Pair dropdown
    pair_buttons = []
    default_phase = phase_names[0] if phase_names else "?"
    for p in pair_names:
        vis = visibility_mask(p, default_phase)
        pair_buttons.append(
            dict(
                label=p,
                method="update",
                args=[{"visible": vis}, {"xaxis": {"title": p.split("×")[1]}, "yaxis": {"title": p.split("×")[0]}}],
            )
        )
    updatemenus.append(dict(buttons=pair_buttons, direction="down", x=0.0, y=1.15, xanchor="left"))

    # Phase dropdown
    phase_buttons = []
    default_pair = pair_names[0] if pair_names else "tenkan×kijun"
    for ph in phase_names:
        vis = visibility_mask(default_pair, ph)
        phase_buttons.append(
            dict(label=str(ph), method="update", args=[{"visible": vis}])
        )
    updatemenus.append(dict(buttons=phase_buttons, direction="down", x=0.25, y=1.15, xanchor="left"))

    fig.update_layout(updatemenus=updatemenus, margin=dict(l=60, r=20, t=80, b=60))
    return fig


def write_html_with_refresh(fig: go.Figure, out_path: Path, refresh_sec: int = 60) -> None:
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    # inject meta refresh
    html = html.replace("<head>", f"<head>\n<meta http-equiv=\"refresh\" content=\"{refresh_sec}\">\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parents[1]
    k_arg = args.k.strip()
    if args.jsonl:
        jsonl_paths = [Path(args.jsonl)] if Path(args.jsonl).exists() else []
    else:
        jsonl_paths = discover_jsonl_paths(k_arg, here)
    if not jsonl_paths:
        print("No JSONL found for given K selection")
        return 0

    # Concatenate all trials
    frames: List[pd.DataFrame] = []
    for p in jsonl_paths:
        try:
            dfi = load_trials(p)
            if not dfi.empty:
                dfi["K"] = p.parent.name
                frames.append(dfi)
        except Exception:
            continue
    if not frames:
        print("No trials in JSONL (after filtering)")
        return 0
    df = pd.concat(frames, ignore_index=True)
    title = f"Ichimoku trials — {k_arg.upper()} (auto-refresh)"

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        tag = k_arg.upper().replace(',', '_')
        out_path = here / "docs" / "IMAGES" / f"heatmaps_live_{tag}.html"

    fig = build_fig(df, mdd_max=float(args.mdd_max), nbins=int(args.bins), title=title)
    write_html_with_refresh(fig, out_path, refresh_sec=60)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



