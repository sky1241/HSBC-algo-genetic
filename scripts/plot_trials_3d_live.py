#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plotly 3D live trials (auto-refresh)")
    p.add_argument("--k", required=True, help="K tag (e.g., K3). Use 'ALL' or 'K2,K3,K5' for multiple")
    p.add_argument("--jsonl", help="Optional JSONL path (defaults to outputs/trial_logs/phase/<K>/trials_from_wfa.jsonl)")
    p.add_argument("--out", help="Output HTML path (default docs/IMAGES/top_trials_live_<K>.html)")
    p.add_argument("--mdd-max", type=float, default=0.50, help="Filter MDD <= mdd_max (default 0.50)")
    return p.parse_args()


def load_df(jsonl_path: Path) -> pd.DataFrame:
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
                        atr_mult=float(params.get("atr_mult", np.nan)),
                        eq_ret=(None if eq_ret is None else float(eq_ret)),
                        mdd=(None if mdd is None else float(mdd)),
                        fold=str(rec.get("run_context", {}).get("fold", "")),
                        phase=str(rec.get("run_context", {}).get("phase_label", "")),
                        trial=int(rec.get("trial_number", -1)),
                    )
                )
            except Exception:
                continue
    df = pd.DataFrame(rows)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["tenkan", "kijun", "atr_mult", "eq_ret"], inplace=True)
    return df


def discover_jsonl_paths(k_arg: str, root: Path) -> List[Path]:
    paths: List[Path] = []
    if "," in k_arg:
        for part in [x.strip().upper() for x in k_arg.split(",") if x.strip()]:
            p = root / "outputs" / "trial_logs" / "phase" / part / "trials_from_wfa.jsonl"
            if p.exists():
                paths.append(p)
    elif k_arg.strip().upper() == "ALL":
        phase_root = root / "outputs" / "trial_logs" / "phase"
        if phase_root.exists():
            for sub in sorted(phase_root.iterdir()):
                if sub.is_dir():
                    p = sub / "trials_from_wfa.jsonl"
                    if p.exists():
                        paths.append(p)
    else:
        p = root / "outputs" / "trial_logs" / "phase" / k_arg.strip().upper() / "trials_from_wfa.jsonl"
        if p.exists():
            paths.append(p)
    return paths


def build_fig(df: pd.DataFrame, mdd_max: float, title: str) -> go.Figure:
    df = df[(df["mdd"].isna()) | (df["mdd"] <= float(mdd_max))].copy()
    if df.empty:
        df = pd.DataFrame(dict(tenkan=[0], kijun=[0], atr_mult=[0], eq_ret=[0]))
    color_vals = df["eq_ret"] * 100.0
    cmin = float(np.percentile(color_vals, 5)) if len(color_vals) > 10 else float(color_vals.min())
    cmax = float(np.percentile(color_vals, 95)) if len(color_vals) > 10 else float(color_vals.max())
    if not np.isfinite(cmin): cmin = -10.0
    if not np.isfinite(cmax): cmax = 10.0
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["tenkan"], y=df["kijun"], z=df["atr_mult"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_vals,
                    colorscale="RdYlGn",
                    cmin=cmin, cmax=cmax,
                    opacity=0.85,
                    colorbar=dict(title="eq_ret (%)"),
                ),
                text=("K=" + df.get("K", pd.Series([""]*len(df))).astype(str) + ", fold=" + df["fold"].astype(str) + ", ph=" + df["phase"].astype(str) + ", tr=" + df["trial"].astype(str)),
                hovertemplate="tenkan=%{x}<br>kijun=%{y}<br>ATRx=%{z}<br>eq_ret=%{marker.color:.2f}%<br>%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="tenkan",
            yaxis_title="kijun",
            zaxis_title="ATR×",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def write_html_with_refresh(fig: go.Figure, out_path: Path, refresh_sec: int = 60) -> None:
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
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

    frames: List[pd.DataFrame] = []
    for p in jsonl_paths:
        try:
            dfi = load_df(p)
            if not dfi.empty:
                dfi["K"] = p.parent.name
                frames.append(dfi)
        except Exception:
            continue
    if not frames:
        print("No trials available")
        return 0
    df = pd.concat(frames, ignore_index=True)
    title = f"Top trials 3D — {k_arg.upper()} (auto-refresh)"

    if args.out:
        out_path = Path(args.out)
    else:
        tag = k_arg.upper().replace(',', '_')
        out_path = here / "docs" / "IMAGES" / f"top_trials_live_{tag}.html"

    fig = build_fig(df, mdd_max=float(args.mdd_max), title=title)
    write_html_with_refresh(fig, out_path, refresh_sec=60)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


