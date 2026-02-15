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
    p.add_argument("--x", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], default="tenkan")
    p.add_argument("--y", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], default="kijun")
    p.add_argument("--z", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], default="atr_mult")
    p.add_argument("--color", choices=["eq_ret","tenkan","kijun","senkou_b","shift","atr_mult"], default="eq_ret")
    p.add_argument("--size", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], default="atr_mult")
    p.add_argument("--slice", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], default=None)
    p.add_argument("--slice-bins", type=int, default=10)
    # Robust gating
    p.add_argument("--min-trades", type=int, default=0)
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--min-calmar", type=float, default=None)
    p.add_argument("--min-lyap", type=float, default=None)
    p.add_argument("--robust", action="store_true", help="Enable robustness gating by parameter bins")
    p.add_argument("--robust-bins", type=int, default=20)
    p.add_argument("--robust-min-n", type=int, default=5)
    p.add_argument("--robust-min-median", type=float, default=None, help="Min median eq_ret (fraction, e.g. 0.05 for +5%)")
    p.add_argument("--robust-max-iqr", type=float, default=None, help="Max IQR eq_ret (fraction)")
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
            # Derive absolute params if only ratios are present
            try:
                tenkan_v = params.get("tenkan")
                kijun_v = params.get("kijun")
                senkou_v = params.get("senkou_b")
                r_kijun_v = params.get("r_kijun")
                r_senkou_v = params.get("r_senkou")
                if (kijun_v is None or (isinstance(kijun_v, float) and np.isnan(kijun_v))) and tenkan_v is not None and r_kijun_v is not None:
                    kijun_v = float(r_kijun_v) * float(tenkan_v)
                if (senkou_v is None or (isinstance(senkou_v, float) and np.isnan(senkou_v))):
                    base_kijun = kijun_v if kijun_v is not None else (float(r_kijun_v) * float(tenkan_v) if (tenkan_v is not None and r_kijun_v is not None) else None)
                    if base_kijun is not None and tenkan_v is not None and r_senkou_v is not None:
                        senkou_v = max(float(base_kijun), float(r_senkou_v) * float(tenkan_v))
                params = dict(params)
                if kijun_v is not None:
                    params["kijun"] = kijun_v
                if senkou_v is not None:
                    params["senkou_b"] = senkou_v
            except Exception:
                pass
            mt = rec.get("metrics_train") or {}
            eq_ret = mt.get("eq_ret")
            if eq_ret is None:
                eq_mult = mt.get("equity_mult")
                if isinstance(eq_mult, (int, float)) and eq_mult > 0:
                    eq_ret = float(eq_mult) - 1.0
            mdd = mt.get("max_drawdown")
            sharpe = mt.get("sharpe_proxy")
            calmar = mt.get("calmar_ratio")
            lyap = mt.get("lyapunov_exponent")
            trades = mt.get("trades")
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
                        sharpe=(None if sharpe is None else float(sharpe)),
                        calmar=(None if calmar is None else float(calmar)),
                        lyap=(None if lyap is None else float(lyap)),
                        trades=(None if trades is None else int(trades)),
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
    k_arg_u = k_arg.strip().upper()
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
        # also include fixed aggregate when viewing ALL
        p_fixed = root / "outputs" / "trial_logs" / "fixed" / "trials_from_wfa.jsonl"
        if p_fixed.exists():
            paths.append(p_fixed)
    else:
        p = root / "outputs" / "trial_logs" / "phase" / k_arg_u / "trials_from_wfa.jsonl"
        if p.exists():
            paths.append(p)
    return paths


def build_fig(
    df: pd.DataFrame,
    mdd_max: float,
    title: str,
    x: str = "tenkan",
    y: str = "kijun",
    z: str = "atr_mult",
    color_dim: str = "eq_ret",
    size_dim: str = "atr_mult",
    slice_name: str | None = None,
    slice_bins: int = 10,
) -> go.Figure:
    df = df[(df["mdd"].isna()) | (df["mdd"] <= float(mdd_max))].copy()
    # Deterministic quality gates
    if "trades" in df.columns and hasattr(build_fig, "_min_trades"):
        pass
    # Apply CLI-provided gates via closure variables later
    if df.empty:
        df = pd.DataFrame(dict(tenkan=[0], kijun=[0], atr_mult=[0], senkou_b=[0], shift=[0], eq_ret=[0]))
    # Color vector
    color_vals = (df["eq_ret"] * 100.0) if color_dim == "eq_ret" else df[color_dim]
    cmin = float(np.percentile(color_vals.dropna(), 5)) if len(color_vals.dropna()) > 10 else float(color_vals.min())
    cmax = float(np.percentile(color_vals.dropna(), 95)) if len(color_vals.dropna()) > 10 else float(color_vals.max())
    if not np.isfinite(cmin): cmin = -10.0
    if not np.isfinite(cmax): cmax = 10.0
    # Size vector
    size_series = (df["eq_ret"] if size_dim == "eq_ret" else df[size_dim])
    try:
        smin = float(np.nanpercentile(size_series, 5)); smax = float(np.nanpercentile(size_series, 95))
        if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
            smin, smax = float(size_series.min()), float(size_series.max())
    except Exception:
        smin, smax = float(size_series.min()), float(size_series.max())
    size_vals = 3.0 + 5.0 * (size_series - smin) / (smax - smin + 1e-9)
    traces: List[go.Scatter3d] = []
    slider_steps: List[Dict[str, Any]] = []
    
    # Additional gating from outer scope via nonlocal read of args is not possible; instead, embed as attributes on the function
    min_trades = getattr(build_fig, "_min_trades", 0)
    min_sharpe = getattr(build_fig, "_min_sharpe", None)
    min_calmar = getattr(build_fig, "_min_calmar", None)
    min_lyap = getattr(build_fig, "_min_lyap", None)
    robust_on = getattr(build_fig, "_robust", False)
    robust_bins = getattr(build_fig, "_robust_bins", 20)
    robust_min_n = getattr(build_fig, "_robust_min_n", 5)
    robust_min_median = getattr(build_fig, "_robust_min_median", None)
    robust_max_iqr = getattr(build_fig, "_robust_max_iqr", None)

    # Apply per-trial gates
    if min_trades:
        if "trades" in df.columns:
            df = df[(df["trades"].isna()) | (df["trades"] >= int(min_trades))]
    if min_sharpe is not None and "sharpe" in df.columns:
        df = df[(df["sharpe"].isna()) | (df["sharpe"] >= float(min_sharpe))]
    if min_calmar is not None and "calmar" in df.columns:
        df = df[(df["calmar"].isna()) | (df["calmar"] >= float(min_calmar))]
    if min_lyap is not None and "lyap" in df.columns:
        df = df[(df["lyap"].isna()) | (df["lyap"] >= float(min_lyap))]

    # Robust gating by parameter bins across all five Ichimoku dims
    if robust_on and not df.empty:
        try:
            bins_t = pd.cut(df["tenkan"], robust_bins)
            bins_kj = pd.cut(df["kijun"], robust_bins)
            bins_sb = pd.cut(df["senkou_b"], robust_bins)
            bins_sh = pd.cut(df["shift"], robust_bins)
            bins_at = pd.cut(df["atr_mult"], robust_bins)
            grp = df.assign(_t=bins_t, _kj=bins_kj, _sb=bins_sb, _sh=bins_sh, _at=bins_at)
            agg = grp.groupby(["_t","_kj","_sb","_sh","_at"])['eq_ret'].agg(['count','median', lambda s: s.quantile(0.75)-s.quantile(0.25)])
            agg.columns = ['n','med','iqr']
            mask = (agg['n'] >= int(robust_min_n))
            if robust_min_median is not None:
                mask &= (agg['med'] >= float(robust_min_median))
            if robust_max_iqr is not None:
                mask &= (agg['iqr'] <= float(robust_max_iqr))
            keep_keys = set(agg[mask].index)
            grp_keys = list(zip(grp["_t"], grp["_kj"], grp["_sb"], grp["_sh"], grp["_at"]))
            df = df[[k in keep_keys for k in grp_keys]]
        except Exception:
            pass
    if slice_name is None:
        traces.append(
            go.Scatter3d(
                x=(df[x] * 100.0 if x == "eq_ret" else df[x]), y=(df[y] * 100.0 if y == "eq_ret" else df[y]), z=(df[z] * 100.0 if z == "eq_ret" else df[z]),
                mode="markers",
                marker=dict(
                    size=size_vals,
                    color=color_vals,
                    colorscale="RdYlGn",
                    cmin=cmin, cmax=cmax,
                    opacity=0.85,
                    colorbar=dict(title=("eq_ret (%)" if color_dim == "eq_ret" else color_dim)),
                ),
                text=("K=" + df.get("K", pd.Series([""]*len(df))).astype(str) + ", fold=" + df["fold"].astype(str) + ", ph=" + df["phase"].astype(str) + ", tr=" + df["trial"].astype(str)),
                hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<br>{color_dim}=%{{marker.color:.2f}}<br>size~{size_dim}<br>%{{text}}<extra></extra>",
                name="all",
            )
        )
    else:
        q = pd.cut(df[slice_name], bins=max(2, int(slice_bins)))
        df = df.assign(_bin=q)
        cats = [c for c in pd.Categorical(q).categories]
        if not cats:
            cats = ["all"]
        for idx, cat in enumerate(cats):
            sub = df[df["_bin"] == cat]
            traces.append(
                go.Scatter3d(
                    x=(sub[x] * 100.0 if x == "eq_ret" else sub[x]), y=(sub[y] * 100.0 if y == "eq_ret" else sub[y]), z=(sub[z] * 100.0 if z == "eq_ret" else sub[z]),
                    mode="markers",
                    marker=dict(
                        size=(3.0 + 5.0 * ( (sub["eq_ret"] if size_dim == "eq_ret" else sub[size_dim]) - smin ) / (smax - smin + 1e-9)),
                        color=((sub["eq_ret"] * 100.0) if color_dim == "eq_ret" else sub[color_dim]),
                        colorscale="RdYlGn",
                        cmin=cmin, cmax=cmax,
                        opacity=0.85,
                        showscale=(idx == 0),
                        colorbar=dict(title=("eq_ret (%)" if color_dim == "eq_ret" else color_dim)) if idx == 0 else None,
                    ),
                    text=("K=" + sub.get("K", pd.Series([""]*len(sub))).astype(str) + ", fold=" + sub.get("fold", pd.Series([""]*len(sub))).astype(str) + ", ph=" + sub.get("phase", pd.Series([""]*len(sub))).astype(str) + ", tr=" + sub.get("trial", pd.Series([""]*len(sub))).astype(str)),
                    hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<br>{color_dim}=%{{marker.color:.2f}}<br>size~{size_dim}<br>%{{text}}<extra></extra>",
                    visible=(idx == 0),
                    name=str(cat),
                )
            )
            slider_steps.append({"label": str(cat), "method": "update", "args": [{"visible": [i == idx for i in range(len(cats))]}, {"title": f"{title} — {slice_name}={cat}"}]})

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=(x + (" (%)" if x == "eq_ret" else "")),
            yaxis_title=(y + (" (%)" if y == "eq_ret" else "")),
            zaxis_title=(("eq_ret (%)" if z == "eq_ret" else ("ATR×" if z == "atr_mult" else z))),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    if slice_name is not None and slider_steps:
        fig.update_layout(sliders=[{"active": 0, "currentvalue": {"prefix": f"{slice_name}: "}, "steps": slider_steps, "x": 0.05, "y": 0.05}])
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

    fig = build_fig(
        df,
        mdd_max=float(args.mdd_max),
        title=title,
        x=str(getattr(args, 'x', 'tenkan')),
        y=str(getattr(args, 'y', 'kijun')),
        z=str(getattr(args, 'z', 'atr_mult')),
        color_dim=str(getattr(args, 'color', 'eq_ret')),
        size_dim=str(getattr(args, 'size', 'atr_mult')),
        slice_name=(str(args.slice) if getattr(args, 'slice', None) else None),
        slice_bins=int(getattr(args, 'slice_bins', 10)),
    )
    write_html_with_refresh(fig, out_path, refresh_sec=60)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


