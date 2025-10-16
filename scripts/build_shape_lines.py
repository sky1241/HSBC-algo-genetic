#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw a 3D shape by connecting high-performance trial points (eq_ret threshold)
using k-nearest neighbors in parameter space. This is purely exploratory
visualization (no restriction applied to optimization).

Output: docs/IMAGES/shape_lines_<K>.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    import plotly.graph_objs as go  # type: ignore
    import plotly.io as pio  # type: ignore
except Exception:
    raise SystemExit("Plotly is required. Please: pip install plotly")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D KNN shape for high-eq_ret trials")
    p.add_argument("--k", required=True, help="K tag (e.g., K3, or K3,K5,K8)")
    p.add_argument("--jsonl", help="Optional JSONL path (otherwise auto-discover by K)")
    p.add_argument("--eq-ret-threshold-pct", type=float, default=500.0)
    p.add_argument("--mdd-max", type=float, default=0.50)
    p.add_argument("--x", default="tenkan", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"])
    p.add_argument("--y", default="kijun", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"])
    p.add_argument("--z", default="atr_mult", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"])
    p.add_argument("--neighbors", type=int, default=5, help="Number of nearest neighbors to connect per point (knn mode)")
    p.add_argument("--connect", choices=["knn","complete"], default="knn", help="Connection mode: knn or complete graph")
    return p.parse_args()


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
    else:
        p = root / "outputs" / "trial_logs" / "phase" / k_arg_u / "trials_from_wfa.jsonl"
        if p.exists():
            paths.append(p)
    return paths


def load_trials_df(jsonl_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                rec=json.loads(line)
            except Exception:
                continue
            params = rec.get("params") or {}
            # Derive absolute params if only ratios provided (mirror of live 3D loader)
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
            except Exception:
                pass

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
                        tenkan=float(tenkan_v if tenkan_v is not None else np.nan),
                        kijun=float(kijun_v if kijun_v is not None else np.nan),
                        senkou_b=float(senkou_v if senkou_v is not None else np.nan),
                        shift=float(params.get("shift", np.nan)),
                        atr_mult=float(params.get("atr_mult", np.nan)),
                        eq_ret=(None if eq_ret is None else float(eq_ret)),
                        mdd=(None if mdd is None else float(mdd)),
                    )
                )
            except Exception:
                continue
    df=pd.DataFrame(rows)
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.dropna(subset=["tenkan","kijun","atr_mult","eq_ret"], inplace=True)
    return df


def build_fig(points: np.ndarray, axis_names: tuple[str,str,str], color_vals: np.ndarray, neighbors: int, connect_mode: str, title: str) -> go.Figure:
    traces: List[go.BaseTraceType] = []
    # Scatter
    traces.append(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode="markers",
                               marker=dict(size=4, color=(color_vals*100.0), colorscale="RdYlGn",
                                           colorbar=dict(title="eq_ret (%)"), opacity=0.9),
                               name="points"))
    # Connection lines
    n=len(points)
    if n>=2:
        if connect_mode == "complete":
            for i in range(n):
                pi=points[i]
                for j in range(i+1, n):
                    pj=points[j]
                    traces.append(go.Scatter3d(x=[pi[0], pj[0]], y=[pi[1], pj[1]], z=[pi[2], pj[2]],
                                               mode="lines", line=dict(color="rgba(0,0,0,0.20)", width=1),
                                               showlegend=False))
        else:
            if neighbors>0:
                for i in range(n):
                    pi=points[i]
                    dists=np.linalg.norm(points - pi, axis=1)
                    idxs=np.argsort(dists)[1:1+min(neighbors, n-1)]
                    for j in idxs:
                        pj=points[j]
                        traces.append(go.Scatter3d(x=[pi[0], pj[0]], y=[pi[1], pj[1]], z=[pi[2], pj[2]],
                                                   mode="lines", line=dict(color="rgba(0,0,0,0.25)", width=2),
                                                   showlegend=False))
    fig=go.Figure(data=traces)
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=axis_names[0], yaxis_title=axis_names[1], zaxis_title=axis_names[2]),
                      margin=dict(l=0,r=0,t=60,b=0))
    return fig


def main() -> int:
    args=parse_args()
    here=Path(__file__).resolve().parents[1]
    paths=[Path(args.jsonl)] if args.jsonl and Path(args.jsonl).exists() else discover_jsonl_paths(args.k, here)
    if not paths:
        print("No JSONL found for given K selection")
        return 0
    frames: List[pd.DataFrame]=[]
    for p in paths:
        try:
            dfi=load_trials_df(p)
            if not dfi.empty:
                frames.append(dfi)
        except Exception:
            continue
    if not frames:
        print("No trials available")
        return 0
    df=pd.concat(frames, ignore_index=True)
    df=df[(df["mdd"].isna()) | (df["mdd"] <= float(args.mdd_max))].copy()
    thr=float(args.eq_ret_threshold_pct)/100.0
    df=df[df["eq_ret"] >= thr].copy()
    if df.empty:
        print("No points above threshold")
        return 0
    axes=(str(args.x), str(args.y), str(args.z))
    pts=df.loc[:, list(axes)].astype(float).to_numpy()
    colors=df["eq_ret"].astype(float).to_numpy()
    title=f"Shape lines — {args.k.upper()} | eq_ret>={args.eq_ret_threshold_pct:.0f}%"
    fig=build_fig(pts, axes, colors, int(args.neighbors), str(args.connect), title)
    out=here / "docs" / "IMAGES" / f"shape_lines_{args.k.upper().replace(',', '_')}.html"
    html=pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


