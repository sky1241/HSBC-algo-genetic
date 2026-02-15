#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a high-performance zone from live Optuna trials by extracting points with
very high eq_ret, computing a 3D shape (convex hull) in parameter space, and
expanding it by a given scale factor to propose a restricted optimization region.

Outputs:
 - docs/IMAGES/shape_zone_<K>.html     (interactive 3D Mesh3d)
 - docs/SHAPE_ZONES/region_<K>.json    (min/max box on chosen axes)

Notes:
 - eq_ret in trials JSONL is a fraction. For a 500% threshold, pass --eq-ret-threshold-pct 500.
 - Defaults axes: x=tenkan, y=kijun, z=atr_mult, filtered by MDD<=0.50.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    import plotly.graph_objs as go  # type: ignore
    import plotly.io as pio  # type: ignore
except Exception as _e:  # pragma: no cover
    raise SystemExit("Plotly is required. Please: pip install plotly")

try:
    from scipy.spatial import ConvexHull  # type: ignore
except Exception:
    ConvexHull = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build shape zone from live trials")
    p.add_argument("--k", required=True, help="K tag (e.g., K3, or K3,K5, or ALL)")
    p.add_argument("--jsonl", help="Optional JSONL path (otherwise auto-discover by K)")
    p.add_argument("--mdd-max", type=float, default=0.50)
    p.add_argument("--eq-ret-threshold-pct", type=float, default=500.0, help="Keep points with eq_ret >= this percent")
    p.add_argument("--scale", type=float, default=4.0, help="Outward scale factor around centroid")
    p.add_argument("--x", default="tenkan", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], help="X axis")
    p.add_argument("--y", default="kijun", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], help="Y axis")
    p.add_argument("--z", default="atr_mult", choices=["tenkan","kijun","senkou_b","shift","atr_mult","eq_ret"], help="Z axis")
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
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
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
    df = pd.DataFrame(rows)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["tenkan", "kijun", "atr_mult", "eq_ret"], inplace=True)
    return df


@dataclass
class ZoneBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def compute_zone(points: np.ndarray, scale: float) -> Tuple[np.ndarray, Optional[ZoneBox], np.ndarray, Optional[np.ndarray]]:
    """Return (centroid, box, hull_vertices, hull_faces) after scaling around centroid.
    If not enough points for a hull, faces is None and hull_vertices are the raw points.
    """
    if points.size == 0:
        return np.zeros(3), None, points, None
    centroid = points.mean(axis=0)
    scaled = centroid + scale * (points - centroid)
    # Bounding box on scaled points
    mins = scaled.min(axis=0)
    maxs = scaled.max(axis=0)
    box = ZoneBox(x_min=float(mins[0]), x_max=float(maxs[0]), y_min=float(mins[1]), y_max=float(maxs[1]), z_min=float(mins[2]), z_max=float(maxs[2]))
    # Convex hull on raw (for faces). If not available or insufficient points, skip faces
    faces = None
    verts = points
    if ConvexHull is not None and len(points) >= 4:
        try:
            hull = ConvexHull(points)
            verts = points
            faces = hull.simplices
        except Exception:
            faces = None
    return centroid, box, verts, faces


def build_figure(points: np.ndarray, verts: np.ndarray, faces: Optional[np.ndarray], box: ZoneBox, axis_names: Tuple[str, str, str], title: str) -> go.Figure:
    scat = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode="markers",
                        marker=dict(size=4, color="rgba(0,120,255,0.8)"), name="high eq_ret points")
    traces: List[go.BaseTraceType] = [scat]
    if faces is not None:
        mesh = go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                         i=faces[:,0], j=faces[:,1], k=faces[:,2],
                         color="rgba(255,0,0,0.3)", name="convex hull")
        traces.append(mesh)
    # Box wireframe (scaled)
    xs = [box.x_min, box.x_max]; ys = [box.y_min, box.y_max]; zs = [box.z_min, box.z_max]
    corners = np.array([[xs[i], ys[j], zs[k]] for i in [0,1] for j in [0,1] for k in [0,1]])
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    for a,b in edges:
        traces.append(go.Scatter3d(x=[corners[a,0], corners[b,0]], y=[corners[a,1], corners[b,1]], z=[corners[a,2], corners[b,2]],
                                   mode="lines", line=dict(color="rgba(0,150,0,0.6)", width=3), name="zone box", showlegend=False))
    fig = go.Figure(data=traces)
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=axis_names[0], yaxis_title=axis_names[1], zaxis_title=axis_names[2]),
                      margin=dict(l=0, r=0, t=60, b=0))
    return fig


def write_html(fig: go.Figure, out_path: Path) -> None:
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parents[1]
    # Discover/load JSONL
    if args.jsonl:
        paths = [Path(args.jsonl)] if Path(args.jsonl).exists() else []
    else:
        paths = discover_jsonl_paths(args.k, here)
    if not paths:
        print("No JSONL found for given K selection")
        return 0
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            dfi = load_trials_df(p)
            if not dfi.empty:
                frames.append(dfi)
        except Exception:
            continue
    if not frames:
        print("No trials available")
        return 0
    df = pd.concat(frames, ignore_index=True)
    # Filters
    df = df[(df["mdd"].isna()) | (df["mdd"] <= float(args.mdd_max))].copy()
    thr_frac = float(args.eq_ret_threshold_pct) / 100.0
    df = df[df["eq_ret"] >= thr_frac].copy()
    if df.empty:
        print("No points above eq_ret threshold after filters")
        return 0
    # Pick axes
    axes = (str(args.x), str(args.y), str(args.z))
    pts = df.loc[:, list(axes)].astype(float).to_numpy()
    centroid, box, verts, faces = compute_zone(pts, float(args.scale))
    if box is None:
        print("Not enough points to compute a zone")
        return 0
    # Figure
    title = f"Shape zone — {args.k.upper()} | eq_ret>={args.eq_ret_threshold_pct:.0f}%, MDD<={args.mdd_max:.2f}, scale x{args.scale}"
    fig = build_figure(pts, verts, faces, box, axes, title)
    out_html = here / "docs" / "IMAGES" / f"shape_zone_{args.k.upper().replace(',', '_')}.html"
    write_html(fig, out_html)
    # Region JSON
    region = {axes[0]: {"min": box.x_min, "max": box.x_max},
              axes[1]: {"min": box.y_min, "max": box.y_max},
              axes[2]: {"min": box.z_min, "max": box.z_max},
              "meta": {"k": args.k, "mdd_max": float(args.mdd_max), "eq_ret_threshold_pct": float(args.eq_ret_threshold_pct), "scale": float(args.scale)}}
    out_json = here / "docs" / "SHAPE_ZONES" / f"region_{args.k.upper().replace(',', '_')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(region, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_html))
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




