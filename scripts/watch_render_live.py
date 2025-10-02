#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Reuse plotting utilities
from plotsafe_imports import import_heat, import_trials3d  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch JSONL and regenerate live dashboards (K or ALL)")
    p.add_argument("--k", required=True, help="K tag (e.g., K3). Use 'ALL' or 'K2,K3,K5' for multiple")
    p.add_argument("--interval", type=int, default=45, help="Seconds between checks (default 45)")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--mdd-max", type=float, default=0.50, help="Filter MDD <= mdd_max (default 0.50)")
    # 3D config pass-through
    p.add_argument("--x", default="tenkan")
    p.add_argument("--y", default="kijun")
    p.add_argument("--z", default="atr_mult")
    p.add_argument("--color", default="eq_ret")
    p.add_argument("--size", default="atr_mult")
    p.add_argument("--slice", default=None)
    p.add_argument("--slice-bins", type=int, default=10)
    p.add_argument("--min-trades", type=int, default=0)
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--min-calmar", type=float, default=None)
    p.add_argument("--min-lyap", type=float, default=None)
    p.add_argument("--robust", action="store_true")
    p.add_argument("--robust-bins", type=int, default=20)
    p.add_argument("--robust-min-n", type=int, default=5)
    p.add_argument("--robust-min-median", type=float, default=None)
    p.add_argument("--robust-max-iqr", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parents[1]
    heat = import_heat()
    tri3d = import_trials3d()

    last_sig: str | None = None
    while True:
        try:
            # Discover JSONL paths for selection
            paths: List[Path] = heat.discover_jsonl_paths(args.k, here)
            # Compute signature (paths + sizes + mtimes)
            sig_parts: List[str] = []
            for p in sorted(paths):
                try:
                    st = p.stat()
                    sig_parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
                except FileNotFoundError:
                    sig_parts.append(f"{p}:0:0")
            sig = "|".join(sig_parts)

            if sig and sig != last_sig:
                # Load and concat
                frames: List[pd.DataFrame] = []
                for p in paths:
                    try:
                        dfi = heat.load_trials(p)
                        if not dfi.empty:
                            dfi["K"] = p.parent.name
                            frames.append(dfi)
                    except Exception:
                        continue
                if frames:
                    df_all = pd.concat(frames, ignore_index=True)
                    # Render heatmap
                    title = f"Ichimoku trials — {args.k.upper()} (auto-refresh)"
                    out_h = here / "docs" / "IMAGES" / f"heatmaps_live_{args.k.upper().replace(',', '_')}.html"
                    fig_h = heat.build_fig(df_all, mdd_max=float(args.mdd_max), nbins=int(args.bins), title=title)
                    heat.write_html_with_refresh(fig_h, out_h, refresh_sec=args.interval)
                    # Render 3D
                    title3 = f"Top trials 3D — {args.k.upper()} (auto-refresh)"
                    out_3 = here / "docs" / "IMAGES" / f"top_trials_live_{args.k.upper().replace(',', '_')}.html"
                    df3 = tri3d.load_df(paths[0])  # not used; we rebuild from df_all for consistent K tag
                    # Reuse 3D builder but with combined df and user-selected axes/color/size/slice
                    # Attach gating args onto function (simple way without changing signatures everywhere)
                    setattr(tri3d.build_fig, "_min_trades", int(args.min_trades))
                    setattr(tri3d.build_fig, "_min_sharpe", (None if args.min_sharpe is None else float(args.min_sharpe)))
                    setattr(tri3d.build_fig, "_min_calmar", (None if args.min_calmar is None else float(args.min_calmar)))
                    setattr(tri3d.build_fig, "_min_lyap", (None if args.min_lyap is None else float(args.min_lyap)))
                    setattr(tri3d.build_fig, "_robust", bool(args.robust))
                    setattr(tri3d.build_fig, "_robust_bins", int(args.robust_bins))
                    setattr(tri3d.build_fig, "_robust_min_n", int(args.robust_min_n))
                    setattr(tri3d.build_fig, "_robust_min_median", (None if args.robust_min_median is None else float(args.robust_min_median)))
                    setattr(tri3d.build_fig, "_robust_max_iqr", (None if args.robust_max_iqr is None else float(args.robust_max_iqr)))
                    fig3 = tri3d.build_fig(
                        df_all.rename(columns={"shift": "shift"}),
                        mdd_max=float(args.mdd_max),
                        title=title3,
                        x=str(args.x), y=str(args.y), z=str(args.z),
                        color_dim=str(args.color), size_dim=str(args.size),
                        slice_name=(str(args.slice) if args.slice else None),
                        slice_bins=int(args.slice_bins),
                    )
                    tri3d.write_html_with_refresh(fig3, out_3, refresh_sec=args.interval)
                last_sig = sig
        except Exception:
            pass
        time.sleep(max(5, int(args.interval)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


