#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize robust top zones from trials_from_wfa.jsonl")
    p.add_argument("--root", default=".")
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--nmin", type=int, default=8)
    p.add_argument("--med", type=float, default=0.02, help="min median eq_ret (fraction)")
    p.add_argument("--iqr", type=float, default=0.05, help="max IQR eq_ret (fraction)")
    p.add_argument("--mdd-max", type=float, default=1.0)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-sharpe", type=float, default=0.0)
    p.add_argument("--k", default="ALL", help="K selection: ALL or comma list")
    p.add_argument("--out-json", default="docs/SUMMARY_TOPZONES.json")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--min-median-trades", type=int, default=0, help="Min median trades per bin for stability")
    return p.parse_args()


def load_trials(root: Path, k_sel: str) -> pd.DataFrame:
    files: List[Path] = []
    trial_root = root / "outputs" / "trial_logs"
    if k_sel.strip().upper() == "ALL":
        if (trial_root / "phase").exists():
            for sub in sorted((trial_root / "phase").iterdir()):
                if sub.is_dir():
                    p = sub / "trials_from_wfa.jsonl"
                    if p.exists():
                        files.append(p)
        p_fixed = trial_root / "fixed" / "trials_from_wfa.jsonl"
        if p_fixed.exists():
            files.append(p_fixed)
    else:
        for part in [x.strip().upper() for x in k_sel.split(",") if x.strip()]:
            p = trial_root / "phase" / part / "trials_from_wfa.jsonl"
            if p.exists():
                files.append(p)

    rows: List[Dict[str, Any]] = []
    for fp in files:
        K = fp.parent.name
        try:
            with fp.open("r", encoding="utf-8") as f:
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
                        eqm = mt.get("equity_mult")
                        if isinstance(eqm, (int, float)) and float(eqm) > 0:
                            eq_ret = float(eqm) - 1.0
                    if eq_ret is None:
                        continue
                    tenkan = params.get("tenkan")
                    kijun = params.get("kijun")
                    sb = params.get("senkou_b")
                    sh = params.get("shift")
                    atr = params.get("atr_mult")
                    rk = params.get("r_kijun")
                    rs = params.get("r_senkou")
                    try:
                        if (kijun is None) and (tenkan is not None) and (rk is not None):
                            kijun = float(rk) * float(tenkan)
                        if (sb is None) and (tenkan is not None):
                            base = kijun if kijun is not None else (float(rk) * float(tenkan) if rk is not None else None)
                            if (base is not None) and (rs is not None):
                                sb = max(float(base), float(rs) * float(tenkan))
                    except Exception:
                        pass
                    rows.append(
                        dict(
                            K=K,
                            tenkan=tenkan,
                            kijun=kijun,
                            senkou_b=sb,
                            shift=sh,
                            atr_mult=atr,
                            eq_ret=eq_ret,
                            trades=mt.get("trades"),
                            sharpe=mt.get("sharpe_proxy"),
                            calmar=mt.get("calmar_ratio"),
                            lyap=mt.get("lyapunov_exponent"),
                            mdd=mt.get("max_drawdown"),
                        )
                    )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def robust_bins(df: pd.DataFrame, bins: int, nmin: int, med: float, iqr: float, min_median_trades: int = 0) -> pd.DataFrame:
    grp = df.assign(
        b_t=pd.cut(df["tenkan"], bins),
        b_kj=pd.cut(df["kijun"], bins),
        b_sb=pd.cut(df["senkou_b"], bins),
        b_sh=pd.cut(df["shift"], bins),
        b_at=pd.cut(df["atr_mult"], bins),
    )
    agg_eq = (
        grp.groupby(["b_t", "b_kj", "b_sb", "b_sh", "b_at"])  # type: ignore
        ["eq_ret"]
        .agg(["count", "median", lambda s: s.quantile(0.75) - s.quantile(0.25)])
        .rename(columns={"<lambda_0>": "iqr"})
    )
    # median trades per bin (stability proxy)
    agg_tr = grp.groupby(["b_t", "b_kj", "b_sb", "b_sh", "b_at"])  # type: ignore
    agg_trades = agg_tr["trades"].median().rename("median_trades")
    agg = pd.concat([agg_eq, agg_trades], axis=1).reset_index()
    m = (agg["count"] >= int(nmin)) & (agg["median"] >= float(med)) & (agg["iqr"] <= float(iqr))
    if int(min_median_trades) > 0 and "median_trades" in agg.columns:
        m &= (agg["median_trades"].fillna(0) >= int(min_median_trades))
    return agg[m].sort_values("median", ascending=False)


def interval_mid(iv: Any) -> float:
    try:
        s = str(iv).strip("[]()")
        a, b = s.split(",")
        return (float(a) + float(b)) / 2.0
    except Exception:
        return float("nan")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    df = load_trials(root, args.k)
    if df.empty:
        print("No data")
        return 0

    # Gating
    g = df.dropna(subset=["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "eq_ret"]).copy()
    g = g[(g["mdd"].isna()) | (g["mdd"] <= float(args.mdd_max))]
    if "trades" in g.columns and int(args.min_trades) > 0:
        g = g[(g["trades"].isna()) | (g["trades"] >= int(args.min_trades))]
    if "sharpe" in g.columns and args.min_sharpe is not None:
        g = g[(g["sharpe"].isna()) | (g["sharpe"] >= float(args.min_sharpe))]

    top_all = robust_bins(
        g, bins=int(args.bins), nmin=int(args.nmin), med=float(args.med), iqr=float(args.iqr), min_median_trades=int(args.min_median_trades)
    ).head(int(args.top))
    # Format
    result: Dict[str, Any] = {"top_overall": [], "per_K": []}
    for _, r in top_all.iterrows():
        tm = interval_mid(r["b_t"]) ; km = interval_mid(r["b_kj"]) ; sm = interval_mid(r["b_sb"]) ; shm = interval_mid(r["b_sh"]) ; am = interval_mid(r["b_at"]) 
        kjrat = (km / tm) if (tm and not math.isnan(tm) and km and not math.isnan(km)) else float("nan")
        sbrat = (sm / km) if (km and not math.isnan(km) and sm and not math.isnan(sm)) else float("nan")
        result["top_overall"].append({
            "median_pct": round(float(r["median"]) * 100.0, 2),
            "iqr_pct": round(float(r["iqr"]) * 100.0, 2),
            "n": int(r["count"]),
            "median_trades": (None if pd.isna(r.get("median_trades", np.nan)) else int(r.get("median_trades", 0))),
            "tenkan": tm, "kijun": km, "senkou_b": sm, "shift": shm, "atr_mult": am,
            "kijun_over_tenkan": kjrat, "senkou_over_kijun": sbrat,
        })

    # Per-K
    for K in sorted(g["K"].dropna().unique()):
        gi = g[g["K"] == K]
        top_k = robust_bins(
            gi, bins=int(args.bins), nmin=int(args.nmin), med=float(args.med), iqr=float(args.iqr), min_median_trades=int(args.min_median_trades)
        ).head(int(args.top))
        items: List[Dict[str, Any]] = []
        for _, r in top_k.iterrows():
            tm = interval_mid(r["b_t"]) ; km = interval_mid(r["b_kj"]) ; sm = interval_mid(r["b_sb"]) ; shm = interval_mid(r["b_sh"]) ; am = interval_mid(r["b_at"]) 
            kjrat = (km / tm) if (tm and not math.isnan(tm) and km and not math.isnan(km)) else float("nan")
            sbrat = (sm / km) if (km and not math.isnan(km) and sm and not math.isnan(sm)) else float("nan")
            items.append({
                "median_pct": round(float(r["median"]) * 100.0, 2),
                "iqr_pct": round(float(r["iqr"]) * 100.0, 2),
                "n": int(r["count"]),
                "median_trades": (None if pd.isna(r.get("median_trades", np.nan)) else int(r.get("median_trades", 0))),
                "tenkan": tm, "kijun": km, "senkou_b": sm, "shift": shm, "atr_mult": am,
                "kijun_over_tenkan": kjrat, "senkou_over_kijun": sbrat,
            })
        result["per_K"].append({"K": str(K), "top": items})

    out_path = root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also, print a concise text summary to stdout
    print("TOP OVERALL (robust):")
    for it in result["top_overall"]:
        print(
            f"med={it['median_pct']:.2f}% IQR={it['iqr_pct']:.2f}% n={it['n']} trades~{it['median_trades']} | "
            f"tenkan~{it['tenkan']:.0f}, kijun~{it['kijun']:.0f} (x{it['kijun_over_tenkan']:.2f}), "
            f"senkou~{it['senkou_b']:.0f} (sb/kj~{it['senkou_over_kijun']:.2f}), shift~{it['shift']:.0f}, ATR~{it['atr_mult']:.2f}"
        )
    print("BY K:")
    for kblk in result["per_K"]:
        print(f"K={kblk['K']}")
        for it in kblk["top"]:
            print(
                f"  med={it['median_pct']:.2f}% IQR={it['iqr_pct']:.2f}% n={it['n']} | "
                f"tenkan~{it['tenkan']:.0f}, kijun~{it['kijun']:.0f} (x{it['kijun_over_tenkan']:.2f}), "
                f"senkou~{it['senkou_b']:.0f} (sb/kj~{it['senkou_over_kijun']:.2f}), shift~{it['shift']:.0f}, ATR~{it['atr_mult']:.2f}"
            )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


