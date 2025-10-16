#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_sys_path()
from features_fourier import FourierConfig, compute_fourier_features  # noqa: E402


def load_price_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to find time and close columns
    time_col = None
    for c in ["timestamp", "time", "date", "datetime"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # Try index-like implicit datetime
        raise FileNotFoundError("No datetime column found in provided CSV")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col).sort_index()
    # Close column
    close_col = None
    for c in ["close", "Close", "c", "price", "ClosePrice"]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise FileNotFoundError("No close column found in provided CSV")
    # Keep only close
    return pd.DataFrame({"close": pd.to_numeric(df[close_col], errors="coerce")}).dropna()


def discover_default_price_path(root: Path) -> Path:
    candidates = [
        root / "data" / "BTC_FUSED_2h_clean.csv",
        root / "data" / "BTC_FUSED_2h.csv",
        root / "data" / "BTC_USDT_2h.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: any 2h
    any_2h = sorted((root / "data").glob("*2h*.csv"))
    if any_2h:
        return any_2h[0]
    raise FileNotFoundError("No suitable H2 CSV found under data/")


def parse_jsonl_trials(jsonl_path: Path) -> pd.DataFrame:
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
            # Derive absolute params if only ratios are present (align with plot_trials_3d_live.py)
            try:
                tenkan_v = params.get("tenkan")
                kijun_v = params.get("kijun")
                senkou_v = params.get("senkou_b")
                r_kijun_v = params.get("r_kijun")
                r_senkou_v = params.get("r_senkou")
                if (kijun_v is None or (isinstance(kijun_v, float) and np.isnan(kijun_v))) and tenkan_v is not None and r_kijun_v is not None:
                    kijun_v = float(r_kijun_v) * float(tenkan_v)
                if (senkou_v is None or (isinstance(senkou_v, float) and np.isnan(senkou_v))):
                    base_kijun = (
                        kijun_v
                        if kijun_v is not None
                        else (float(r_kijun_v) * float(tenkan_v) if (tenkan_v is not None and r_kijun_v is not None) else None)
                    )
                    if base_kijun is not None and tenkan_v is not None and r_senkou_v is not None:
                        senkou_v = max(float(base_kijun), float(r_senkou_v) * float(tenkan_v))
                params = dict(params)
                if kijun_v is not None:
                    params["kijun"] = kijun_v
                if senkou_v is not None:
                    params["senkou_b"] = senkou_v
            except Exception:
                pass

            # Prefer test metrics if available
            mt = rec.get("metrics_test") or rec.get("metrics_train") or {}
            eq_ret = mt.get("eq_ret")
            if eq_ret is None:
                eq_mult = mt.get("equity_mult")
                if isinstance(eq_mult, (int, float)) and eq_mult > 0:
                    eq_ret = float(eq_mult) - 1.0
            mdd = mt.get("max_drawdown")
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
                        trades=(None if trades is None else int(trades)),
                        trial=int(rec.get("trial_number", -1)),
                        fold=str(rec.get("run_context", {}).get("fold", "")),
                    )
                )
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["tenkan", "kijun", "senkou_b", "eq_ret", "mdd", "trades"], inplace=True)
    return df


@dataclass
class IchimokuRanges:
    tenkan_min: int
    tenkan_max: int
    kijun_min: int
    kijun_max: int
    senkou_min: int
    senkou_max: int
    shift_rec: int


def derive_ranges_from_period(p_period: float) -> IchimokuRanges:
    # Map dominant period P -> ranges (bars)
    # Tenkan ~ P/4 .. P/2 ; Kijun ~ 0.75P .. 1.25P ; Senkou_B ~ 1.5P .. 3P
    P = max(4.0, float(p_period))
    tk_min = int(max(4, round(0.25 * P)))
    tk_max = int(max(tk_min + 1, round(0.50 * P)))
    kj_min = int(max(tk_max, round(0.75 * P)))
    kj_max = int(max(kj_min + 1, round(1.25 * P)))
    sb_min = int(max(kj_max, round(1.50 * P)))
    sb_max = int(max(sb_min + 1, round(3.00 * P)))
    shift_rec = int(np.clip(round(0.50 * kj_min), 10, 75))
    return IchimokuRanges(tk_min, tk_max, kj_min, kj_max, sb_min, sb_max, shift_rec)


def inside_ranges(row: pd.Series, ranges: IchimokuRanges) -> bool:
    return (
        ranges.tenkan_min <= row["tenkan"] <= ranges.tenkan_max
        and ranges.kijun_min <= row["kijun"] <= ranges.kijun_max
        and ranges.senkou_min <= row["senkou_b"] <= ranges.senkou_max
    )


def build_report_md(
    out_path: Path,
    K: str,
    period_stats: Dict[str, float],
    ranges: IchimokuRanges,
    stats_inside: Dict[str, float],
    stats_out: Dict[str, float],
    atr_hint: Tuple[float, float, float] | None,
) -> None:
    lines: List[str] = []
    lines.append(f"### Fourier-guided Ichimoku ({K})")
    lines.append("")
    lines.append("- Data: BTC H2 (Fourier Welch) — trailing dominant period and volatility")
    lines.append(
        "- Dominant period P1 (bars): median={med:.1f}, p25={p25:.1f}, p75={p75:.1f}".format(
            med=period_stats.get("med", float("nan")),
            p25=period_stats.get("p25", float("nan")),
            p75=period_stats.get("p75", float("nan")),
        )
    )
    lines.append(
        "- Suggested ranges: tenkan=[{t0},{t1}], kijun=[{k0},{k1}], senkou_b=[{s0},{s1}], shift≈{sh}".format(
            t0=ranges.tenkan_min,
            t1=ranges.tenkan_max,
            k0=ranges.kijun_min,
            k1=ranges.kijun_max,
            s0=ranges.senkou_min,
            s1=ranges.senkou_max,
            sh=ranges.shift_rec,
        )
    )
    if atr_hint is not None:
        lines.append(
            "- ATR× hint (from volatility terciles): low≈{a:.1f}, mid≈{b:.1f}, high≈{c:.1f}".format(
                a=float(atr_hint[0]), b=float(atr_hint[1]), c=float(atr_hint[2])
            )
        )
    lines.append("")
    lines.append("- Filter: MDD<=50%, trades>=280; metric: test if available else train")
    lines.append(
        "- Inside recommended zone: n={n}, eq_med={med:.2f}%, eq_p90={p90:.2f}%".format(
            n=int(stats_inside.get("n", 0)),
            med=float(stats_inside.get("med", float("nan"))),
            p90=float(stats_inside.get("p90", float("nan"))),
        )
    )
    lines.append(
        "- Outside zone: n={n}, eq_med={med:.2f}%, eq_p90={p90:.2f}%".format(
            n=int(stats_out.get("n", 0)),
            med=float(stats_out.get("med", float("nan"))),
            p90=float(stats_out.get("p90", float("nan"))),
        )
    )
    verdict = "NON"
    try:
        if stats_inside.get("p90", -1.0) > stats_out.get("p90", -1.0):
            verdict = "OUI"
    except Exception:
        pass
    lines.append("")
    lines.append(f"- Verdict (Fourier -> Ichimoku zone improves tails): {verdict}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def robust_percentiles(vec: np.ndarray, qs: List[float]) -> List[float]:
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan")] * len(qs)
    return [float(np.nanpercentile(v, q)) for q in qs]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fourier-guided Ichimoku signal for K3")
    ap.add_argument("--k", default="K3")
    ap.add_argument("--price-csv", default=None)
    ap.add_argument(
        "--jsonl",
        default=None,
        help="Path to trials_from_wfa.jsonl (defaults to outputs/trial_logs/phase/<K>/trials_from_wfa.jsonl)",
    )
    ap.add_argument("--mdd-max", type=float, default=0.50)
    ap.add_argument("--min-trades", type=int, default=280)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = _project_root()
    # Load price
    price_path = Path(args.price_csv) if args.price_csv else discover_default_price_path(root)
    df_price = load_price_df(price_path)

    # Fourier features
    cfg = FourierConfig(price_col="close", fs_per_day=12.0)
    feats = compute_fourier_features(df_price, cfg)
    p1 = feats["P1_period"].to_numpy(dtype=float)
    p_med, p25, p75 = robust_percentiles(p1, [50, 25, 75])
    ranges = derive_ranges_from_period(p_med if np.isfinite(p_med) else np.nanmedian(p1))

    # ATR hints from volatility terciles mapped to multipliers in [5, 15]
    vol = feats["volatility"].to_numpy(dtype=float)
    v25, v50, v75 = robust_percentiles(vol, [25, 50, 75])
    atr_hint = None
    if np.isfinite(v25) and np.isfinite(v50) and np.isfinite(v75):
        # Map linearly vol quantiles to ATR× in [5, 15]
        def map_vol(x: float, lo: float, hi: float) -> float:
            if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return 10.0
            alpha = (x - lo) / (hi - lo)
            return float(5.0 + 10.0 * np.clip(alpha, 0.0, 1.0))

        atr_hint = (map_vol(v25, v25, v75), map_vol(v50, v25, v75), map_vol(v75, v25, v75))

    # Load trials
    k = str(args.k).strip().upper()
    jsonl_path = Path(args.jsonl) if args.jsonl else (root / "outputs" / "trial_logs" / "phase" / k / "trials_from_wfa.jsonl")
    if not jsonl_path.exists():
        print(f"No JSONL found at {jsonl_path}")
        return 0
    trials = parse_jsonl_trials(jsonl_path)
    if trials.empty:
        print("No trials available")
        return 0
    # Filter
    trials = trials[(trials["mdd"] <= float(args.mdd_max)) & (trials["trades"] >= int(args.min_trades))].copy()
    if trials.empty:
        print("No trials pass filters")
        return 0

    # Inside/outside stats
    mask_in = trials.apply(lambda r: inside_ranges(r, ranges), axis=1)
    inside = trials[mask_in]
    outside = trials[~mask_in]

    def stats_block(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {"n": 0, "med": float("nan"), "p90": float("nan")}
        eq = (df["eq_ret"].to_numpy(dtype=float) * 100.0)
        eq = eq[np.isfinite(eq)]
        if eq.size == 0:
            return {"n": 0, "med": float("nan"), "p90": float("nan")}
        return {
            "n": int(eq.size),
            "med": float(np.nanpercentile(eq, 50)),
            "p90": float(np.nanpercentile(eq, 90)),
        }

    stats_in = stats_block(inside)
    stats_out = stats_block(outside)

    # Report
    out_md = Path(args.out) if args.out else (root / "docs" / f"FOURIER_K3_SIGNAL_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.md")
    build_report_md(
        out_md,
        k,
        {"med": p_med, "p25": p25, "p75": p75},
        ranges,
        stats_in,
        stats_out,
        atr_hint,
    )
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


