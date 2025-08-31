# -*- coding: utf-8 -*-
"""
phase_aware_module.py  —  Single-file helper for phase-aware Ichimoku + ATR (HSBC)

Drop-in module for Cursor / any Python codebase.
- Fourier features (dominant period, Low-Frequency Power ratio)
- Halving-aligned phase engine (H_buy, R_t, M/V/DD, phase classification)
- Ichimoku + ATR (Wilder-like) utilities
- HSBC-style scheduler: pool cadence & seed selection per phase
- Minimal CLI to smoke-test with a CSV cache (timestamp,open,high,low,close,volume)

Dependencies: numpy, pandas

Quick start (CLI):
    python phase_aware_module.py --csv data/BTC_USDT_2h.csv --seed 42

Integration (code):
    from phase_aware_module import phase_snapshot, select_seeds_for_today
    feat = phase_snapshot(df)  # df = OHLCV 2H DataFrame, index=datetime
    selection = select_seeds_for_today(df, feat)
    print(selection['phase'], selection['intensity'], selection['selected_seeds'])
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------
# Default Configs (override at call if needed)
# ------------------------------

HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",  # UTC may be 19 in some TZs; keep as ISO date
]

# Phase thresholds (heuristics) — tune via WF
PHASE_THRESHOLDS = {
    "accumulation": {"M_min": 0.0,  "V_max": 0.5, "DD_min": -0.25},
    "expansion":    {"M_min": 0.10, "V_min": 0.5, "DD_min": -0.20},
    "euphoria":     {"M_min": 0.25, "V_min": 0.8},
    "distribution": {"M_max": 0.10, "DD_max": -0.10},
    "bear":         {"M_max": 0.0,  "DD_max": -0.35},
}

# R_t bands => pool cadence hints
R_BANDS = {
    "low":  (0.0, 1.30),
    "mid":  (1.30, 1.60),
    "high": (1.60, 2.00),
    "peak": (2.00, 99.0),
}

# Seeds by phase (tenkan, kijun, senkou_b, shift, atr_mult)
Seed = Tuple[int, int, int, int, float]
SEEDS_BY_PHASE: Dict[str, List[Seed]] = {
    "accumulation": [(6, 26, 52, 26, 1.8), (7, 34, 60, 26, 1.5), (9, 43, 70, 26, 2.0)],
    "expansion":    [(6, 43, 100, 26, 3.0), (7, 55, 120, 30, 2.8), (9, 65, 120, 26, 3.5)],
    "euphoria":     [(6, 55, 120, 26, 4.0), (7, 65, 150, 26, 4.5), (9, 80, 200, 30, 5.0)],
    "distribution": [(9, 65, 120, 26, 2.0), (10, 80, 150, 30, 2.5), (12, 100, 200, 30, 3.0)],
    "bear":         [(6, 26, 100, 26, 3.0), (7, 34, 150, 26, 3.5), (9, 55, 200, 30, 4.0)],
}

# ------------------------------
# Small utilities
# ------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Fix Python/NumPy RNGs for reproducible exploration."""
    random.seed(seed)
    np.random.seed(seed)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rolling_vol_annualized(logret: pd.Series, lookback: int = 30, periods_per_year: int = 365) -> pd.Series:
    return logret.rolling(lookback).std() * math.sqrt(periods_per_year)

def drawdown(series: pd.Series, lookback: int = 365) -> pd.Series:
    roll_peak = series.rolling(lookback, min_periods=1).max()
    return (series - roll_peak) / roll_peak

# ------------------------------
# Ichimoku & ATR
# ------------------------------

def _highest(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).max()

def _lowest(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).min()


def scale_ichimoku(base: Tuple[int, int, int] = (9, 26, 52), *, days_per_week: int = 7, bars_per_day: int = 12) -> Tuple[int, int, int]:
    """Scale Ichimoku windows from daily defaults to the target timeframe.

    Parameters
    ----------
    base: tuple of ints
        Baseline Ichimoku windows expressed in *days* for a 7‑day week
        (tenkan, kijun, senkou_b).
    days_per_week: int
        Number of trading days per week for the asset (e.g., 5 for stocks,
        7 for crypto).
    bars_per_day: int
        Number of bars per day for the timeframe (e.g., 12 for 2‑hour bars).

    Returns
    -------
    Tuple[int, int, int]
        Scaled (tenkan, kijun, senkou_b) in **bars**.
    """

    scale = bars_per_day * (days_per_week / 7.0)
    return tuple(int(round(x * scale)) for x in base)

def ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int, shift: int) -> pd.DataFrame:
    """Compute Ichimoku lines. df must have columns: high, low, close."""
    high = df["high"]; low = df["low"]; close = df["close"]
    tenkan_line = (_highest(high, tenkan) + _lowest(low, tenkan)) / 2.0
    kijun_line  = (_highest(high, kijun)  + _lowest(low, kijun))  / 2.0
    ssa = ((tenkan_line + kijun_line) / 2.0).shift(shift)
    ssb = ((_highest(high, senkou_b) + _lowest(low, senkou_b)) / 2.0).shift(shift)
    cloud_top = np.maximum(ssa, ssb)
    cloud_bot = np.minimum(ssa, ssb)
    chikou = close.shift(-shift)
    return pd.DataFrame({
        "tenkan": tenkan_line, "kijun": kijun_line, "ssa": ssa, "ssb": ssb,
        "cloud_top": cloud_top, "cloud_bot": cloud_bot, "chikou": chikou
    }, index=df.index)

def true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder-style smoothed ATR using EMA alpha=1/period (close enough)."""
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ------------------------------
# Fourier features (simple PSD via FFT)
# ------------------------------

def _periodogram_psd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 4:
        return np.array([]), np.array([])
    w = np.hanning(n)
    xw = np.where(np.isnan(x), 0.0, x) * w
    fft = np.fft.rfft(xw)
    psd = (np.abs(fft) ** 2) / np.sum(w**2)
    freqs = np.fft.rfftfreq(n, d=1.0)  # unit sampling
    return freqs, psd

def dominant_period(close: pd.Series, window: int) -> float:
    """Return dominant period (bars) in the last `window` points."""
    if len(close) < window:
        return float("nan")
    seg = close.iloc[-window:]
    freqs, psd = _periodogram_psd(seg.values)
    if len(freqs) == 0:
        return float("nan")
    idx = int(np.nanargmax(psd[1:]) + 1) if len(psd) > 1 else int(np.nanargmax(psd))
    f_star = float(freqs[idx])
    return (1.0 / f_star) if f_star > 0 else float("nan")

def low_freq_power_ratio(close: pd.Series, window: int, f0: float) -> float:
    """Low-frequency power ratio in (0,1]. f0 ~ cutoff frequency (bars^-1)."""
    if len(close) < window:
        return float("nan")
    seg = close.iloc[-window:]
    freqs, psd = _periodogram_psd(seg.values)
    if len(freqs) == 0:
        return float("nan")
    mask = freqs < f0
    num = float(np.nansum(psd[mask]))
    den = float(np.nansum(psd))
    return (num / den) if den > 0 else float("nan")

# ------------------------------
# Phase Engine (halving-aligned)
# ------------------------------

def _compute_h_buy(close: pd.Series, halving_date: pd.Timestamp, pre_days: int = 90, post_days: int = 30) -> float:
    start = halving_date - pd.Timedelta(days=pre_days)
    end   = halving_date + pd.Timedelta(days=post_days)
    sli = close.loc[(close.index >= start) & (close.index <= end)]
    return float(sli.max()) if len(sli) else float("nan")

def _nearest_past_halving(ts: pd.Timestamp, halvings: List[pd.Timestamp]) -> Optional[pd.Timestamp]:
    past = [h for h in halvings if h <= ts]
    return max(past) if past else None

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["logret"] = np.log(df["close"]).diff()
    out["ema20"] = ema(df["close"], 20)
    out["ema100"] = ema(df["close"], 100)
    out["M"] = out["ema20"] / out["ema100"] - 1.0
    out["V_ann"] = rolling_vol_annualized(out["logret"], 30, 365)
    out["DD"] = drawdown(df["close"], 365)
    return out

def _classify_phase(m: float, v: float, dd: float, thr: Dict[str, Dict[str, float]]) -> str:
    if np.isnan(m) or np.isnan(v) or np.isnan(dd):
        return "unknown"
    bear = thr.get("bear", {})
    euph = thr.get("euphoria", {})
    dist = thr.get("distribution", {})
    expa = thr.get("expansion", {})
    accu = thr.get("accumulation", {})

    if m < bear.get("M_max", -np.inf) and dd <= bear.get("DD_max", -np.inf):
        return "bear"
    if m > euph.get("M_min",  np.inf) and v >= euph.get("V_min",  np.inf):
        return "euphoria"
    if (m < dist.get("M_max", np.inf)) or (dd < dist.get("DD_max", np.inf)):
        return "distribution"
    if (m > expa.get("M_min", -np.inf)) and (v >= expa.get("V_min", 0.0)) and (dd > expa.get("DD_min", -1.0)):
        return "expansion"
    if (m >= accu.get("M_min", 0.0)) and (v < accu.get("V_max", 1.0)) and (dd > accu.get("DD_min", -1.0)):
        return "accumulation"
    return "distribution"

def phase_snapshot(
    df_ohlcv: pd.DataFrame,
    halving_dates: List[str] = HALVING_DATES,
    thresholds: Dict[str, Dict[str, float]] = PHASE_THRESHOLDS,
    hbuy_window: Tuple[int, int] = (90, 30),
) -> pd.DataFrame:
    """
    Compute per-bar phase features aligned to last halving.
    Returns DataFrame with: close, M, V_ann, DD, last_halving, days_since_halving, H_buy, R_t, phase
    df_ohlcv must include columns: open, high, low, close, volume; index is datetime (2H recommended).
    """
    df = df_ohlcv.copy()
    feats = _compute_features(df)

    halvings = [pd.Timestamp(d) for d in halving_dates]
    feats["last_halving"] = feats.index.map(lambda ts: _nearest_past_halving(ts, halvings))
    feats = feats.dropna(subset=["last_halving"])
    feats["days_since_halving"] = (feats.index - feats["last_halving"]).dt.days

    # H_buy per halving
    hbuy_map: Dict[pd.Timestamp, float] = {}
    for h in halvings:
        hb = _compute_h_buy(feats["close"], h, hbuy_window[0], hbuy_window[1])
        hbuy_map[h] = hb
    feats["H_buy"] = feats["last_halving"].map(hbuy_map)
    feats["R_t"]  = feats["close"] / feats["H_buy"]

    feats["phase"] = [
        _classify_phase(m, v, dd, thresholds) for m, v, dd in zip(feats["M"], feats["V_ann"], feats["DD"])
    ]
    return feats[["close", "M", "V_ann", "DD", "last_halving", "days_since_halving", "H_buy", "R_t", "phase"]]

# ------------------------------
# HSBC Scheduler (intensity, seeds, mutation, scheduling)
# ------------------------------

def _in_band(x: float, band: Tuple[float, float]) -> bool:
    return (x >= band[0]) and (x < band[1])

def choose_intensity(R: float, bands: Dict[str, Tuple[float, float]] = R_BANDS, dd14: Optional[float] = None) -> str:
    intensity = "low"
    if _in_band(R, bands["mid"]):
        intensity = "medium"
    elif _in_band(R, bands["high"]):
        intensity = "high"
    elif _in_band(R, bands["peak"]):
        intensity = "exploit"
    # Guard-rail: drawdown short-term -> slow down
    if dd14 is not None and dd14 <= -0.12:
        intensity = "low"
    return intensity

def seeds_for_phase(phase: str, mapping: Dict[str, List[Seed]] = SEEDS_BY_PHASE) -> List[Seed]:
    return [(int(a), int(b), int(c), int(d), float(e)) for a, b, c, d, e in mapping.get(phase, [])]

def mutate_seeds(seeds: List[Seed], pct: float = 0.20) -> List[Seed]:
    out: List[Seed] = []
    for (t, k, sb, sh, atr) in seeds:
        dt = max(1, int(round(t * pct)))
        dk = max(1, int(round(k * pct)))
        ds = max(1, int(round(sb * pct)))
        datr = max(0.1, atr * pct)
        out.append((t + dt, k, sb, sh, atr + datr))
        out.append((t, k + dk, sb, sh, atr))
        out.append((t, k, sb + ds, sh, max(1.5, atr - datr)))
    return out

def schedule_pools(intensity: str, candidates: List[Seed]) -> List[Seed]:
    if intensity == "low":
        return candidates[:2]
    if intensity == "medium":
        return candidates[:4]
    if intensity == "high":
        return candidates[:8]
    if intensity == "exploit":
        return candidates[:3]
    return candidates[:3]

def select_seeds_for_today(
    df_ohlcv: pd.DataFrame,
    feats: Optional[pd.DataFrame] = None,
    seeds_map: Dict[str, List[Seed]] = SEEDS_BY_PHASE,
    bands: Dict[str, Tuple[float, float]] = R_BANDS,
    mutate_pct: float = 0.20,
) -> Dict[str, object]:
    """
    Return a selection dict for today's (last bar) configuration:
        {
          'phase': 'expansion',
          'R_t': 1.42,
          'intensity': 'medium',
          'selected_seeds': [ (tenkan,kijun,senkou_b,shift,atr_mult), ... ],
          'candidates': [...],
        }
    """
    if feats is None:
        feats = phase_snapshot(df_ohlcv)

    last = feats.iloc[-1]
    R = float(last["R_t"]) if last.get("R_t") is not None else float("nan")
    phase = str(last.get("phase", "unknown"))

    # Optional short-term DD(14d) guard rail (assuming 12 bars/day for 2h)
    dd14 = None
    try:
        if len(feats) >= 14 * 12:
            window = feats["close"].iloc[-14 * 12 :]
            dd14 = float((window - window.max()) / window.max()).iloc[-1]
    except Exception:
        dd14 = None

    intensity = choose_intensity(R, bands=bands, dd14=dd14)
    seeds = seeds_for_phase(phase, mapping=seeds_map)

    candidates = seeds
    if intensity != "exploit":
        candidates = mutate_seeds(seeds, pct=mutate_pct) + seeds

    selected = schedule_pools(intensity, candidates)

    return {
        "phase": phase,
        "R_t": R,
        "intensity": intensity,
        "selected_seeds": selected,
        "candidates": candidates,
    }

# ------------------------------
# Optional: Ichimoku suggestions from Fourier
# ------------------------------

def suggest_ichimoku_from_fft(
    close: pd.Series,
    window: int = 2160,
    *,
    days_per_week: int = 7,
    bars_per_day: int = 12,
) -> Dict[str, Optional[float]]:
    """Suggest Ichimoku windows from dominant period ``P`` (bars).

    Parameters
    ----------
    close: pd.Series
        Price series.
    window: int
        Number of bars for FFT window.
    days_per_week: int
        Trading days per week. Defaults to 7 (crypto).
    bars_per_day: int
        Bars per day for the timeframe.
    """
    base_t, base_k, base_sb = scale_ichimoku(days_per_week=days_per_week, bars_per_day=bars_per_day)
    P = dominant_period(close, window=window)
    if not np.isfinite(P):
        return {"P": None, "tenkan": None, "kijun": None, "senkou_b": None, "shift": float(base_k)}
    return {
        "P": P,
        "tenkan": max(float(base_t), P / 8.0),
        "kijun": max(float(base_k), P / 2.0),
        "senkou_b": max(float(base_sb), P),
        "shift": float(base_k),
    }

# ------------------------------
# Minimal CLI for smoke test
# ------------------------------

def _read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[cols]


def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Phase-aware Ichimoku helper (single file).")
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: timestamp,open,high,low,close,volume")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-fft", action="store_true")
    args = ap.parse_args()

    set_global_seed(args.seed)
    df = _read_csv(args.csv)

    feats = phase_snapshot(df)
    sel = select_seeds_for_today(df, feats)

    out = {
        "last_bar": str(df.index[-1]),
        "phase": sel["phase"],
        "R_t": (round(sel["R_t"], 4) if (sel["R_t"] is not None and np.isfinite(sel["R_t"])) else None),
        "intensity": sel["intensity"],
        "selected_seeds": sel["selected_seeds"],
        "candidates_count": len(sel["candidates"]),
    }

    if args.print_fft:
        rec = suggest_ichimoku_from_fft(df["close"])
        out["fft_suggestion"] = {k: (round(v, 2) if isinstance(v, (int, float)) and v is not None else v) for k, v in rec.items()}

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
