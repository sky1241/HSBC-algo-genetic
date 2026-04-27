"""Range/regime detection indicators — ALPHA-1.

Cinq indicators classiques pour distinguer trend vs range, avec un score composite.

Références:
- ADX: Wilder J.W. (1978), New Concepts in Technical Trading Systems.
  Trend si ADX > 25, range si ADX < 20.
- Choppiness Index: Dreiss B. (1991).
  CI = 100 * log10(sum(TR, n) / (HH-LL)) / log10(n).
  Range si CI > 61.8, trend si CI < 38.2.
- Efficiency Ratio: Kaufman P. (1995), Smarter Trading.
  ER = abs(close-close_n) / sum(abs(diff)). 0 = random, 1 = pure trend.
- Bollinger Band Width: Bollinger J. (2002).
  width = (upper - lower) / middle. Squeeze quand width au plus bas N-percentile.
- Hurst exponent (R/S): Hurst H.E. (1951), via Mandelbrot.
  H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = persistent (trend).

API publique:
    adx(high, low, close, period=14) -> pd.Series
    choppiness_index(high, low, close, period=14) -> pd.Series
    efficiency_ratio(close, period=10) -> pd.Series
    bollinger_band_width(close, period=20, k=2) -> pd.Series
    hurst_exponent(returns, max_lag=20) -> float
    range_score(df, ...) -> pd.DataFrame   # composite multi-indicator
    is_range_bar(df, threshold=0.6) -> pd.Series  # bool par bar
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """TR = max(H-L, |H-C_prev|, |L-C_prev|)."""
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)


def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: EMA-like avec alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


# ============================================================
# 1. ADX — Wilder 1978
# ============================================================

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index. ADX > 25 = trend; ADX < 20 = range."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    tr = _true_range(high, low, close)
    atr = _wilder_smooth(tr, period)

    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_smooth(dx, period)


# ============================================================
# 2. Choppiness Index — Dreiss 1991
# ============================================================

def choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """CI ∈ [0, 100]. > 61.8 = range/consolidation, < 38.2 = trend."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tr = _true_range(high, low, close)
    sum_tr = tr.rolling(period).sum()
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100.0 * np.log10(sum_tr / rng) / np.log10(period)


# ============================================================
# 3. Efficiency Ratio — Kaufman 1995
# ============================================================

def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """ER ∈ [0, 1]. ~0 = random/range, ~1 = pure trend."""
    close = pd.Series(close)
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).clip(0, 1)


# ============================================================
# 4. Bollinger Band Width
# ============================================================

def bollinger_band_width(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """BBW = (upper - lower) / middle = 2k * std/MA. Squeeze = BBW au plus bas percentile."""
    close = pd.Series(close)
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return (upper - lower) / ma.replace(0, np.nan)


def bbw_squeeze(close: pd.Series, period: int = 20, k: float = 2.0,
                lookback: int = 120, percentile: float = 0.20) -> pd.Series:
    """True quand BBW est dans le bas `percentile` de ses `lookback` valeurs.

    Indicateur de "squeeze" annonciateur de breakout.
    """
    bbw = bollinger_band_width(close, period, k)
    rolling_q = bbw.rolling(lookback).quantile(percentile)
    return bbw < rolling_q


# ============================================================
# 5. Hurst exponent — R/S analysis
# ============================================================

def hurst_exponent(prices: pd.Series | np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent via la méthode des moments (Mandelbrot, Mandelbrot-Wallis).

    Args:
        prices: série de PRIX (pas de returns). On utilise log-prix pour la stabilité.
        max_lag: lag max pour la régression log-log.

    H < 0.5  → mean-reverting (range/anti-persistent)
    H ≈ 0.5  → random walk
    H > 0.5  → persistent (trend)

    Méthode: pour brown motion B(t), std(B(t+lag) - B(t)) ∝ lag^H.
    On régresse log(std) vs log(lag) pour obtenir H.
    """
    arr = np.asarray(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < max_lag * 2:
        return float("nan")
    # Travail sur log-prix pour la stationnarité de variance
    arr = np.log(arr)
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = arr[lag:] - arr[:-lag]
        std = np.std(diff, ddof=0)
        if std <= 0 or not np.isfinite(std):
            return float("nan")
        tau.append(std)
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return float(slope)


def rolling_hurst(prices: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Hurst sur fenêtre glissante (prend une série de PRIX)."""
    prices = pd.Series(prices)
    out = pd.Series(np.nan, index=prices.index)
    for i in range(window, len(prices) + 1):
        slice_ = prices.iloc[i - window:i].to_numpy()
        out.iloc[i - 1] = hurst_exponent(slice_, max_lag)
    return out


# ============================================================
# Score composite
# ============================================================

@dataclass(slots=True)
class RangeScoreWeights:
    adx: float = 0.30
    choppiness: float = 0.25
    efficiency: float = 0.20
    bbw_low: float = 0.10
    hurst: float = 0.15


def _norm_adx_to_range(adx_val: pd.Series) -> pd.Series:
    """ADX 0→100 vers range_score [0=trend, 1=range].
    < 20 = full range (1.0), > 25 = full trend (0.0), entre = lerp.
    """
    return (1.0 - ((adx_val - 20.0) / 5.0).clip(0, 1)).fillna(0.5)


def _norm_chop_to_range(chop: pd.Series) -> pd.Series:
    """CI 0→100, > 61.8 = range, < 38.2 = trend."""
    return ((chop - 38.2) / (61.8 - 38.2)).clip(0, 1).fillna(0.5)


def _norm_er_to_range(er: pd.Series) -> pd.Series:
    """ER ~0 = range, ~1 = trend → range_score = 1 - ER."""
    return (1.0 - er).clip(0, 1).fillna(0.5)


def _norm_hurst_to_range(h: pd.Series) -> pd.Series:
    """H < 0.5 = mean-rev (range-like), H > 0.5 = persistent (trend)."""
    return (1.0 - ((h - 0.4) / 0.2).clip(0, 1)).fillna(0.5)


def range_score(
    df: pd.DataFrame,
    period_adx: int = 14,
    period_chop: int = 14,
    period_er: int = 10,
    period_bbw: int = 20,
    bbw_lookback: int = 120,
    hurst_window: int = 100,
    weights: RangeScoreWeights | None = None,
) -> pd.DataFrame:
    """Calcule tous les indicators et un `range_score` composite ∈ [0, 1].

    DataFrame d'entrée doit avoir colonnes 'high', 'low', 'close'.
    Retourne le DataFrame d'origine + colonnes:
      - adx, ci, er, bbw, bbw_squeeze, hurst, range_score
    """
    if weights is None:
        weights = RangeScoreWeights()
    out = df.copy()
    out["adx"] = adx(df["high"], df["low"], df["close"], period_adx)
    out["ci"] = choppiness_index(df["high"], df["low"], df["close"], period_chop)
    out["er"] = efficiency_ratio(df["close"], period_er)
    out["bbw"] = bollinger_band_width(df["close"], period_bbw)
    out["bbw_squeeze"] = bbw_squeeze(df["close"], period_bbw, lookback=bbw_lookback)
    # Hurst sur série de PRIX (pas returns)
    out["hurst"] = rolling_hurst(df["close"], window=hurst_window)

    # Score composite [0=trend, 1=range]
    s_adx = _norm_adx_to_range(out["adx"])
    s_chop = _norm_chop_to_range(out["ci"])
    s_er = _norm_er_to_range(out["er"])
    s_bbw = out["bbw_squeeze"].astype(float).fillna(0.0)
    s_hurst = _norm_hurst_to_range(out["hurst"])

    out["range_score"] = (
        weights.adx * s_adx
        + weights.choppiness * s_chop
        + weights.efficiency * s_er
        + weights.bbw_low * s_bbw
        + weights.hurst * s_hurst
    )
    return out


def is_range_bar(df: pd.DataFrame, threshold: float = 0.6, **kwargs) -> pd.Series:
    """Helper: True si la bar est en régime range (range_score > threshold)."""
    scored = range_score(df, **kwargs)
    return scored["range_score"] > threshold


__all__ = [
    "adx",
    "choppiness_index",
    "efficiency_ratio",
    "bollinger_band_width",
    "bbw_squeeze",
    "hurst_exponent",
    "rolling_hurst",
    "range_score",
    "is_range_bar",
    "RangeScoreWeights",
]
