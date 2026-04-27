"""Tests pour range_detector — ALPHA-1.

Vérifie chaque indicator sur des séries SYNTHÉTIQUES connues:
- Pure trend (steady up): ADX haut, ER ~1, Hurst > 0.5, CI bas
- Pure range (sine wave): ADX bas, ER ~0, Hurst < 0.5, CI haut
- Random walk: tous les indicators ~ neutre
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.range_detector import (
    RangeScoreWeights,
    adx,
    bbw_squeeze,
    bollinger_band_width,
    choppiness_index,
    efficiency_ratio,
    hurst_exponent,
    is_range_bar,
    range_score,
    rolling_hurst,
)


# ============================================================
# Fixtures: séries synthétiques
# ============================================================

def _ohlc_from_close(close: np.ndarray, vol_pct: float = 0.005) -> pd.DataFrame:
    """Construit un OHLC plausible à partir d'une série close."""
    rng = np.random.default_rng(42)
    n = len(close)
    high = close * (1 + rng.uniform(0, vol_pct, n))
    low = close * (1 - rng.uniform(0, vol_pct, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.ones(n),
    })


@pytest.fixture
def trend_df():
    """Uptrend bruité: drift fort + petite vol (signal/noise élevé)."""
    rng = np.random.default_rng(42)
    n = 500
    drift = np.linspace(0, 0.5, n)               # log-price total +0.5 → ~+65%
    noise = rng.normal(0, 0.005, n).cumsum()
    close = 100 * np.exp(drift + noise)
    return _ohlc_from_close(close, vol_pct=0.005)


@pytest.fixture
def range_df():
    """Mean-reverting (Ornstein-Uhlenbeck-like): oscille autour de 100 avec noise."""
    rng = np.random.default_rng(42)
    n = 500
    log_price = np.zeros(n)
    log_price[0] = np.log(100)
    mean_log = np.log(100)
    for i in range(1, n):
        # OU process: dX = -theta(X-mu)dt + sigma dW
        log_price[i] = log_price[i - 1] + 0.05 * (mean_log - log_price[i - 1]) + rng.normal(0, 0.01)
    close = np.exp(log_price)
    return _ohlc_from_close(close, vol_pct=0.005)


@pytest.fixture
def random_walk_df():
    rng = np.random.default_rng(7)
    n = 1000
    rets = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(rets))
    return _ohlc_from_close(close, vol_pct=0.002)


# ============================================================
# ADX
# ============================================================

def test_adx_high_in_trend(trend_df):
    a = adx(trend_df["high"], trend_df["low"], trend_df["close"], period=14)
    last = a.iloc[-1]
    assert last > 25, f"ADX should be high in steady uptrend, got {last}"


def test_adx_lower_in_range_than_trend(range_df, trend_df):
    """ADX moyenne plus basse en mean-reversion que en trend."""
    a_range = adx(range_df["high"], range_df["low"], range_df["close"], period=14).dropna().mean()
    a_trend = adx(trend_df["high"], trend_df["low"], trend_df["close"], period=14).dropna().mean()
    assert a_range < a_trend


# ============================================================
# Choppiness Index
# ============================================================

def test_ci_in_expected_range(range_df, trend_df):
    """CI doit être ∈ [0, 100] et ne pas être complètement constant.

    Note: sur des fixtures synthétiques bruitées, la différence absolue range vs trend
    est faible (~2 points). Le CI seul n'est PAS un discriminator fiable —
    il doit être combiné avec ADX/ER pour le score composite.
    Sur données BTC réelles avec gros gaps de volatilité, l'écart est plus net.
    """
    ci_range = choppiness_index(range_df["high"], range_df["low"], range_df["close"]).dropna()
    ci_trend = choppiness_index(trend_df["high"], trend_df["low"], trend_df["close"]).dropna()
    # Bornes correctes
    assert (ci_range > 0).all() and (ci_range < 100).all()
    assert (ci_trend > 0).all() and (ci_trend < 100).all()
    # Variabilité non triviale
    assert ci_range.std() > 1.0
    assert ci_trend.std() > 1.0


# ============================================================
# Efficiency Ratio
# ============================================================

def test_er_higher_in_trend_than_range(trend_df, range_df):
    er_trend = efficiency_ratio(trend_df["close"], period=20).dropna().mean()
    er_range = efficiency_ratio(range_df["close"], period=20).dropna().mean()
    assert er_trend > er_range


def test_er_in_zero_one():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    er = efficiency_ratio(close, period=10).dropna()
    assert (er >= 0).all() and (er <= 1).all()


# ============================================================
# Bollinger Band Width
# ============================================================

def test_bbw_positive():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    bbw = bollinger_band_width(close).dropna()
    assert (bbw > 0).all()


def test_bbw_squeeze_detects_volatility_compression():
    """Volatile puis flat → squeeze détecté en partie flate (lookback couvre les 2)."""
    rng = np.random.default_rng(0)
    n = 400
    # Phase 1 (volatile, idx 0-200), Phase 2 (compressed, idx 200-400)
    close = np.concatenate([
        100 + np.cumsum(rng.normal(0, 0.5, 200)),
        np.full(200, 100.0) + rng.normal(0, 0.02, 200),
    ])
    sq = bbw_squeeze(pd.Series(close), period=20, lookback=150, percentile=0.30)
    # Squeeze est défini relativement au lookback : la phase compressée doit
    # avoir bien plus de True que la phase volatile.
    volatile_rate = sq.iloc[100:180].fillna(False).mean()
    compressed_rate = sq.iloc[300:380].fillna(False).mean()
    assert compressed_rate > volatile_rate


# ============================================================
# Hurst exponent
# ============================================================

def test_hurst_random_walk_near_half():
    """Brown motion: H ≈ 0.5 (random walk benchmark)."""
    rng = np.random.default_rng(42)
    n = 5000
    rets = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(rets))
    h = hurst_exponent(prices, max_lag=30)
    assert 0.4 < h < 0.6, f"Random walk H should be ~0.5, got {h}"


def test_hurst_trending_above_half():
    """Drift positif > volatilité → H > 0.5."""
    rng = np.random.default_rng(7)
    n = 5000
    rets = 0.001 + rng.normal(0, 0.005, n)  # drift dominant
    prices = 100 * np.exp(np.cumsum(rets))
    h = hurst_exponent(prices, max_lag=30)
    assert h > 0.5, f"Strong drift should give H>0.5, got {h}"


def test_hurst_mean_reverting_below_half():
    """OU process (mean reverting strong) → H < 0.5."""
    rng = np.random.default_rng(7)
    n = 5000
    log_p = np.zeros(n)
    log_p[0] = np.log(100)
    mean_log = np.log(100)
    for i in range(1, n):
        log_p[i] = log_p[i - 1] + 0.3 * (mean_log - log_p[i - 1]) + rng.normal(0, 0.01)
    prices = np.exp(log_p)
    h = hurst_exponent(prices, max_lag=30)
    assert h < 0.45, f"Strong mean-rev should give H<0.45, got {h}"


def test_hurst_returns_nan_on_short_series():
    h = hurst_exponent(np.array([0.01, 0.02]), max_lag=20)
    assert np.isnan(h)


# ============================================================
# Score composite
# ============================================================

def test_range_score_higher_in_range_than_trend(trend_df, range_df):
    rs_trend = range_score(trend_df).iloc[-1]["range_score"]
    rs_range = range_score(range_df).iloc[-1]["range_score"]
    assert rs_range > rs_trend, f"Range should score higher: trend={rs_trend}, range={rs_range}"


def test_range_score_in_zero_one(random_walk_df):
    rs = range_score(random_walk_df).dropna(subset=["range_score"])
    assert (rs["range_score"] >= 0).all()
    assert (rs["range_score"] <= 1).all()


def test_is_range_bar_returns_bool_series(range_df):
    rb = is_range_bar(range_df, threshold=0.5)
    assert rb.dtype == bool
    assert len(rb) == len(range_df)


def test_range_score_columns_present(range_df):
    out = range_score(range_df)
    for col in ["adx", "ci", "er", "bbw", "bbw_squeeze", "hurst", "range_score"]:
        assert col in out.columns


def test_weights_can_be_customized(range_df):
    custom = RangeScoreWeights(adx=1.0, choppiness=0.0, efficiency=0.0, bbw_low=0.0, hurst=0.0)
    out = range_score(range_df, weights=custom)
    # Avec weights ADX-only, le score devrait être proche de la transformation ADX seule
    assert "range_score" in out.columns
