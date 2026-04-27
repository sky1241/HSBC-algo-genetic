"""Tests pour regime_filter (filtre ER+ADX avant entrée Ichimoku)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from services.regime_filter import (
    RegimeFilterConfig,
    adx_value,
    efficiency_ratio_value,
    filter_entry_signals_inplace,
    is_trend_regime,
)


def _trend_df(n=300):
    rng = np.random.default_rng(42)
    drift = np.linspace(0, 0.5, n)
    noise = rng.normal(0, 0.005, n).cumsum()
    close = 100 * np.exp(drift + noise)
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({"high": high, "low": low, "close": close})


def _range_df(n=300):
    rng = np.random.default_rng(42)
    log_p = np.zeros(n)
    log_p[0] = np.log(100)
    for i in range(1, n):
        log_p[i] = log_p[i - 1] + 0.05 * (np.log(100) - log_p[i - 1]) + rng.normal(0, 0.01)
    close = np.exp(log_p)
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    return pd.DataFrame({"high": high, "low": low, "close": close})


# ============================================================
# adx_value, efficiency_ratio_value
# ============================================================

def test_adx_value_higher_in_trend():
    a_trend = adx_value(_trend_df()['high'], _trend_df()['low'], _trend_df()['close'])
    a_range = adx_value(_range_df()['high'], _range_df()['low'], _range_df()['close'])
    assert a_trend > a_range


def test_er_value_higher_in_trend():
    er_trend = efficiency_ratio_value(_trend_df()['close'], period=30)
    er_range = efficiency_ratio_value(_range_df()['close'], period=30)
    assert er_trend > er_range


def test_er_in_zero_one():
    er = efficiency_ratio_value(_trend_df()['close'], period=30)
    assert 0.0 <= er <= 1.0


# ============================================================
# is_trend_regime
# ============================================================

def test_is_trend_disabled_always_true():
    cfg = RegimeFilterConfig(enabled=False)
    assert is_trend_regime(_range_df(), cfg) is True


def test_is_trend_recognizes_trend_market():
    cfg = RegimeFilterConfig(enabled=True, min_er=0.20, min_adx=20.0)
    assert is_trend_regime(_trend_df(), cfg) is True


def test_is_trend_blocks_range_market():
    cfg = RegimeFilterConfig(enabled=True, min_er=0.30, min_adx=22.0)
    assert is_trend_regime(_range_df(), cfg) is False


def test_is_trend_safe_fallback_when_insufficient_data():
    cfg = RegimeFilterConfig(enabled=True, er_period=30, adx_period=14)
    short_df = _trend_df(n=10)
    # Insuffisant → fallback True (ne pas bloquer)
    assert is_trend_regime(short_df, cfg) is True


# ============================================================
# filter_entry_signals_inplace
# ============================================================

def test_filter_disabled_does_nothing():
    cfg = RegimeFilterConfig(enabled=False)
    df = _range_df()
    df['signal_long'] = False
    df['signal_short'] = False
    df.iloc[-1, df.columns.get_loc('signal_long')] = True
    info = filter_entry_signals_inplace(df, cfg)
    assert info['status'] == 'disabled'
    assert df.iloc[-1]['signal_long'] is True or df.iloc[-1]['signal_long'] == True
    assert info['n_signals_neutralized'] == 0


def test_filter_neutralizes_in_range():
    cfg = RegimeFilterConfig(enabled=True, min_er=0.30, min_adx=22.0)
    df = _range_df()
    df['signal_long'] = False
    df['signal_short'] = False
    df.iloc[-1, df.columns.get_loc('signal_long')] = True
    info = filter_entry_signals_inplace(df, cfg)
    assert info['status'] == 'range'
    assert df.iloc[-1]['signal_long'] == False  # neutralisé
    assert info['n_signals_neutralized'] == 1


def test_filter_keeps_signals_in_trend():
    cfg = RegimeFilterConfig(enabled=True, min_er=0.20, min_adx=20.0)
    df = _trend_df()
    df['signal_long'] = False
    df['signal_short'] = False
    df.iloc[-1, df.columns.get_loc('signal_long')] = True
    info = filter_entry_signals_inplace(df, cfg)
    assert info['status'] == 'trend'
    assert df.iloc[-1]['signal_long'] == True
    assert info['n_signals_neutralized'] == 0


def test_filter_returns_er_and_adx_values():
    cfg = RegimeFilterConfig(enabled=True)
    df = _trend_df()
    df['signal_long'] = False
    df['signal_short'] = False
    info = filter_entry_signals_inplace(df, cfg)
    assert info['er'] is not None
    assert info['adx'] is not None
    assert 0.0 <= info['er'] <= 1.0
