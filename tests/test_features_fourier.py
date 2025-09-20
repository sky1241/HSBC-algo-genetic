from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features_fourier


def _make_sine_wave(period: int, n_samples: int, freq: str = "2h") -> pd.Series:
    index = pd.date_range("2020-01-01", periods=n_samples, freq=freq)
    x = np.arange(n_samples)
    signal = 100.0 + np.sin(2 * np.pi * x / period)
    return pd.Series(signal, index=index, name="close")


def test_compute_welch_features_detects_period() -> None:
    period_bars = 24
    series = _make_sine_wave(period=period_bars, n_samples=600)
    cfg = features_fourier.FourierConfig(
        fs_per_day=12.0,
        nperseg_grid=(256,),
        noverlap_ratio=0.5,
        lfp_horizon_days=10.0,
    )
    features = features_fourier.compute_welch_features(series, cfg)
    valid = features["P1_period"].dropna()
    assert not valid.empty
    detected = valid.iloc[-1]
    assert pytest.approx(detected, rel=0.2) == period_bars
    lfp = features["LFP_ratio"].dropna().iloc[-1]
    assert 0.0 <= lfp <= 1.0


def test_compute_fourier_features_adds_volatility() -> None:
    series = _make_sine_wave(period=30, n_samples=400)
    df = pd.DataFrame({"close": series})
    cfg = features_fourier.FourierConfig(
        fs_per_day=12.0,
        nperseg_grid=(128,),
        noverlap_ratio=0.5,
        lfp_horizon_days=8.0,
        volatility_window=48,
    )
    features = features_fourier.compute_fourier_features(df, cfg)
    assert {"log_return", "volatility"} <= set(features.columns)
    assert features["volatility"].notna().sum() > 0
