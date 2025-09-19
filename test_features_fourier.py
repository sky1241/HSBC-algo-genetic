import numpy as np
import pandas as pd
import pytest

from src.features_fourier import compute_welch_features


def test_compute_welch_features_detects_dominant_period_and_lfp() -> None:
    fs_per_day = 12.0
    dominant_period = 64  # bars (≈5.33 days at 12 bars/day)
    n = 900
    index = pd.date_range("2024-01-01", periods=n, freq="2h")
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(42)
    signal = 2.0 * np.sin(2 * np.pi * t / dominant_period)
    low_component = 0.6 * np.sin(2 * np.pi * t / (dominant_period * 3))
    high_component = 0.25 * np.sin(2 * np.pi * t / 12.0)
    noise = 0.05 * rng.standard_normal(n)
    close = pd.Series(signal + low_component + high_component + noise, index=index)

    features = compute_welch_features(
        close,
        fs_per_day=fs_per_day,
        nperseg_grid=[64, 96, 128],
        noverlap_ratio=(0.5, 0.75),
    )

    recent = features[features["P1_period"].notna()].iloc[-1]
    assert recent["P1_period"] == pytest.approx(dominant_period, rel=0.2)
    assert 0.5 < recent["LFP_ratio"] <= 1.0

