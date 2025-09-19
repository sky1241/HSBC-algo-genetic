from __future__ import annotations

import numpy as np


def daily_loss_threshold(atr: float, k: float) -> float:
    """Return daily loss threshold as k * ATR.

    If ATR is NaN or None, return infinity to disable the limit.
    """
    if atr is None or (isinstance(atr, float) and np.isnan(atr)):
        return float("inf")
    return k * atr


def lfp_position_multiplier(
    lfp_ratio: float,
    *,
    calm_threshold: float = 0.5,
    trend_threshold: float = 0.75,
    calm_multiplier: float = 0.6,
    trend_multiplier: float = 1.4,
) -> float:
    """Map a Welch ``LFP_ratio`` to a position sizing multiplier.

    The defaults assume ``compute_welch_features`` outputs values in [0, 1].
    Low ratios (energy spread across frequencies) shrink risk, while higher
    ratios (dominant low-frequency structure) expand it. Values between
    ``calm_threshold`` and ``trend_threshold`` are linearly interpolated.
    """

    if not isinstance(lfp_ratio, (float, int)) or not np.isfinite(lfp_ratio):
        return 1.0
    if lfp_ratio <= calm_threshold:
        return float(calm_multiplier)
    if lfp_ratio >= trend_threshold:
        return float(trend_multiplier)
    slope = (trend_multiplier - calm_multiplier) / (trend_threshold - calm_threshold)
    return float(calm_multiplier + slope * (lfp_ratio - calm_threshold))


def lfp_regime_bucket(
    lfp_ratio: float,
    *,
    thresholds: tuple[float, float] = (0.55, 0.75),
) -> str:
    """Classify market regime from ``LFP_ratio`` for future HMM alignment."""

    if not isinstance(lfp_ratio, (float, int)) or not np.isfinite(lfp_ratio):
        return "unknown"
    low, high = thresholds
    if lfp_ratio < low:
        return "volatile"
    if lfp_ratio < high:
        return "balanced"
    return "trending"
