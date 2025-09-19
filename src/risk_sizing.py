"""Risk sizing helper utilities.

This module centralises computations related to Average True Range (ATR),
regime-specific leverage adjustments and funding guard-rails used by the
backtest engine.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Return the Average True Range (ATR) for *df*.

    Parameters
    ----------
    df:
        DataFrame with at least ``high``, ``low`` and ``close`` columns.
    period:
        Averaging period used for the Wilder smoothing. Must be positive.

    Returns
    -------
    pandas.Series
        ATR values aligned with ``df``'s index.
    """
    if period <= 0:
        raise ValueError("period must be strictly positive")

    for col in ("high", "low", "close"):
        if col not in df:
            raise KeyError(f"Missing required column '{col}' for ATR computation")

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    true_range = tr_components.max(axis=1, skipna=True)

    atr = true_range.ewm(alpha=1 / float(period), adjust=False).mean()
    atr = atr.ffill()
    return atr.fillna(0.0)


def atr_mult_by_regime(labels: Sequence[Any] | pd.Series,
                       cfg: Mapping[Any, float]) -> pd.Series:
    """Map each regime label to an ATR multiplier.

    ``cfg`` must provide a ``"default"`` entry used whenever a label is
    missing from the mapping. The returned series is aligned with the input
    order.
    """
    if "default" not in cfg:
        raise KeyError("cfg must define a 'default' ATR multiplier")

    if isinstance(labels, pd.Series):
        index = labels.index
        label_series = labels
    else:
        index = pd.RangeIndex(len(labels))
        label_series = pd.Series(labels, index=index)

    def _lookup(label: Any) -> float:
        try:
            mult = cfg[label]
        except KeyError:
            mult = cfg["default"]
        if not np.isfinite(mult) or mult <= 0:
            return float(cfg["default"])
        return float(mult)

    multipliers = label_series.map(_lookup).astype(float)
    return multipliers.reindex(index)


def position_size(equity: float, atr: float, mult: float,
                  cap_leverage: float) -> float:
    """Return position size scaled by ATR and leverage cap.

    The function allocates notional ``equity * cap_leverage`` and scales it by
    the volatility proxy ``atr * mult``. Invalid inputs return ``0.0``.
    """
    if (not np.isfinite(equity)) or equity <= 0:
        return 0.0
    if (not np.isfinite(atr)) or atr <= 0:
        return 0.0
    if (not np.isfinite(mult)) or mult <= 0:
        return 0.0
    if (not np.isfinite(cap_leverage)) or cap_leverage <= 0:
        return 0.0

    risk_unit = atr * mult
    if risk_unit <= 0:
        return 0.0

    notional = equity * cap_leverage
    return float(notional / risk_unit)


def apply_funding_gating(funding_series: pd.Series,
                         thresholds: Mapping[str, float] | None) -> pd.Series:
    """Return a gating series (1.0 allowed, 0.0 blocked) based on funding.

    Parameters
    ----------
    funding_series:
        Series of funding rates/costs expressed per period.
    thresholds:
        Mapping that may define ``"min"`` and/or ``"max"`` tolerances. When
        ``None`` the gate is always open.
    """
    if thresholds is None:
        return pd.Series(1.0, index=funding_series.index, dtype=float)

    lower = float(thresholds.get("min", -np.inf))
    upper = float(thresholds.get("max", np.inf))

    gating = ((funding_series >= lower) & (funding_series <= upper)).astype(float)
    return gating.reindex(funding_series.index).fillna(0.0)
