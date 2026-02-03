"""Minimal risk sizing helpers used by the walk-forward engine."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def simulate_strategy(df: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    """Return a series of strategy returns under a simple Ichimoku-like rule."""

    close = df["close"].astype(float)
    tenkan = close.rolling(int(params["tenkan"]), min_periods=1).mean()
    kijun = close.rolling(int(params["kijun"]), min_periods=1).mean()
    signal = np.sign(tenkan - kijun)
    lag = max(int(params.get("shift", 1)) // 2, 1)
    signal = signal.shift(lag).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    scale = 1.0 / max(float(params.get("atr_mult", 1.0)), 1.0)
    strategy_returns = signal * returns * scale
    return strategy_returns


def run_phase_strategy(
    df: pd.DataFrame,
    phases: pd.Series,
    params_by_phase: Dict[str, Dict[str, float]],
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Simulate a phase-aware strategy.

    Returns
    -------
    tuple
        ``(global_returns, per_phase_returns)`` where ``per_phase_returns`` is a
        mapping from phase label to a series aligned with ``df.index``.
    """

    global_returns = pd.Series(0.0, index=df.index, dtype=float)
    per_phase: dict[str, pd.Series] = {}
    for phase, params in params_by_phase.items():
        mask = phases == phase
        if not mask.any():
            continue
        df_phase = df.loc[mask]
        phase_returns = simulate_strategy(df_phase, params)
        phase_returns = phase_returns.reindex(df.index, fill_value=0.0)
        per_phase[phase] = phase_returns
        global_returns = global_returns.add(phase_returns, fill_value=0.0)
    return global_returns, per_phase


__all__ = ["simulate_strategy", "run_phase_strategy"]
