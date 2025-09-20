"""Risk sizing helpers for the phasenaware walk-forward pipeline."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _safe_series(series: pd.Series) -> pd.Series:
    return series.astype(float).replace([np.inf, -np.inf], np.nan)


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    high = _safe_series(df["high"])
    low = _safe_series(df["low"])
    close = _safe_series(df["close"])
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    tr = ranges.max(axis=1)
    return tr.fillna(0.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    tr = compute_true_range(df)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr.replace(0.0, np.nan)


def position_from_atr(atr: pd.Series, atr_mult: float, cap_leverage: float = 5.0) -> pd.Series:
    leverage = atr_mult / atr
    leverage = leverage.replace([np.inf, -np.inf], np.nan)
    leverage = leverage.clip(lower=0.0, upper=float(cap_leverage))
    return leverage.fillna(0.0)


def _apply_transaction_costs(position: pd.Series, fee: float) -> pd.Series:
    if fee <= 0:
        return pd.Series(0.0, index=position.index)
    turnover = position.diff().abs()
    if turnover.empty:
        return turnover
    turnover.iat[0] = abs(position.iat[0])
    return turnover * float(fee)


def _apply_funding_costs(position: pd.Series, funding: pd.Series | None) -> pd.Series:
    if funding is None:
        return pd.Series(0.0, index=position.index)
    aligned = funding.reindex(position.index).fillna(0.0)
    return position.shift(1).fillna(0.0) * aligned.astype(float)


def enforce_max_drawdown(returns: pd.Series, max_drawdown: float) -> pd.Series:
    if max_drawdown is None or max_drawdown <= 0:
        return returns
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    breach = drawdown < -abs(max_drawdown)
    if not breach.any():
        return returns
    breach_index = breach[breach].index[0]
    adjusted = returns.copy()
    adjusted.loc[adjusted.index > breach_index] = 0.0
    return adjusted


def simulate_strategy(
    df: pd.DataFrame,
    params: Dict[str, float],
    *,
    fee: float = 0.0,
    funding: pd.Series | None = None,
    max_drawdown: float | None = None,
    atr_period: int | None = None,
    cap_leverage: float = 5.0,
) -> pd.Series:
    """Return net strategy returns (fees + funding) under Ichimoku-inspired rules."""

    close = _safe_series(df["close"])
    tenkan = close.rolling(int(params.get("tenkan", 9)), min_periods=1).mean()
    kijun = close.rolling(int(params.get("kijun", 26)), min_periods=1).mean()
    raw_signal = np.sign(tenkan - kijun)
    shift = max(int(params.get("shift", 1)), 1)
    signal = pd.Series(raw_signal, index=close.index).shift(shift).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    atr_mult = float(params.get("atr_mult", 1.0))
    default_atr = max(int(params.get("tenkan", 9)), 5)
    atr_len = int(atr_period or params.get("atr_period", default_atr))
    atr = compute_atr(df, period=atr_len)
    leverage = position_from_atr(atr, atr_mult, cap_leverage=cap_leverage)
    position = signal * leverage
    gross_returns = position.shift(1).fillna(0.0) * returns
    fees = _apply_transaction_costs(position, fee)
    funding_costs = _apply_funding_costs(position, funding)
    strategy_returns = gross_returns - fees - funding_costs
    if max_drawdown is not None:
        strategy_returns = enforce_max_drawdown(strategy_returns, max_drawdown)
    return strategy_returns.fillna(0.0)


def run_phase_strategy(
    df: pd.DataFrame,
    phases: pd.Series,
    params_by_phase: Dict[str, Dict[str, float]],
    *,
    fee: float = 0.0,
    funding: pd.Series | None = None,
    max_drawdown: float | None = None,
    cap_leverage: float = 5.0,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Simulate a phase-aware strategy and aggregate per-phase returns."""

    global_returns = pd.Series(0.0, index=df.index, dtype=float)
    per_phase: dict[str, pd.Series] = {}
    for phase, params in params_by_phase.items():
        mask = phases == phase
        if not mask.any():
            continue
        df_phase = df.loc[mask]
        funding_phase = funding.loc[mask] if funding is not None else None
        phase_returns = simulate_strategy(
            df_phase,
            params,
            fee=fee,
            funding=funding_phase,
            max_drawdown=max_drawdown,
            cap_leverage=cap_leverage,
        )
        phase_returns = phase_returns.reindex(df.index, fill_value=0.0)
        per_phase[phase] = phase_returns
        global_returns = global_returns.add(phase_returns, fill_value=0.0)
    return global_returns, per_phase


__all__ = [
    "compute_true_range",
    "compute_atr",
    "position_from_atr",
    "enforce_max_drawdown",
    "simulate_strategy",
    "run_phase_strategy",
]
