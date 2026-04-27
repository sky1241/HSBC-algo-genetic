"""Minimal risk sizing helpers used by the walk-forward engine.

The pure-strategy ``simulate_strategy`` returns gross per-period returns under
a simplified Ichimoku-like rule. Since the audit (B5) flagged that the
backtest did not subtract fees / funding / slippage, this module now also
exposes a ``realistic_costs`` flag that routes returns through
``cost_model.apply_costs_to_returns``. The default is **True**: callers that
need raw gross returns must opt in via ``realistic_costs=False``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import cost_model


def _build_signal(df: pd.DataFrame, params: Dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """Return ``(signal, returns)`` where signal is the held position."""
    close = df["close"].astype(float)
    tenkan = close.rolling(int(params["tenkan"]), min_periods=1).mean()
    kijun = close.rolling(int(params["kijun"]), min_periods=1).mean()
    signal = np.sign(tenkan - kijun)
    lag = max(int(params.get("shift", 1)) // 2, 1)
    signal = signal.shift(lag).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    return signal, returns


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Approximate ATR if high/low/close are present, else fall back to std."""
    if {"high", "low", "close"}.issubset(df.columns):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window, min_periods=1).mean().bfill()
    # No OHLC: use rolling abs return as proxy.
    close = df["close"].astype(float)
    return (close.pct_change().abs() * close).rolling(window, min_periods=1).mean().fillna(0.0)


def simulate_strategy(
    df: pd.DataFrame,
    params: Dict[str, float],
    *,
    realistic_costs: bool = True,
    fees: Optional[cost_model.BinanceFutureFees] = None,
    funding_rates: Optional[pd.Series] = None,
    leverage: float = 1.0,
    depth_usdt: float = 50_000.0,
    is_taker: bool = True,
) -> pd.Series:
    """Return a series of strategy returns under a simple Ichimoku-like rule.

    Parameters
    ----------
    df, params:
        See module docstring. ``params`` keys: ``tenkan``, ``kijun``,
        ``shift``, ``atr_mult``.
    realistic_costs:
        When True (default) deducts fees, funding and slippage via the
        ``cost_model`` module. Set to False to recover the legacy gross
        backtest (used internally by tests / sanity checks).
    fees, funding_rates, leverage, depth_usdt, is_taker:
        Forwarded to ``cost_model.apply_costs_to_returns`` when realistic
        costs are enabled.
    """
    signal, returns = _build_signal(df, params)
    scale = 1.0 / max(float(params.get("atr_mult", 1.0)), 1.0)
    strategy_returns = signal * returns * scale
    if not realistic_costs:
        return strategy_returns
    price = df["close"].astype(float)
    atr = _compute_atr(df)
    return cost_model.apply_costs_to_returns(
        strategy_returns,
        position=signal * scale,
        price=price,
        atr=atr,
        funding_rates=funding_rates,
        fees=fees,
        leverage=leverage,
        depth_usdt=depth_usdt,
        is_taker=is_taker,
    )


def run_phase_strategy(
    df: pd.DataFrame,
    phases: pd.Series,
    params_by_phase: Dict[str, Dict[str, float]],
    *,
    realistic_costs: bool = True,
    fees: Optional[cost_model.BinanceFutureFees] = None,
    funding_rates: Optional[pd.Series] = None,
    leverage: float = 1.0,
    depth_usdt: float = 50_000.0,
    is_taker: bool = True,
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
        phase_returns = simulate_strategy(
            df_phase,
            params,
            realistic_costs=realistic_costs,
            fees=fees,
            funding_rates=funding_rates,
            leverage=leverage,
            depth_usdt=depth_usdt,
            is_taker=is_taker,
        )
        phase_returns = phase_returns.reindex(df.index, fill_value=0.0)
        per_phase[phase] = phase_returns
        global_returns = global_returns.add(phase_returns, fill_value=0.0)
    return global_returns, per_phase


__all__ = ["simulate_strategy", "run_phase_strategy"]
