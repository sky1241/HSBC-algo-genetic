"""Backtest engine integrating risk sizing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .risk_sizing import (
    apply_funding_gating,
    atr_mult_by_regime,
    compute_atr,
    position_size,
)


@dataclass(slots=True)
class BacktestResult:
    """Container holding the main backtest artefacts."""

    equity_curve: pd.Series
    position_history: pd.Series
    max_drawdown: float
    fees_paid: float
    funding_paid: float
    stopped_for_mdd: bool
    net_equity: float
    daily_stop_count: int


REQUIRED_COLUMNS = {
    "high",
    "low",
    "close",
    "signal",
    "return",
    "fee_rate",
    "funding_rate",
}


def run_backtest(
    data: pd.DataFrame,
    regime_labels: pd.Series | Mapping | None,
    atr_period: int,
    regime_cfg: Mapping,
    *,
    initial_equity: float = 1000.0,
    cap_leverage: float = 1.0,
    daily_loss_cap: float = 0.05,
    funding_thresholds: Optional[Mapping[str, float]] = None,
) -> BacktestResult:
    """Execute a simplified backtest using ATR-based risk sizing.

    Parameters
    ----------
    data:
        OHLC dataframe with additional columns ``signal`` (direction), ``return``
        (period return as decimal), ``fee_rate`` and ``funding_rate``.
    regime_labels:
        Regime classification aligned with ``data``. ``None`` means a constant
        default multiplier.
    atr_period:
        Period used for ATR computation.
    regime_cfg:
        Mapping from regime labels to ATR multipliers, must include ``default``.
    initial_equity:
        Starting portfolio value.
    cap_leverage:
        Maximum leverage allowed when sizing positions.
    daily_loss_cap:
        Maximum tolerated loss for a single day expressed as a fraction of the
        day's starting equity. ``0`` disables the guard-rail.
    funding_thresholds:
        Optional bounds used to gate exposure based on funding costs.
    """
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"data is missing required columns: {missing_str}")

    df = data.copy()

    atr = compute_atr(df, atr_period)

    if regime_labels is None:
        regime_series = pd.Series(["default"] * len(df), index=df.index)
    elif isinstance(regime_labels, pd.Series):
        regime_series = regime_labels.reindex(df.index)
        regime_series = regime_series.ffill().bfill()
        regime_series = regime_series.fillna("default")
    else:
        regime_series = pd.Series(list(regime_labels), index=df.index)

    multipliers = atr_mult_by_regime(regime_series, regime_cfg)
    funding_gate = apply_funding_gating(df["funding_rate"], funding_thresholds)

    equity = float(initial_equity)
    peak_equity = equity
    prev_position = 0.0
    mdd_stop = False
    daily_stop_triggered = False
    daily_stop_count = 0
    fees_paid = 0.0
    funding_paid = 0.0

    prev_day = None
    day_start_equity = equity
    daily_loss = 0.0

    equity_values: list[float] = []
    position_values: list[float] = []

    for idx, row in df.iterrows():
        # Reset day-level guards
        if isinstance(idx, pd.Timestamp):
            current_day = idx.normalize()
        else:
            current_day = idx
        if current_day != prev_day:
            daily_loss = 0.0
            daily_stop_triggered = False
            day_start_equity = equity
            prev_day = current_day

        signal = float(row.get("signal", 0.0))
        period_return = float(row.get("return", 0.0))
        fee_rate = abs(float(row.get("fee_rate", 0.0)))
        funding_rate = float(row.get("funding_rate", 0.0))
        gate_allowed = bool(funding_gate.loc[idx] >= 0.5)

        if not gate_allowed:
            target_position = 0.0
        elif mdd_stop or (daily_stop_triggered and signal != 0.0):
            target_position = 0.0
        else:
            atr_value = float(atr.loc[idx])
            mult = float(multipliers.loc[idx])
            target_position = position_size(equity, atr_value, mult, cap_leverage)
            target_position *= signal

        pnl = prev_position * period_return
        funding_cost = abs(prev_position) * funding_rate if gate_allowed else 0.0
        position_change = target_position - prev_position
        fee_cost = abs(position_change) * fee_rate

        net_change = pnl - fee_cost - funding_cost

        equity += net_change
        fees_paid += fee_cost
        funding_paid += funding_cost
        daily_loss += net_change

        equity_values.append(equity)
        position_values.append(target_position)

        prev_position = target_position

        if equity > peak_equity:
            peak_equity = equity

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = 1.0 - (equity / peak_equity)
        if drawdown > 0.5:
            mdd_stop = True
            prev_position = 0.0
            position_values[-1] = 0.0

        if daily_loss_cap > 0 and not daily_stop_triggered:
            loss_limit = -daily_loss_cap * day_start_equity
            if daily_loss <= loss_limit:
                daily_stop_triggered = True
                daily_stop_count += 1
                prev_position = 0.0
                position_values[-1] = 0.0

    equity_series = pd.Series(equity_values, index=df.index, dtype=float)
    position_series = pd.Series(position_values, index=df.index, dtype=float)

    running_max = equity_series.cummax()
    drawdowns = 1.0 - equity_series / running_max.replace(0, np.nan)
    max_drawdown = float(drawdowns.max(skipna=True) or 0.0)

    return BacktestResult(
        equity_curve=equity_series,
        position_history=position_series,
        max_drawdown=max_drawdown,
        fees_paid=float(fees_paid),
        funding_paid=float(funding_paid),
        stopped_for_mdd=bool(mdd_stop),
        net_equity=float(equity_series.iloc[-1]),
        daily_stop_count=int(daily_stop_count),
    )
