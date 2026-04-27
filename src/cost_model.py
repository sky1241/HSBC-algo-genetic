"""Realistic cost model for Binance USDM Futures backtests.

Implements **B5** of the quant audit plan:
    1. Trading fees (taker/maker, USDT-margined perpetuals).
    2. Funding rate (paid every 8h UTC, long pays short when rate > 0).
    3. Slippage (spread + market impact + volatility-driven).

All public helpers are pure (no side effects) so they are safe to call from
inside a backtest loop. Reference defaults reflect Binance VIP 0 + observed BTC
funding history (2020-2025) when historical data is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Fees
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinanceFutureFees:
    """Fee schedule (basis points) for Binance USDM Futures.

    Defaults = VIP 0 (40 bps taker = 0.04%, 20 bps maker = 0.02%).
    Note: a "bp" here means 1/100 of a percent, i.e. 1e-4. So
    ``taker_bps = 4.0`` means a 4 bp = 0.04% fee on notional.
    """

    taker_bps: float = 4.0
    maker_bps: float = 2.0


def compute_trade_cost(
    notional_usdt: float,
    is_taker: bool = True,
    fees: Optional[BinanceFutureFees] = None,
) -> float:
    """Return the **negative** USDT cost of opening or closing a single leg.

    Parameters
    ----------
    notional_usdt:
        Position notional in USDT (must be >= 0).
    is_taker:
        ``True`` (default) for market orders, ``False`` for resting limit fills.
    fees:
        Custom fee schedule. Defaults to ``BinanceFutureFees()``.

    Returns
    -------
    float
        Cost in USDT, **negative** (e.g. ``-0.04`` for a 100 USDT taker trade).
    """
    if notional_usdt < 0:
        raise ValueError("notional_usdt must be non-negative")
    if fees is None:
        fees = BinanceFutureFees()
    bps = fees.taker_bps if is_taker else fees.maker_bps
    return -float(notional_usdt) * float(bps) * 1e-4


# ---------------------------------------------------------------------------
# 2. Funding
# ---------------------------------------------------------------------------


# Binance pays funding at 00:00, 08:00, 16:00 UTC.
FUNDING_HOURS_UTC = (0, 8, 16)
# Historical BTC/USDT mean over 2020-2025 ~= +0.01% per 8h. Used as fallback
# when no funding-rate history is provided.
DEFAULT_FUNDING_RATE_8H = 1e-4


def apply_funding(
    returns: pd.Series,
    position: pd.Series,
    funding_rates: Optional[pd.Series] = None,
    default_rate_8h: float = DEFAULT_FUNDING_RATE_8H,
) -> pd.Series:
    """Subtract realized funding payments from a return series.

    Parameters
    ----------
    returns:
        Per-period strategy returns (decimal, e.g. 0.012 = +1.2%).
    position:
        Sign of the position over time: +1 long, -1 short, 0 cash. Index must
        match ``returns``.
    funding_rates:
        Optional series of per-8h funding rates aligned (or ffill-able) on
        ``returns.index``. If ``None``, ``default_rate_8h`` is used as a
        constant proxy (CAUTION: optimistic in low-vol regimes, pessimistic
        when shorts are crowded).
    default_rate_8h:
        Fallback rate used when ``funding_rates`` is missing.

    Notes
    -----
    Long pays short when rate > 0:
        adjusted_return = return - position * funding_rate

    To avoid look-ahead, the rate posted at time ``t`` adjusts the **next**
    period's return (i.e. rate_t -> applied at t+1). When the strategy index
    is daily, all 3 daily fundings are aggregated into one per-day deduction.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(position, pd.Series):
        raise TypeError("position must be a pandas Series")

    pos = position.reindex(returns.index).fillna(0.0).astype(float)

    if funding_rates is None:
        # Build a synthetic funding series: constant rate per 8h slot.
        if isinstance(returns.index, pd.DatetimeIndex):
            # 3 fundings per day: total = 3 * default_rate_8h per day. To stay
            # bar-frequency agnostic, infer slots/bar from index spacing.
            inferred_freq = _infer_funding_periods_per_bar(returns.index)
            total_rate_per_bar = default_rate_8h * inferred_freq
            funding_for_bar = pd.Series(total_rate_per_bar, index=returns.index)
        else:
            funding_for_bar = pd.Series(default_rate_8h, index=returns.index)
    else:
        funding_for_bar = _resample_funding_to_bars(funding_rates, returns.index)

    # No look-ahead: shift rates forward by one bar so the rate observed at t
    # affects the return at t+1 (the trader can only act after observing it).
    funding_for_bar = funding_for_bar.shift(1).fillna(0.0)

    adjusted = returns.astype(float) - pos * funding_for_bar
    return adjusted


def _infer_funding_periods_per_bar(index: pd.DatetimeIndex) -> float:
    """Approximate how many 8h funding slots fall inside one bar.

    1 day -> 3 slots, 8h -> 1 slot, 1h -> 1/8 slot, etc.
    """
    if len(index) < 2:
        return 3.0  # daily fallback
    delta = index[1] - index[0]
    hours = delta.total_seconds() / 3600.0
    if hours <= 0:
        return 3.0
    return hours / 8.0


def _resample_funding_to_bars(
    funding_rates: pd.Series, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Aggregate (sum) intra-bar funding rates to ``target_index`` frequency."""
    if not isinstance(funding_rates.index, pd.DatetimeIndex):
        raise TypeError("funding_rates must have a DatetimeIndex")
    # Forward-fill funding observations onto each bar of ``target_index``.
    # If the original series is sparser than bars, we simply ffill the last
    # known rate. If denser, we sum the rates accumulated over the bar.
    series = funding_rates.sort_index().astype(float)
    if len(target_index) < 2:
        aligned = series.reindex(target_index, method="ffill").fillna(0.0)
        return aligned
    bar_delta = target_index[1] - target_index[0]
    bar_hours = bar_delta.total_seconds() / 3600.0
    if bar_hours >= 8.0:
        # Sum all funding payments that fall inside [t, t+bar) into bar t.
        bins = pd.cut(
            series.index,
            bins=list(target_index) + [target_index[-1] + bar_delta],
            right=False,
            include_lowest=True,
        )
        agg = series.groupby(bins, observed=False).sum()
        agg.index = target_index[: len(agg)]
        aligned = agg.reindex(target_index).fillna(0.0)
    else:
        # Bars are smaller than 8h: forward-fill the last known rate (each bar
        # gets its slice of the 8h payment proportional to bar length).
        aligned = series.reindex(target_index, method="ffill").fillna(0.0)
        aligned = aligned * (bar_hours / 8.0)
    return aligned


# ---------------------------------------------------------------------------
# 3. Slippage
# ---------------------------------------------------------------------------


def compute_slippage(
    atr: float,
    qty_btc: float,
    price: float,
    depth_usdt: float = 50_000.0,
) -> float:
    """Estimate slippage in USDT for a single trade.

    Components (basis points of notional):
        - 0.5 bp baseline (half spread on a tight pair).
        - 0.5 bp * (notional / depth_usdt) for market impact (linear in size).
        - 0.2 bp * (atr / price * 10000) for vol-driven uncertainty.

    Parameters
    ----------
    atr:
        Average True Range in price units (must be >= 0).
    qty_btc:
        Order size in base asset units.
    price:
        Reference (mid) price in USDT.
    depth_usdt:
        Approximate displayed depth at top of book in USDT. 50 kUSDT is a
        conservative BTC/USDT proxy (real top-1bp depth is often higher).

    Returns
    -------
    float
        Slippage cost in **USDT, positive** (subtract from PnL on entry, add
        to entry price for longs / subtract for shorts).
    """
    if price <= 0:
        return 0.0
    qty_btc = abs(float(qty_btc))
    notional = qty_btc * float(price)
    if notional <= 0 or depth_usdt <= 0:
        return 0.0
    spread_bps = 0.5
    impact_bps = 0.5 * (notional / float(depth_usdt))
    vol_bps = 0.2 * (max(float(atr), 0.0) / float(price) * 10_000.0)
    slippage_bps = spread_bps + impact_bps + vol_bps
    return notional * slippage_bps * 1e-4


# ---------------------------------------------------------------------------
# 4. Convenience: combined return adjustment
# ---------------------------------------------------------------------------


def apply_costs_to_returns(
    raw_returns: pd.Series,
    position: pd.Series,
    *,
    price: Optional[pd.Series] = None,
    atr: Optional[pd.Series] = None,
    funding_rates: Optional[pd.Series] = None,
    fees: Optional[BinanceFutureFees] = None,
    leverage: float = 1.0,
    depth_usdt: float = 50_000.0,
    is_taker: bool = True,
) -> pd.Series:
    """Apply fees + funding + slippage to a per-bar return series.

    The ``raw_returns`` are gross strategy returns assuming a $1 notional
    book. Costs are subtracted in **return space**:
        - Trade cost (entry/exit): 2 * fee_bps * |Δ position| (round-trip).
        - Slippage: 2 * slip_bps * |Δ position|.
        - Funding: applied to the held position via ``apply_funding``.

    Setting fees / slippage components to zero recovers the original series
    (used by the unit test suite to verify backward compatibility).
    """
    if fees is None:
        fees = BinanceFutureFees()

    pos = position.reindex(raw_returns.index).fillna(0.0).astype(float)
    delta_pos = pos.diff().abs().fillna(pos.abs())  # first bar = open from cash

    fee_bps = fees.taker_bps if is_taker else fees.maker_bps
    fee_drag = delta_pos * (fee_bps * 1e-4) * leverage

    if price is not None and atr is not None:
        price_aligned = price.reindex(raw_returns.index).ffill()
        atr_aligned = atr.reindex(raw_returns.index).ffill().fillna(0.0)
        # slippage_bps per unit notional (independent of qty since we work in
        # return space): take median qty proxy = leverage * 1 USDT / price.
        slip_bps = (
            0.5
            + 0.2 * (atr_aligned / price_aligned.replace(0, np.nan) * 10_000.0)
        ).fillna(0.5)
        # Market-impact bp depends on notional. In return space we assume the
        # backtest uses unit notional ~= leverage USDT, so impact ~= 0 unless
        # leverage is huge — keep a conservative floor of 0.5 bp.
        impact_bps = (leverage * 1.0 / depth_usdt) * 0.5  # tiny but nonzero
        slip_bps = slip_bps + impact_bps
        slip_drag = delta_pos * (slip_bps * 1e-4) * leverage
    else:
        slip_drag = pd.Series(0.0, index=raw_returns.index)

    funded = apply_funding(raw_returns, pos, funding_rates=funding_rates)
    return funded - fee_drag - slip_drag


__all__ = [
    "BinanceFutureFees",
    "FUNDING_HOURS_UTC",
    "DEFAULT_FUNDING_RATE_8H",
    "compute_trade_cost",
    "apply_funding",
    "compute_slippage",
    "apply_costs_to_returns",
]
