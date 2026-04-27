"""Unit tests for ``src.cost_model``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cost_model import (
    BinanceFutureFees,
    DEFAULT_FUNDING_RATE_8H,
    apply_funding,
    compute_slippage,
    compute_trade_cost,
)


# ---------- compute_trade_cost ---------------------------------------------


def test_trade_cost_taker_default_vip0():
    # 100 USDT * 4 bp = 100 * 4e-4 = 0.04 USDT.
    cost = compute_trade_cost(100.0, is_taker=True)
    assert cost == pytest.approx(-0.04, rel=1e-9)


def test_trade_cost_maker_cheaper_than_taker():
    taker = compute_trade_cost(1000.0, is_taker=True)
    maker = compute_trade_cost(1000.0, is_taker=False)
    assert maker > taker  # less negative
    assert maker == pytest.approx(-0.20, rel=1e-9)


def test_trade_cost_zero_notional():
    assert compute_trade_cost(0.0) == 0.0


def test_trade_cost_custom_fees():
    custom = BinanceFutureFees(taker_bps=10.0, maker_bps=5.0)
    assert compute_trade_cost(1000.0, is_taker=True, fees=custom) == pytest.approx(-1.0)


def test_trade_cost_negative_notional_raises():
    with pytest.raises(ValueError):
        compute_trade_cost(-100.0)


# ---------- apply_funding ---------------------------------------------------


def _daily_index(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_funding_zero_position_does_not_change_returns():
    idx = _daily_index()
    rets = pd.Series(0.001, index=idx)
    pos = pd.Series(0.0, index=idx)
    out = apply_funding(rets, pos)
    pd.testing.assert_series_equal(out, rets, check_names=False)


def test_funding_long_pays_when_rate_positive():
    idx = _daily_index(10)
    rets = pd.Series(0.0, index=idx)
    pos = pd.Series(1.0, index=idx)
    # Default rate per 8h = 1e-4 -> per day = 3e-4 long position pays this.
    out = apply_funding(rets, pos)
    # First bar = 0 (no rate yet, look-ahead protection); subsequent bars =
    # -3e-4.
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[5] == pytest.approx(-3 * DEFAULT_FUNDING_RATE_8H)


def test_funding_short_receives_when_rate_positive():
    idx = _daily_index(10)
    rets = pd.Series(0.0, index=idx)
    pos = pd.Series(-1.0, index=idx)
    out = apply_funding(rets, pos)
    assert out.iloc[5] == pytest.approx(+3 * DEFAULT_FUNDING_RATE_8H)


def test_funding_no_lookahead():
    """Rate posted at t adjusts return at t+1 only."""
    idx = _daily_index(5)
    rets = pd.Series(0.0, index=idx)
    pos = pd.Series(1.0, index=idx)
    rates = pd.Series([0.01, 0.0, 0.0, 0.0, 0.0], index=idx)
    out = apply_funding(rets, pos, funding_rates=rates)
    # The rate at idx[0] should not affect the return at idx[0].
    assert out.iloc[0] == pytest.approx(0.0)


# ---------- compute_slippage -----------------------------------------------


def test_slippage_baseline_small_order():
    # qty * price = 100 USDT, depth = 50k -> impact tiny. atr=0 -> vol tiny.
    slip = compute_slippage(atr=0.0, qty_btc=0.001, price=100_000.0, depth_usdt=50_000.0)
    # baseline 0.5 bp + impact = 0.5 * (100/50_000) = 0.001 bp -> ~0.501 bp
    # of 100 USDT = ~0.005 USDT
    assert slip == pytest.approx(0.005010, rel=1e-3)


def test_slippage_grows_with_size():
    small = compute_slippage(atr=10.0, qty_btc=0.01, price=100_000.0, depth_usdt=50_000.0)
    large = compute_slippage(atr=10.0, qty_btc=1.0, price=100_000.0, depth_usdt=50_000.0)
    assert large > small * 50  # superlinear (impact grows linearly with notional)


def test_slippage_grows_with_volatility():
    low_vol = compute_slippage(atr=5.0, qty_btc=0.1, price=100_000.0)
    high_vol = compute_slippage(atr=500.0, qty_btc=0.1, price=100_000.0)
    assert high_vol > low_vol


def test_slippage_zero_price_safe():
    assert compute_slippage(atr=10.0, qty_btc=1.0, price=0.0) == 0.0


def test_slippage_reference_value():
    """Reference: BTC at 100k, ATR=2000 (2%), qty=0.5 BTC, depth=50k.

    notional = 50_000 USDT, baseline=0.5 bp, impact=0.5*1=0.5 bp,
    vol = 0.2 * (2000/100_000 * 10_000) = 0.2 * 200 = 40 bp.
    Total = 41 bp on 50_000 USDT = 205 USDT.
    """
    slip = compute_slippage(atr=2000.0, qty_btc=0.5, price=100_000.0, depth_usdt=50_000.0)
    assert slip == pytest.approx(205.0, rel=1e-3)
