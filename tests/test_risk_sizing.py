import pandas as pd
import pytest

from src.backtest import run_backtest


def test_position_size_changes_with_regime():
    index = pd.date_range("2023-01-01", periods=4, freq="D")
    data = pd.DataFrame(
        {
            "high": [110, 111, 112, 113],
            "low": [100, 101, 102, 103],
            "close": [105, 106, 107, 108],
            "signal": [1, 1, 1, 1],
            "return": [0.0, 0.0, 0.0, 0.0],
            "fee_rate": [0.0, 0.0, 0.0, 0.0],
            "funding_rate": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    regimes = pd.Series(["bull", "bear", "bull", "bear"], index=index)
    regime_cfg = {"default": 1.0, "bull": 1.0, "bear": 2.0}

    result = run_backtest(
        data,
        regimes,
        atr_period=3,
        regime_cfg=regime_cfg,
        initial_equity=10_000.0,
        cap_leverage=1.0,
        daily_loss_cap=0.05,
    )

    bull_positions = result.position_history[regimes == "bull"]
    bear_positions = result.position_history[regimes == "bear"]

    assert bull_positions.mean() > bear_positions.mean()
    assert not bull_positions.eq(bear_positions.iloc[0]).all()


def test_funding_costs_reduce_equity():
    index = pd.date_range("2023-02-01", periods=3, freq="D")
    data = pd.DataFrame(
        {
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "signal": [1, 1, 1],
            "return": [0.0, 0.0, 0.0],
            "fee_rate": [0.0, 0.0, 0.0],
            "funding_rate": [0.001, 0.001, 0.001],
        },
        index=index,
    )
    regimes = pd.Series(["neutral"] * len(index), index=index)
    regime_cfg = {"default": 1.0, "neutral": 1.0}

    result = run_backtest(
        data,
        regimes,
        atr_period=2,
        regime_cfg=regime_cfg,
        initial_equity=1_000.0,
        cap_leverage=1.0,
        daily_loss_cap=0.0,
    )

    expected_funding = (
        result.position_history.shift(1).fillna(0.0).abs() * data["funding_rate"]
    ).sum()

    assert result.funding_paid == pytest.approx(expected_funding)
    assert result.net_equity == pytest.approx(1_000.0 - expected_funding)
    assert result.funding_paid > 0
