import numpy as np
import pandas as pd

from src import risk_sizing


def _make_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="2h")
    base = np.linspace(100, 110, len(index))
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": np.linspace(50, 60, len(index)),
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def test_simulate_strategy_includes_costs():
    df = _make_df()
    params = {"tenkan": 3, "kijun": 5, "shift": 2, "atr_mult": 1.5}
    funding = pd.Series(0.001, index=df.index)

    gross_returns = risk_sizing.simulate_strategy(df, params)
    net_returns = risk_sizing.simulate_strategy(
        df,
        params,
        fee=0.001,
        funding=funding,
        max_drawdown=0.5,
    )
    assert gross_returns.sum() >= net_returns.sum()


def test_enforce_max_drawdown_stops_series():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.02, -0.25, 0.03, 0.04, 0.05], index=index)
    adjusted = risk_sizing.enforce_max_drawdown(returns, 0.1)
    assert adjusted.loc[index[1]] == returns.loc[index[1]]
    assert (adjusted.loc[index[2:]] == 0.0).all()


def test_atr_mult_by_regime_uses_default_for_unknown():
    labels = pd.Series(["trend", "range", "transition", "unknown"], index=pd.RangeIndex(4))
    cfg = {"trend": 1.5, "range": 0.8, "default": 1.0}
    multipliers = risk_sizing.atr_mult_by_regime(labels, cfg)
    assert multipliers.loc[0] == 1.5
    assert multipliers.loc[1] == 0.8
    assert multipliers.loc[2] == 1.0
    assert multipliers.loc[3] == 1.0


def test_position_size_handles_invalid_inputs():
    assert risk_sizing.position_size(1000.0, 0.0, 1.0, 2.0) == 0.0
    assert risk_sizing.position_size(0.0, 1.0, 1.0, 2.0) == 0.0
    value = risk_sizing.position_size(1000.0, 10.0, 2.0, 1.0)
    assert value == 50.0


def test_apply_funding_gating_thresholds():
    index = pd.date_range("2024-01-01", periods=4, freq="8h")
    series = pd.Series([0.01, -0.02, 0.03, 0.0], index=index)
    gating = risk_sizing.apply_funding_gating(series, {"min": -0.01, "max": 0.02})
    expected = pd.Series([1.0, 0.0, 0.0, 1.0], index=index)
    pd.testing.assert_series_equal(gating, expected)
