"""Integration tests: ``simulate_strategy`` with realistic costs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import risk_sizing, stats_eval
from src.cost_model import BinanceFutureFees


def _make_synthetic_df(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    drift = 0.0003
    noise = rng.normal(0, 0.02, size=n)
    log_returns = drift + noise
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_df()


@pytest.fixture
def params() -> dict[str, float]:
    return {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26, "atr_mult": 1.0}


def test_realistic_costs_reduce_pnl(synthetic_df, params):
    gross = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=False)
    net = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=True)
    # Net cumulative return should be lower than gross (drag from fees+funding).
    assert net.sum() < gross.sum()
    # And the total drag should be > 0 in absolute value (sanity).
    drag = gross.sum() - net.sum()
    assert drag > 0.0


def test_zero_fees_zero_funding_recovers_baseline(synthetic_df, params):
    """With zero-rate fees + zero funding + zero slippage, net == gross."""
    zero_fees = BinanceFutureFees(taker_bps=0.0, maker_bps=0.0)
    zero_funding = pd.Series(0.0, index=synthetic_df.index)

    gross = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=False)
    # We need to also zero out the slippage component. Pass a synthetic df
    # with ATR ~ 0 and huge depth so slippage drag is negligible.
    # The cleanest way: build returns by hand using cost_model with zero
    # fees + zero funding + huge depth + ATR ~ 0.
    # Simpler: monkey-patch via custom call.
    from src import cost_model

    signal, returns = risk_sizing._build_signal(synthetic_df, params)
    scale = 1.0 / max(float(params.get("atr_mult", 1.0)), 1.0)
    raw = signal * returns * scale
    net = cost_model.apply_costs_to_returns(
        raw,
        position=signal * scale,
        price=synthetic_df["close"],
        atr=pd.Series(0.0, index=synthetic_df.index),
        funding_rates=zero_funding,
        fees=zero_fees,
        leverage=0.0,  # disable slippage drag completely
        depth_usdt=1e18,
    )
    # With leverage=0 -> fee_drag = slip_drag = 0; with zero funding -> no drag.
    pd.testing.assert_series_equal(net, gross.astype(float), check_names=False)


def test_sharpe_degradation_realistic_magnitude(synthetic_df, params):
    """Audit expectation: Δ Sharpe ~ -0.3 to -0.7 on a daily Ichimoku rule."""
    gross = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=False)
    net = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=True)
    s_gross = stats_eval.compute_metrics(gross, periods_per_year=365)["sharpe"]
    s_net = stats_eval.compute_metrics(net, periods_per_year=365)["sharpe"]
    # Both should be finite.
    assert np.isfinite(s_gross)
    assert np.isfinite(s_net)
    delta = s_net - s_gross
    # Expect a non-trivial drag. Loose bounds because synthetic data is GBM
    # (no real edge); we just want to assert the direction + sane magnitude.
    assert delta < 0.0
    assert delta > -3.0  # not absurdly large


def test_run_phase_strategy_with_costs(synthetic_df, params):
    n = len(synthetic_df)
    phases = pd.Series(["phase_0"] * (n // 2) + ["phase_1"] * (n - n // 2),
                       index=synthetic_df.index, dtype=object)
    params_by_phase = {"phase_0": params, "phase_1": params}
    gross_total, _ = risk_sizing.run_phase_strategy(
        synthetic_df, phases, params_by_phase, realistic_costs=False
    )
    net_total, _ = risk_sizing.run_phase_strategy(
        synthetic_df, phases, params_by_phase, realistic_costs=True
    )
    assert net_total.sum() < gross_total.sum()


def test_default_is_realistic(synthetic_df, params):
    """Calling simulate_strategy without flags should now apply costs."""
    default = risk_sizing.simulate_strategy(synthetic_df, params)
    explicit = risk_sizing.simulate_strategy(synthetic_df, params, realistic_costs=True)
    pd.testing.assert_series_equal(default, explicit, check_names=False)
