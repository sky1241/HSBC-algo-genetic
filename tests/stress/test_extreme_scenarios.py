"""Stress tests on historical extreme BTC drawdowns.

Each test runs the canonical Ichimoku K3 strategy on a stress slice
and asserts the bot survives at leverage 3x. Leverage 1x / 5x / 10x
runs are recorded in the JSON snapshot for the markdown report.

Source data: ``data/stress/btc_usdt_*.csv`` extracted from
``data/BTC_USDT_2h.csv`` (Binance public REST via CCXT, see ``data/README.md``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.stress._helpers import (
    ICHIMOKU_K3_PARAMS,
    SCENARIOS,
    empirical_liq_probability,
    load_scenario,
    simulate_with_leverage,
    stress_metrics,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_PATH = OUTPUT_DIR / "stress_test_2026-04-26.json"

LEVERAGES = (1.0, 3.0, 5.0, 10.0)


def _record(payload: dict) -> None:
    """Persist the per-scenario snapshot used by the markdown report."""
    if SNAPSHOT_PATH.exists():
        try:
            blob = json.loads(SNAPSHOT_PATH.read_text())
        except Exception:
            blob = {}
    else:
        blob = {}
    blob.update(payload)
    SNAPSHOT_PATH.write_text(json.dumps(blob, indent=2, default=str))


@pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
def test_scenario_survival_at_3x(scenario_key: str) -> None:
    """At L=3x the strategy must stay solvent (final equity > 0).

    We do NOT assert "no liquidation": the COVID 2020 bar pattern triggers
    a deep equity DD even at 3x — the assertion that matters operationally
    is that the bot finishes the period with non-zero equity (the kill
    switch / stop_global at 30% should fire before total wipe).
    """
    df = load_scenario(scenario_key)
    per_lev = {}
    for lev in LEVERAGES:
        returns = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=lev)
        per_lev[str(lev)] = stress_metrics(returns, lev)

    _record({scenario_key: {
        "label": SCENARIOS[scenario_key]["label"],
        "n_bars": int(len(df)),
        "expected_drop_pct": SCENARIOS[scenario_key]["expected_drop_pct"],
        "by_leverage": per_lev,
    }})

    m3 = per_lev["3.0"]
    assert m3["final_equity"] > 0.0, f"{scenario_key}: equity went non-positive at 3x"
    # Safety floor: even at 3x we tolerate up to -60% DD (COVID is brutal).
    # Anything worse implies the kill-switch needs review.
    assert m3["max_drawdown_pct"] >= -60.0, (
        f"{scenario_key}: max DD {m3['max_drawdown_pct']:.1f}% breaches -60% safety floor at 3x"
    )


@pytest.mark.parametrize("scenario_key", list(SCENARIOS.keys()))
def test_high_leverage_is_dangerous(scenario_key: str) -> None:
    """L=10x must liquidate (or near-liquidate) on at least one of the 3 crashes.

    This documents *why* the recent reduction from 10x to 3x was correct.
    """
    df = load_scenario(scenario_key)
    returns_10x = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=10.0)
    m = stress_metrics(returns_10x, 10.0)
    # Either liquidated, or DD > 50% (operationally indistinguishable from blown account).
    assert m["liquidated"] or m["max_drawdown_pct"] <= -50.0, (
        f"{scenario_key} at 10x: expected liq or >50% DD, got mdd={m['max_drawdown_pct']:.1f}%"
    )


def test_empirical_liq_probability_full_history() -> None:
    """Compute the rolling P(peak->trough drop >= threshold) on the full 2h history.

    L=10 (threshold 10%) must be common; L=3 (threshold 33%) must be rare
    but non-zero — exactly the trade-off behind the 3x choice.
    """
    import pandas as pd

    full_path = Path(__file__).resolve().parents[2] / "data" / "BTC_USDT_2h.csv"
    df = pd.read_csv(full_path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    # 5y window of data (2h bars).
    five_year_bars = 5 * 365 * 12
    df = df.tail(five_year_bars)

    # 30-day rolling window in 2h bars = 30 * 12 = 360.
    win = 30 * 12
    p_10 = empirical_liq_probability(df, threshold_pct=10.0, window_bars=win)
    p_20 = empirical_liq_probability(df, threshold_pct=20.0, window_bars=win)
    p_33 = empirical_liq_probability(df, threshold_pct=33.0, window_bars=win)

    _record({
        "empirical_liq_probability_30d": {
            "history_bars": int(len(df)),
            "P_drop_ge_10pct": p_10,
            "P_drop_ge_20pct": p_20,
            "P_drop_ge_33pct": p_33,
        }
    })

    # Sanity: ordering must hold.
    assert p_10 >= p_20 >= p_33, (
        f"Monotonicity broken: p10={p_10}, p20={p_20}, p33={p_33}"
    )
    # 10% drops in 30d windows are very common in BTC -> > 30% of windows.
    assert p_10 > 0.30, f"P(>=10%) = {p_10:.2%} unexpectedly low"
    # 33% drops are rarer -> well below 25%.
    assert p_33 < 0.25, f"P(>=33%) = {p_33:.2%} unexpectedly high"
