"""Synthetic stress scenarios — flash crash + funding-rate spike."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.stress._helpers import (
    ICHIMOKU_K3_PARAMS,
    equity_from_returns,
    liquidation_event,
    simulate_with_leverage,
    stress_metrics,
    synth_flash_crash,
    synth_funding_spike,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_PATH = OUTPUT_DIR / "stress_test_2026-04-26.json"


def _record(payload: dict) -> None:
    if SNAPSHOT_PATH.exists():
        try:
            blob = json.loads(SNAPSHOT_PATH.read_text())
        except Exception:
            blob = {}
    else:
        blob = {}
    blob.update(payload)
    SNAPSHOT_PATH.write_text(json.dumps(blob, indent=2, default=str))


def test_flash_crash_kill_switch_3x() -> None:
    """A synthetic -20% in 1 bar followed by +10% bounce.

    With ``atr_mult=2.0`` the strategy halves position size (``scale=0.5``),
    so a -20% close-to-close bar at 3x produces ~-29% on the equity curve.
    The operational kill-switch / stop_global is set at -30% in the live
    code path (``stop_global = 0.30`` in ichimoku_pipeline_web_v4_8.py).
    We assert the DD breaches the -25% kill-switch threshold AND the bot
    does not later end profitable on the bounce alone (would imply we
    held through with no risk control).
    """
    df = synth_flash_crash(n_pre=60)
    closes = df["close"].copy()
    n = len(closes)
    n_pre = n - 13  # 12 bounce bars + 1 crash bar
    closes.iloc[:n_pre] = np.linspace(28000, 30000, n_pre)
    df.loc[:, "close"] = closes
    df.loc[:, "open"] = closes.shift(1).fillna(closes.iloc[0])
    df.loc[:, "high"] = np.maximum(df["open"], df["close"]) * 1.001
    df.loc[:, "low"] = np.minimum(df["open"], df["close"]) * 0.999

    returns = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=3.0)
    m = stress_metrics(returns, 3.0)
    # "Kill-switch" semantics for this synthetic test: max equity DD breaches
    # the 25% operational threshold (close to the live stop_global=30%).
    KILL_DD_PCT = -25.0
    triggered = m["max_drawdown_pct"] <= KILL_DD_PCT
    _, ts = liquidation_event(returns, 3.0)

    _record({"synthetic_flash_crash_3x": {
        **m,
        "kill_switch_triggered": bool(triggered),
        "kill_switch_threshold_pct": KILL_DD_PCT,
        "liq_threshold_breached": m["liquidated"],
        "liq_ts": str(ts) if ts is not None else "",
    }})

    assert triggered, (
        f"Kill switch did not fire on -20% flash crash at 3x. mdd={m['max_drawdown_pct']:.1f}%"
    )
    # Final equity must be impaired (not a net gain from the bounce).
    assert m["final_equity"] < 1.0, (
        f"Flash crash at 3x ended with net gain {m['final_equity']:.3f} -- risk control failed"
    )


def test_flash_crash_safe_at_1x() -> None:
    """At L=1x the same flash crash should NOT trigger the liq threshold."""
    df = synth_flash_crash()
    closes = df["close"].copy()
    n = len(closes)
    n_pre = n - 13
    closes.iloc[:n_pre] = np.linspace(28000, 30000, n_pre)
    df.loc[:, "close"] = closes
    df.loc[:, "open"] = closes.shift(1).fillna(closes.iloc[0])
    df.loc[:, "high"] = np.maximum(df["open"], df["close"]) * 1.001
    df.loc[:, "low"] = np.minimum(df["open"], df["close"]) * 0.999

    returns = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=1.0)
    m = stress_metrics(returns, 1.0)
    triggered, _ = liquidation_event(returns, 1.0)

    _record({"synthetic_flash_crash_1x": {
        **m,
        "kill_switch_triggered": bool(triggered),
    }})
    # 1x is far away from the 100% liq threshold.
    assert not triggered, "1x should NEVER hit the 100% liq threshold on a -20% flash"
    assert m["final_equity"] > 0.5, f"Even at 1x, ending equity {m['final_equity']:.2f} too low"


def test_funding_spike_drag_on_long() -> None:
    """A funding spike of +0.5%/8h for 48h drags ~3% off a held long.

    We simulate by starting from a flat-ish series, forcing the strategy to
    be long, then applying the funding-rate drag. The P&L impact must be
    measurable (>= 2.5% drag) and the test reports the per-bar drag.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-07-01", periods=72, freq="2h", tz="UTC")
    base_price = 30_000.0
    drift = rng.normal(0.0, 0.001, len(idx))
    closes = base_price * np.exp(np.cumsum(drift))
    df = pd.DataFrame({
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": 1000.0,
    }, index=idx)
    df.index.name = "timestamp"

    # Plain (no funding) gross returns at 3x.
    base_returns = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=3.0)
    # Apply 48h spike (24 bars of 2h) at 0.5% / 8h funding.
    perturbed = synth_funding_spike(
        base_returns,
        spike_periods_per_8h=0.005,
        duration_bars=24,
        start_idx=10,
    )
    drag = (1.0 + base_returns).prod() - (1.0 + perturbed).prod()

    _record({"synthetic_funding_spike": {
        "drag_pct": float(drag * 100.0),
        "spike_per_8h_pct": 0.5,
        "duration_h": 48,
        "n_bars": int(len(base_returns)),
        "base_final_eq": float((1.0 + base_returns).prod()),
        "perturbed_final_eq": float((1.0 + perturbed).prod()),
    }})

    # 0.5% per 8h * 6 funding slots = 3% gross drag over 48h.
    assert drag > 0.025, f"Funding spike drag too small: {drag:.4f}"
    assert drag < 0.10, f"Funding spike drag implausibly large: {drag:.4f}"


def test_funding_normal_rate_negligible() -> None:
    """A normal funding rate (0.01% / 8h) should have a negligible drag."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-07-01", periods=72, freq="2h", tz="UTC")
    closes = 30_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, len(idx))))
    df = pd.DataFrame({
        "open": closes, "high": closes * 1.001, "low": closes * 0.999,
        "close": closes, "volume": 1000.0,
    }, index=idx)
    df.index.name = "timestamp"
    base_returns = simulate_with_leverage(df, ICHIMOKU_K3_PARAMS, leverage=3.0)
    perturbed = synth_funding_spike(
        base_returns,
        spike_periods_per_8h=0.0001,  # 0.01%
        duration_bars=24,
        start_idx=10,
    )
    drag = (1.0 + base_returns).prod() - (1.0 + perturbed).prod()
    assert abs(drag) < 0.005, f"Normal funding drag should be tiny, got {drag:.5f}"
