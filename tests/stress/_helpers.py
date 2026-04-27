"""Shared helpers for stress-test scenarios.

We deliberately reuse :func:`src.risk_sizing.simulate_strategy` and
:func:`src.stats_eval.compute_metrics` so the stress runs use the same
backtest engine as the rest of the codebase. No new strategy is implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Allow running tests with /home/ludov/HSBC-algo-genetic on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import risk_sizing, stats_eval  # noqa: E402


# Ichimoku K3 1D phase rotation baseline params (matches src.optimizer.BASELINE_PARAMS).
ICHIMOKU_K3_PARAMS: Dict[str, float] = {
    "tenkan": 9,
    "kijun": 26,
    "senkou_b": 52,
    "shift": 26,
    "atr_mult": 2.0,
}

# Stress data locations (2h timeframe — Binance BTC/USDT).
STRESS_DATA_DIR = ROOT / "data" / "stress"

SCENARIOS: Dict[str, Dict[str, str]] = {
    "luna_2022": {
        "csv": "btc_usdt_2022_luna.csv",
        "label": "LUNA / UST depeg (2022-04-15 -> 2022-06-30)",
        "expected_drop_pct": 55.0,
    },
    "ftx_2022": {
        "csv": "btc_usdt_2022_ftx.csv",
        "label": "FTX collapse (2022-10-15 -> 2022-11-30)",
        "expected_drop_pct": 25.0,
    },
    "covid_2020": {
        "csv": "btc_usdt_2020_covid.csv",
        "label": "COVID flash crash (2020-03-01 -> 2020-03-31)",
        "expected_drop_pct": 50.0,
    },
}

# Periods per year for 2h bars (12 bars/day * 365).
PERIODS_PER_YEAR_2H = 12 * 365


def load_scenario(name: str) -> pd.DataFrame:
    cfg = SCENARIOS[name]
    path = STRESS_DATA_DIR / cfg["csv"]
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df


def liquidation_threshold(leverage: float, maintenance_margin: float = 0.005) -> float:
    """Approximate adverse move (fraction) that liquidates a position.

    Simplified model: liq when adverse move exceeds 1/L minus maintenance margin.
    For L=3 -> ~33%, L=5 -> ~20%, L=10 -> ~10%.
    """
    return max(1.0 / float(leverage) - float(maintenance_margin), 0.0)


def simulate_with_leverage(
    df: pd.DataFrame,
    params: Dict[str, float],
    leverage: float,
    *,
    realistic_costs: bool = False,
    funding_rates: Optional[pd.Series] = None,
) -> pd.Series:
    """Run the canonical Ichimoku-like strategy with a leverage multiplier.

    We use ``realistic_costs=False`` by default in stress tests so leverage
    purely amplifies P&L: the goal is to expose tail-risk, not measure
    fee impact (already covered in B5 / cost_model tests).
    """
    raw = risk_sizing.simulate_strategy(
        df,
        params,
        realistic_costs=realistic_costs,
        leverage=leverage,
        funding_rates=funding_rates,
    )
    if not realistic_costs:
        raw = raw * float(leverage)
    return raw


def equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def liquidation_event(returns: pd.Series, leverage: float) -> tuple[bool, Optional[pd.Timestamp]]:
    """Detect first bar where compounded equity drops below the liq threshold.

    A trade is "liquidated" if the leveraged drawdown from the previous peak
    exceeds 1/L. We use rolling peak (per-bar drawdown of the equity curve).
    """
    eq = equity_from_returns(returns)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    threshold = -liquidation_threshold(leverage)
    triggered = dd <= threshold
    if triggered.any():
        first_idx = triggered.idxmax()
        return True, first_idx
    return False, None


def stress_metrics(returns: pd.Series, leverage: float) -> Dict[str, float]:
    """Compute the headline stress metrics for one (scenario, leverage) run."""
    base = stats_eval.compute_metrics(returns, PERIODS_PER_YEAR_2H)
    eq = equity_from_returns(returns)
    final_equity = float(eq.iloc[-1]) if len(eq) else 1.0
    peak = eq.cummax()
    dd_curve = eq / peak - 1.0
    max_dd = float(dd_curve.min()) if len(dd_curve) else 0.0
    n_stops = int((returns < 0).sum())
    liquidated, liq_ts = liquidation_event(returns, leverage)
    return {
        "leverage": float(leverage),
        "final_equity": final_equity,
        "pnl_pct": (final_equity - 1.0) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sortino": float(base.get("sortino", float("nan"))),
        "ulcer_pct": float(base.get("ulcer", float("nan"))) * 100.0,
        "dd_duration_bars": float(base.get("dd_duration", float("nan"))),
        "n_negative_bars": n_stops,
        "liquidated": bool(liquidated),
        "liq_timestamp": str(liq_ts) if liq_ts is not None else "",
    }


def empirical_liq_probability(df: pd.DataFrame, threshold_pct: float, window_bars: int) -> float:
    """Empirical fraction of rolling windows where peak->trough exceeds ``threshold_pct``."""
    if len(df) < window_bars + 1:
        return float("nan")
    close = df["close"].astype(float).to_numpy()
    n = len(close)
    hits = 0
    total = 0
    for i in range(0, n - window_bars):
        seg = close[i:i + window_bars]
        peak = seg.max()
        trough = seg.min()
        if peak <= 0:
            continue
        max_drop = (trough - peak) / peak * 100.0  # negative
        if -max_drop >= threshold_pct:
            hits += 1
        total += 1
    return hits / total if total > 0 else float("nan")


def synth_flash_crash(start_price: float = 30000.0, n_pre: int = 24) -> pd.DataFrame:
    """Build a synthetic 2h OHLC frame with a -20% crash in 1h then a +10% bounce.

    Layout (2h bars except where noted):
      - n_pre bars of flat noise.
      - 1 bar (compressed to ~1h equivalent) with -20% close.
      - 12 bars (24h) recovering ~+10%.
    """
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-06-01", periods=n_pre + 1 + 12, freq="2h", tz="UTC")
    closes = np.empty(len(times), dtype=float)
    closes[:n_pre] = start_price * (1.0 + rng.normal(0, 0.002, n_pre)).cumprod()
    crash_close = closes[n_pre - 1] * 0.80  # -20%
    closes[n_pre] = crash_close
    # +10% recovery from crash close, distributed over 12 bars.
    bounce_target = crash_close * 1.10
    ramp = np.linspace(crash_close, bounce_target, 13)[1:]
    closes[n_pre + 1:] = ramp
    opens = np.concatenate(([closes[0]], closes[:-1]))
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
        },
        index=pd.Index(times, name="timestamp"),
    )
    return df


def synth_funding_spike(
    base_returns: pd.Series,
    spike_periods_per_8h: float = 0.005,
    duration_bars: int = 24,
    start_idx: int = 5,
) -> pd.Series:
    """Apply a funding spike of ``spike_periods_per_8h`` per 8h slot for ``duration_bars``.

    Returns the *post-funding* per-bar return series (negative drag if long).
    For 2h bars, each 8h slot covers 4 bars, so per-bar drag = spike / 4.
    """
    funding = pd.Series(0.0, index=base_returns.index)
    per_bar = spike_periods_per_8h / 4.0
    end_idx = min(start_idx + duration_bars, len(funding))
    funding.iloc[start_idx:end_idx] = -per_bar  # long pays funding when positive funding rate
    return base_returns + funding
