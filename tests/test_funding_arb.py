"""Tests pour la stratégie funding rate arb delta-neutral.

Couverture:
- Signal threshold (positif/négatif/seuil exact).
- Simulator: fees comptés correctement (4 legs round-trip).
- Simulator: funding encaissé uniquement aux barres 00/08/16 UTC.
- Simulator: borrow cost activé seulement si signal +1 et allow_short_spot.
- DSR + Hansen SPA exécutables sur returns simulés.
- Smoke test sur data réelle (BTC_USDT_2h + funding_rate_BTCUSDT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.funding_arb import (
    H2_PERIODS_PER_YEAR,
    align_funding_to_h2,
    funding_arb_signal,
    simulate_funding_arb,
)
from src.reality_check import hansen_spa_test
from src.stats_eval import deflated_sharpe_ratio


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ============================================================
# Signal
# ============================================================

def test_signal_threshold_positive_short_perp():
    """Funding > +threshold ⇒ -1 (short perp + long spot)."""
    funding = pd.Series([1e-3, 2e-4, 5e-5], index=pd.date_range("2024-01-01", periods=3, freq="2h"))
    sig = funding_arb_signal(funding, threshold_bps=5.0)  # 5e-4
    assert sig.tolist() == [-1, 0, 0]


def test_signal_threshold_negative_long_perp():
    """Funding < -threshold ⇒ +1 (long perp + short spot)."""
    funding = pd.Series([-1e-3, -1e-4, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="2h"))
    sig = funding_arb_signal(funding, threshold_bps=5.0)
    assert sig.tolist() == [1, 0, 0]


def test_signal_threshold_strict():
    """À exactement +threshold, on n'arme PAS le signal (strict >)."""
    funding = pd.Series([5e-4], index=pd.date_range("2024-01-01", periods=1, freq="2h"))
    sig = funding_arb_signal(funding, threshold_bps=5.0)
    assert sig.tolist() == [0]


# ============================================================
# Simulator: fees
# ============================================================

def _build_dummy(n=20, funding_value=1e-3):
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="2h")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    funding = pd.Series(funding_value, index=idx)
    return df, funding


def test_simulator_fees_no_signal_no_cost():
    """Si funding sous le seuil partout, position reste flat ⇒ pas de frais."""
    df, funding = _build_dummy(funding_value=1e-5)  # 0.1 bp (< 5 bps)
    res = simulate_funding_arb(df, funding, fee_taker_bps=4.0, threshold_bps=5.0)
    assert res.n_trades == 0
    assert res.total_fees == pytest.approx(0.0, abs=1e-12)
    assert (res.position == 0).all()


def test_simulator_fees_round_trip_count():
    """Funding constant > seuil ⇒ on entre une fois, on reste, pas de close.
    Open = 2 legs taker = 8 bps. Funding reçu chaque 00/08/16 UTC."""
    df, funding = _build_dummy(n=12, funding_value=1e-3)  # 12 bars × 2h = 24h = 3 funding events
    res = simulate_funding_arb(df, funding, fee_taker_bps=4.0, threshold_bps=5.0)
    assert res.n_trades == 1
    expected_open_fees = -2 * 4.0 * 1e-4  # -8 bps
    assert res.total_fees == pytest.approx(expected_open_fees, rel=1e-9)
    # Funding reçu: 3 events × 1e-3 = 3e-3 (signal -1, funding +ve)
    # Premier bar à 00:00 + 4 bars plus tard (08:00) + 4 bars plus tard (16:00).
    assert res.total_funding_received == pytest.approx(3 * 1e-3, rel=1e-9)
    assert (res.position == -1).all()


def test_simulator_fees_close_when_signal_flips_to_zero():
    """Signal qui passe à 0 ⇒ on ferme (2 legs taker)."""
    n = 10
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="2h")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    # Signal -1 pendant 5 bars, puis 0
    fund_vals = [1e-3] * 5 + [0.0] * 5
    funding = pd.Series(fund_vals, index=idx)
    res = simulate_funding_arb(df, funding, fee_taker_bps=4.0, threshold_bps=5.0)
    # 1 open (2 legs) + 1 close (2 legs) = 4 legs = 16 bps
    assert res.total_fees == pytest.approx(-16e-4, rel=1e-9)
    assert res.n_trades == 1


def test_simulator_funding_only_on_8h_bars():
    """Funding encaissé seulement quand bar.hour ∈ {0,8,16}."""
    # 1 funding event à 00:00 (premier bar). Bars suivantes 02,04,06 — pas de funding.
    n = 4
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="2h")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    funding = pd.Series(1e-3, index=idx)
    res = simulate_funding_arb(df, funding, fee_taker_bps=4.0, threshold_bps=5.0)
    # Uniquement la barre à 00:00 doit recevoir funding.
    assert res.funding_pnl.iloc[0] == pytest.approx(1e-3, rel=1e-9)
    assert (res.funding_pnl.iloc[1:] == 0).all()


def test_simulator_short_spot_disabled_by_default():
    """Funding très négatif: sans allow_short_spot, on reste flat."""
    df, funding = _build_dummy(funding_value=-1e-3)
    res = simulate_funding_arb(df, funding, threshold_bps=5.0, allow_short_spot=False)
    assert (res.position == 0).all()
    assert res.n_trades == 0


def test_simulator_short_spot_with_borrow_cost():
    """allow_short_spot=True ⇒ position +1 et borrow cost > 0."""
    df, funding = _build_dummy(n=12, funding_value=-1e-3)
    res = simulate_funding_arb(
        df, funding,
        threshold_bps=5.0,
        spot_short_borrow_apr=0.10,
        allow_short_spot=True,
    )
    assert (res.position == 1).all()
    # Borrow: 0.10 * (2/8760) per bar × 12 bars
    expected_borrow = -0.10 * (2.0 / (365.0 * 24.0)) * 12
    assert res.total_borrow == pytest.approx(expected_borrow, rel=1e-9)
    # Funding: signal +1 et funding -1e-3 ⇒ -(+1)*(-1e-3) = +1e-3 reçu × 3 events
    assert res.total_funding_received == pytest.approx(3 * 1e-3, rel=1e-9)


def test_simulator_index_mismatch_raises():
    df, funding = _build_dummy(n=5)
    funding2 = funding.iloc[:3]
    with pytest.raises(ValueError):
        simulate_funding_arb(df, funding2)


# ============================================================
# DSR + SPA on simulator output
# ============================================================

def test_dsr_runs_on_simulated_returns():
    """DSR doit être appelable sur returns du simulator avec n_trials=5."""
    df, funding = _build_dummy(n=200, funding_value=2e-3)
    res = simulate_funding_arb(df, funding, threshold_bps=5.0)
    sr = res.sharpe
    if not np.isfinite(sr):
        pytest.skip("Sharpe NaN (trop peu de variance) — skip DSR")
    # Annualiser le SR pour DSR (psr attend SR cohérent unité). On passe per-bar.
    sr_per_bar = res.returns.mean() / res.returns.std(ddof=0)
    dsr = deflated_sharpe_ratio(
        sharpe_observed=sr_per_bar,
        n_obs=len(res.returns),
        n_trials=5,
    )
    assert 0.0 <= dsr <= 1.0


def test_spa_runs_on_panel_of_thresholds():
    """Hansen SPA sur un panel de 3 thresholds simulés."""
    df, funding = _build_dummy(n=300, funding_value=3e-4)
    panel = []
    for thr in (3.0, 5.0, 7.0):
        res = simulate_funding_arb(df, funding, threshold_bps=thr)
        panel.append(res.returns.to_numpy())
    excess = np.column_stack(panel)
    spa = hansen_spa_test(excess, n_bootstrap=200, block_size_mean=8.0, seed=42)
    assert 0.0 <= spa.p_value_consistent <= 1.0
    assert spa.n_strategies == 3


# ============================================================
# Smoke test on real data
# ============================================================

def test_smoke_on_real_data():
    btc_path = DATA_DIR / "BTC_USDT_2h.csv"
    funding_path = DATA_DIR / "funding_rate_BTCUSDT.csv"
    if not btc_path.exists() or not funding_path.exists():
        pytest.skip("Real data missing — skip smoke")

    df = pd.read_csv(btc_path, parse_dates=["timestamp"]).set_index("timestamp")
    funding_raw = pd.read_csv(funding_path, parse_dates=["timestamp"])
    # Crop to 2020-01 → 2025-08
    df = df.loc["2020-01-01":"2025-08-01"]
    aligned = align_funding_to_h2(funding_raw, df)
    # Drop bars where funding NaN (early period before funding history)
    mask = aligned.notna()
    df = df.loc[mask]
    aligned = aligned.loc[mask]
    res = simulate_funding_arb(df, aligned, fee_taker_bps=4.0, threshold_bps=5.0)
    # Sanity: positions cohérentes, returns finis.
    assert len(res.equity_curve) == len(df)
    assert np.isfinite(res.equity_curve.iloc[-1])
    assert res.metadata["n_bars"] == len(df)
