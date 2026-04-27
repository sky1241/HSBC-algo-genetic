"""Tests pour DSR / PSR / Sortino / Ulcer / DD-duration — B3 quant audit.

Références:
- Bailey & López de Prado (2012) — Probabilistic Sharpe Ratio
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio
- Martin (1989) — Ulcer Index
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats_eval import (
    EULER_MASCHERONI,
    compute_metrics,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_h0,
    probabilistic_sharpe_ratio,
)


# ============================================================
# PSR — Probabilistic Sharpe Ratio
# ============================================================

def test_psr_zero_observed_at_zero_threshold_is_half():
    """PSR(SR_obs=0, threshold=0) = 0.5 (50/50, aucune info)."""
    p = probabilistic_sharpe_ratio(sharpe_observed=0.0, n_obs=252)
    assert p == pytest.approx(0.5, abs=0.01)


def test_psr_high_sharpe_returns_high_probability():
    """SR observé largement > 0 sur grand T → proba élevée."""
    p = probabilistic_sharpe_ratio(sharpe_observed=2.0, n_obs=1000)
    assert p > 0.95


def test_psr_negative_sharpe_below_half():
    """SR observé < 0 → proba < 0.5."""
    p = probabilistic_sharpe_ratio(sharpe_observed=-0.5, n_obs=252)
    assert p < 0.5


def test_psr_more_observations_more_confidence():
    """Pour un même SR, plus d'observations ⇒ proba plus extrême."""
    sr_short = probabilistic_sharpe_ratio(0.5, n_obs=20)
    sr_long = probabilistic_sharpe_ratio(0.5, n_obs=2000)
    assert sr_long > sr_short


def test_psr_negative_skew_penalizes():
    """Skew négatif (queue gauche fat) ⇒ var(SR) plus élevée ⇒ proba moins favorable
    quand le SR observé est positif. Petit échantillon pour ne pas saturer à 1."""
    p_normal = probabilistic_sharpe_ratio(0.4, n_obs=30, skew=0.0, kurt=3.0)
    p_neg_skew = probabilistic_sharpe_ratio(0.4, n_obs=30, skew=-1.5, kurt=3.0)
    # Sanity: les deux strictement entre 0 et 1
    assert 0.5 < p_normal < 1.0
    assert 0.5 < p_neg_skew < 1.0
    assert p_neg_skew < p_normal


def test_psr_handles_n_obs_too_small():
    assert np.isnan(probabilistic_sharpe_ratio(1.0, n_obs=1))
    assert np.isnan(probabilistic_sharpe_ratio(1.0, n_obs=0))


# ============================================================
# Expected max Sharpe under H0
# ============================================================

def test_expected_max_increases_with_n_trials():
    """Plus on essaie de configurations, plus le SR max attendu sous H0 est haut."""
    s10 = expected_max_sharpe_under_h0(10)
    s100 = expected_max_sharpe_under_h0(100)
    s10000 = expected_max_sharpe_under_h0(10000)
    assert 0 < s10 < s100 < s10000


def test_expected_max_one_trial_is_zero():
    """N=1 ⇒ pas de data snooping ⇒ seuil = 0."""
    assert expected_max_sharpe_under_h0(1) == 0.0


def test_expected_max_uses_euler_constant():
    """Sanity check: la constante d'Euler est bien utilisée."""
    # Pour N=100, var=1: vérifions à la main
    from scipy.stats import norm as _norm
    expected = (
        (1 - EULER_MASCHERONI) * _norm.ppf(1 - 1 / 100)
        + EULER_MASCHERONI * _norm.ppf(1 - 1 / (100 * np.e))
    )
    assert expected_max_sharpe_under_h0(100) == pytest.approx(expected, rel=1e-9)


# ============================================================
# DSR — Deflated Sharpe Ratio
# ============================================================

def test_dsr_one_trial_equivalent_to_psr_at_zero():
    """N=1 trial ⇒ seuil 0 ⇒ DSR == PSR(0)."""
    psr = probabilistic_sharpe_ratio(1.0, n_obs=252)
    dsr = deflated_sharpe_ratio(1.0, n_obs=252, n_trials=1)
    assert dsr == pytest.approx(psr, rel=1e-9)


def test_dsr_decreases_with_more_trials():
    """Pour un même SR observé, plus de trials testés ⇒ DSR plus bas."""
    sr = 1.5
    d10 = deflated_sharpe_ratio(sr, n_obs=500, n_trials=10)
    d1000 = deflated_sharpe_ratio(sr, n_obs=500, n_trials=1000)
    d100000 = deflated_sharpe_ratio(sr, n_obs=500, n_trials=100000)
    assert d10 > d1000 > d100000


def test_dsr_can_drop_below_significance_with_many_trials():
    """Un Sharpe rapporté élevé après N trials énormes peut ne pas être significatif."""
    # SR observé moyen-bon mais après 100k trials Optuna, var=1 ⇒ seuil élevé
    d = deflated_sharpe_ratio(
        sharpe_observed=1.2, n_obs=252, n_trials=100_000, var_sr_trials=1.0,
    )
    assert d < 0.95  # pas significatif après deflation


def test_dsr_in_zero_one_range():
    """DSR doit toujours être entre 0 et 1 (c'est une probabilité)."""
    for sr in [-2.0, -0.5, 0.0, 0.5, 1.5, 3.0]:
        for n_trials in [1, 100, 10000]:
            d = deflated_sharpe_ratio(sr, n_obs=500, n_trials=n_trials)
            if not np.isnan(d):
                assert 0.0 <= d <= 1.0, f"DSR out of range for sr={sr}, n_trials={n_trials}: {d}"


# ============================================================
# Sortino, Ulcer, DD duration via compute_metrics
# ============================================================

def _make_returns(values, freq="D"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_compute_metrics_includes_sortino_ulcer_dd_duration():
    rs = _make_returns([0.01, -0.005, 0.02, -0.01, 0.005, -0.002, 0.01] * 4)
    m = compute_metrics(rs, periods_per_year=252)
    assert "sortino" in m
    assert "ulcer" in m
    assert "dd_duration" in m
    assert np.isfinite(m["sortino"])
    assert m["ulcer"] >= 0
    assert m["dd_duration"] >= 0


def test_sortino_higher_when_downside_smaller():
    """Distribution asymétrique avec petits drawdowns ⇒ Sortino > Sharpe."""
    upside_heavy = _make_returns([0.05, -0.001, 0.04, -0.001, 0.03] * 10)
    m = compute_metrics(upside_heavy, periods_per_year=252)
    # avec quasi pas de downside, sortino doit largement dépasser sharpe
    assert m["sortino"] > m["sharpe"]


def test_ulcer_index_zero_when_monotonic_up():
    """Une série de returns toujours positifs ⇒ jamais en drawdown ⇒ ulcer = 0."""
    rs = _make_returns([0.01] * 50)
    m = compute_metrics(rs, periods_per_year=252)
    assert m["ulcer"] == pytest.approx(0.0, abs=1e-9)
    assert m["dd_duration"] == 0


def test_dd_duration_counts_consecutive_drawdown_periods():
    """Drawdown de 3 périodes consécutives puis retour au peak."""
    # +10%, -5%, -5%, -5%, +20%  → on est sous peak pour 3 périodes (puis on dépasse)
    rs = _make_returns([0.10, -0.05, -0.05, -0.05, 0.20])
    m = compute_metrics(rs, periods_per_year=252)
    assert m["dd_duration"] == 3
