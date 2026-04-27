"""Tests pour tail_risk (VaR, CVaR, Cornish-Fisher, GPD POT) — B17."""
from __future__ import annotations

import numpy as np
import pytest

from src.tail_risk import (
    cornish_fisher_var,
    gpd_es,
    gpd_mle,
    gpd_var,
    historical_cvar,
    historical_var,
    pot_excesses,
)


# ============================================================
# Historical VaR / CVaR
# ============================================================

def test_historical_var_normal():
    """VaR 95% d'une distribution N(0, 1) ≈ 1.645."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 1, 10000)
    v = historical_var(rets, alpha=0.05)
    assert 1.5 < v < 1.8, f"VaR(N(0,1)) ≈ 1.645, got {v}"


def test_historical_cvar_greater_than_var():
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 1, 10000)
    v = historical_var(rets, alpha=0.05)
    cv = historical_cvar(rets, alpha=0.05)
    assert cv > v, f"ES doit être > VaR, got cv={cv}, v={v}"


def test_var_cvar_handle_short_series():
    assert np.isnan(historical_var(np.array([1.0]), 0.05))
    assert np.isnan(historical_cvar(np.array([1.0]), 0.05))


def test_var_invalid_alpha_raises():
    with pytest.raises(ValueError):
        historical_var(np.zeros(100), alpha=0)
    with pytest.raises(ValueError):
        historical_var(np.zeros(100), alpha=1)


# ============================================================
# Cornish-Fisher
# ============================================================

def test_cornish_fisher_equals_normal_when_no_skew_no_kurt():
    """Avec skew=0 et excess_kurt=0, CF VaR = VaR Gaussien."""
    cf = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, alpha=0.05)
    # VaR Gaussien à 95% = -z_0.05 ≈ 1.645
    assert cf == pytest.approx(1.6449, abs=0.01)


def test_cornish_fisher_higher_with_negative_skew():
    """Skew négatif (queue gauche fat) ⇒ VaR plus élevé qu'avec skew=0."""
    cf_normal = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, alpha=0.05)
    cf_neg_skew = cornish_fisher_var(0.0, 1.0, -1.0, 0.0, alpha=0.05)
    assert cf_neg_skew > cf_normal


def test_cornish_fisher_higher_with_positive_excess_kurt_at_extreme_quantile():
    """Kurtosis excess positive ⇒ VaR plus élevé aux quantiles extrêmes (α≤0.01).

    Note: à α=0.05, l'effet de kurt sur la queue gauche est ambigu (la formule
    de l'expansion d'Edgeworth a un coefficient (z³-3z)/24 dont le signe change).
    L'effet "fat tails ⇒ VaR plus extrême" se manifeste à α=0.01 et plus extrême.
    """
    cf_normal = cornish_fisher_var(0.0, 1.0, 0.0, 0.0, alpha=0.01)
    cf_fat = cornish_fisher_var(0.0, 1.0, 0.0, 3.0, alpha=0.01)
    assert cf_fat > cf_normal


# ============================================================
# POT + GPD
# ============================================================

def test_pot_excesses_returns_threshold_and_excess():
    losses = np.linspace(0, 10, 1000)
    u, exc = pot_excesses(losses, threshold_quantile=0.95)
    assert u == pytest.approx(9.5, abs=0.01)
    # ~5% de losses au-dessus
    assert exc.size > 30
    # Tous positifs
    assert (exc > 0).all()


def test_gpd_mle_on_simulated_pareto():
    """MLE sur des excesses générés depuis une GPD avec shape connue."""
    rng = np.random.default_rng(7)
    # Si Y ~ Exp(1), X = sigma * (e^Y - 1) / shape ~ GPD(shape, sigma) avec ξ > 0
    true_shape = 0.3
    true_scale = 1.0
    n = 5000
    u_uni = rng.uniform(0, 1, n)
    excesses = true_scale / true_shape * ((1 - u_uni) ** (-true_shape) - 1)
    shape_hat, scale_hat = gpd_mle(excesses)
    # Tolérance large car method of moments est moins précis que MLE pur
    assert 0.1 < shape_hat < 0.5, f"shape_hat={shape_hat}"
    assert 0.6 < scale_hat < 1.5, f"scale_hat={scale_hat}"


def test_gpd_mle_handles_short_sample():
    s, sc = gpd_mle(np.array([1.0]))
    assert np.isnan(s) and np.isnan(sc)


def test_gpd_var_recovers_threshold_at_threshold_quantile():
    """À α = 1 - p_excess, VaR ≈ threshold (par construction)."""
    n_total = 10000
    n_excess = 500  # 5%
    p_excess = n_excess / n_total
    threshold = 2.0
    var_at_thresh = gpd_var(threshold, 0.2, 1.0, n_total, n_excess, alpha=1 - p_excess)
    assert var_at_thresh == pytest.approx(threshold, abs=0.01)


def test_gpd_var_increases_with_alpha():
    var_99 = gpd_var(2.0, 0.2, 1.0, 10000, 500, alpha=0.99)
    var_995 = gpd_var(2.0, 0.2, 1.0, 10000, 500, alpha=0.995)
    var_999 = gpd_var(2.0, 0.2, 1.0, 10000, 500, alpha=0.999)
    assert var_99 < var_995 < var_999


def test_gpd_var_zero_shape_uses_exponential():
    """ξ = 0 ⇒ VaR = u - σ·log(...). Doit donner valeur finie raisonnable."""
    var_99 = gpd_var(2.0, 0.0, 1.0, 10000, 500, alpha=0.99)
    assert np.isfinite(var_99)
    assert var_99 > 2.0


def test_gpd_es_greater_than_var():
    var_99 = gpd_var(2.0, 0.2, 1.0, 10000, 500, alpha=0.99)
    es_99 = gpd_es(2.0, 0.2, 1.0, var_99, alpha=0.99)
    assert es_99 > var_99


def test_gpd_es_inf_when_shape_above_one():
    """ξ ≥ 1 ⇒ ES indéfini (variance infinie)."""
    es = gpd_es(2.0, 1.5, 1.0, 5.0)
    assert es == float("inf") or np.isinf(es)


# ============================================================
# Integration: pipeline complet sur returns
# ============================================================

def test_full_pipeline_on_synthetic_returns():
    """Pipeline complet: returns → VaR/CVaR historique → CF → POT/GPD."""
    rng = np.random.default_rng(42)
    # Returns mixture: 95% N(0, 0.01), 5% N(-0.05, 0.02) → fat left tail
    n = 5000
    is_tail = rng.random(n) < 0.05
    rets = np.where(is_tail, rng.normal(-0.05, 0.02, n), rng.normal(0, 0.01, n))

    v_hist = historical_var(rets, 0.05)
    cv_hist = historical_cvar(rets, 0.05)
    cf = cornish_fisher_var(rets.mean(), rets.std(), 0.0, 5.0, 0.05)

    losses = -rets
    u, exc = pot_excesses(losses, 0.95)
    shape, scale = gpd_mle(exc)
    var_99 = gpd_var(u, shape, scale, n, exc.size, 0.99)

    # Tous finis et croissants pour des seuils de plus en plus extrêmes
    assert all(np.isfinite(x) for x in [v_hist, cv_hist, cf, var_99])
    assert v_hist > 0
    assert cv_hist >= v_hist
    assert var_99 > 0
