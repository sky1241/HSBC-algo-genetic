"""P12 — Tests MSM Calvet-Fisher 2004 (multifractal vol forecasting)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.msm_calvet import (
    MSMFitResult,
    classify_macro_regime,
    fit_msm,
    forecast_msm_volatility,
    simulate_msm,
)


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------


def test_simulate_msm_returns_finite_positive_vol():
    """simulate_msm produit returns finis et sigma_t > 0 partout."""
    returns, sigma_t = simulate_msm(
        n=500, m0=1.4, sigma_bar=0.02, gamma_k=0.5, k_bar=4, seed=0,
    )
    assert returns.shape == (500,)
    assert sigma_t.shape == (500,)
    assert np.all(np.isfinite(returns))
    assert np.all(sigma_t > 0)


def test_simulate_msm_higher_m0_yields_fatter_tails():
    """m0 plus haut → vol plus extrême → kurtosis plus élevée."""
    _, sigma_low = simulate_msm(
        n=2000, m0=1.1, sigma_bar=0.02, gamma_k=0.3, k_bar=4, seed=1,
    )
    _, sigma_high = simulate_msm(
        n=2000, m0=1.7, sigma_bar=0.02, gamma_k=0.3, k_bar=4, seed=1,
    )
    # m0 plus haut → ratio sigma_max/sigma_min plus grand
    ratio_low = sigma_low.max() / sigma_low.min()
    ratio_high = sigma_high.max() / sigma_high.min()
    assert ratio_high > ratio_low


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def test_msm_handles_short_series():
    """< 50 obs → résultat empty."""
    r = pd.Series(np.random.normal(0, 0.01, 30))
    fit = fit_msm(r, k_bar=4)
    assert fit.converged is False
    assert fit.n_obs == 0


def test_msm_fit_returns_valid_params_on_synthetic_series():
    """Fit sur série simulée → params dans bornes valides."""
    returns, _ = simulate_msm(
        n=800, m0=1.4, sigma_bar=0.02, gamma_k=0.4, k_bar=4, seed=42,
    )
    fit = fit_msm(returns, k_bar=4)
    assert fit.converged is True or fit.log_likelihood != 0.0
    # Bornes spec
    assert 1.0 < fit.m0 < 2.0
    assert fit.sigma_bar > 0
    assert 0.0 < fit.gamma_k < 1.0
    assert fit.n_obs == 800
    # pi_filtered = distribution sur 2^k_bar
    assert fit.pi_filtered.shape == (16,)
    assert fit.pi_filtered.sum() == pytest.approx(1.0, rel=1e-6)


def test_msm_fit_handles_nan_dropping():
    r = pd.Series([0.01, np.nan, -0.005] * 100)
    fit = fit_msm(r, k_bar=3)
    assert fit.n_obs == 200  # NaN strip


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def test_forecast_horizon_one_returns_positive_finite():
    """Forecast h=1 → vol > 0 finie."""
    returns, _ = simulate_msm(
        n=500, m0=1.4, sigma_bar=0.02, gamma_k=0.4, k_bar=4, seed=3,
    )
    fit = fit_msm(returns, k_bar=4)
    f = forecast_msm_volatility(fit, horizon=1)
    assert f.shape == (1,)
    assert np.isfinite(f[0])
    assert f[0] > 0


def test_forecast_invalid_horizon_returns_empty():
    fit = fit_msm(pd.Series(np.random.normal(0, 0.01, 100)), k_bar=3)
    assert forecast_msm_volatility(fit, horizon=0).size == 0
    assert forecast_msm_volatility(fit, horizon=-5).size == 0


def test_forecast_long_horizon_converges_to_unconditional():
    """h très grand → sigma_forecast → sigma_bar (régime stationnaire)."""
    returns, _ = simulate_msm(
        n=500, m0=1.4, sigma_bar=0.02, gamma_k=0.5, k_bar=4, seed=7,
    )
    fit = fit_msm(returns, k_bar=4)
    f = forecast_msm_volatility(fit, horizon=200)
    # Dernière valeur : doit être proche de sigma_bar (E[∏M]=1 → sigma=sigma_bar)
    assert f[-1] == pytest.approx(fit.sigma_bar, rel=0.15)


def test_forecast_empty_fit_returns_empty():
    """Fit empty (n_obs=0) → forecast vide."""
    empty = fit_msm(pd.Series([0.001] * 10), k_bar=3)  # < 50 → empty
    assert forecast_msm_volatility(empty, horizon=10).size == 0


# ---------------------------------------------------------------------------
# classify_macro_regime
# ---------------------------------------------------------------------------


def test_classify_macro_regime_thresholds():
    sigma_bar = 0.02
    assert classify_macro_regime(0.5 * sigma_bar, sigma_bar) == "calm"
    assert classify_macro_regime(1.0 * sigma_bar, sigma_bar) == "normal"
    assert classify_macro_regime(1.5 * sigma_bar, sigma_bar) == "stressed"
    assert classify_macro_regime(3.0 * sigma_bar, sigma_bar) == "crisis"


def test_classify_macro_regime_edge_cases():
    """sigma_bar=0 ou NaN → "normal" (safe default)."""
    assert classify_macro_regime(0.05, 0.0) == "normal"
    assert classify_macro_regime(float("nan"), 0.02) == "normal"
