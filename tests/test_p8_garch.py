"""P8 — Tests TGARCH(1,1) + EGARCH(1,1) (vol asymétrique)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.garch import (
    classify_vol_regime,
    detect_leverage_effect,
    fit_egarch,
    fit_tgarch,
    forecast_egarch,
    forecast_tgarch,
)


# ---------------------------------------------------------------------------
# Helpers : génération synthétique de séries TGARCH/EGARCH
# ---------------------------------------------------------------------------


def _simulate_tgarch(n, omega, alpha, gamma, beta, seed=0):
    """Simule une série suivant TGARCH(1,1) GJR pour valider la récupération des params."""
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / max(1.0 - alpha - gamma / 2.0 - beta, 0.05)
    eps[0] = rng.standard_normal() * np.sqrt(sigma2[0])
    for t in range(1, n):
        ind = 1.0 if eps[t - 1] < 0.0 else 0.0
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + gamma * eps[t - 1] ** 2 * ind + beta * sigma2[t - 1]
        eps[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    return pd.Series(eps), sigma2


# ---------------------------------------------------------------------------
# fit_tgarch
# ---------------------------------------------------------------------------


def test_tgarch_recovers_params_on_synthetic_series():
    """Fit TGARCH sur série simulée → params estimés proches des vrais."""
    true_omega, true_alpha, true_gamma, true_beta = 1e-5, 0.05, 0.10, 0.80
    eps, _ = _simulate_tgarch(3000, true_omega, true_alpha, true_gamma, true_beta, seed=42)
    res = fit_tgarch(eps)
    assert res["converged"]
    # Params doivent être dans l'ordre de grandeur (MLE bruité avec n=3000)
    assert 0.0 < res["omega"] < 5e-5
    assert 0.0 <= res["alpha"] < 0.20
    assert 0.04 < res["gamma"] < 0.20
    assert 0.65 < res["beta"] < 0.95
    assert res["leverage_effect"] is True
    assert res["stationary"] is True


def test_tgarch_no_leverage_when_gamma_zero():
    """Si on simule TGARCH sans asymétrie (gamma=0), le fit doit estimer gamma ~0."""
    eps, _ = _simulate_tgarch(3000, 1e-5, 0.10, 0.0, 0.85, seed=7)
    res = fit_tgarch(eps)
    assert res["converged"]
    assert res["gamma"] < 0.05  # quasi-nul
    assert res["leverage_effect"] is False


def test_tgarch_handles_short_series():
    """< 50 obs → résultat empty avec converged=False."""
    eps = pd.Series(np.random.normal(0, 0.01, 30))
    res = fit_tgarch(eps)
    assert res["model"] == "TGARCH"
    assert res["converged"] is False
    assert res["n_obs"] == 0


def test_tgarch_handles_nan_dropping():
    """fit doit dropper les NaN proprement."""
    eps = pd.Series([0.01, np.nan, -0.02] * 100)
    res = fit_tgarch(eps)
    assert res["n_obs"] == 200  # NaN strip → 200 valides


# ---------------------------------------------------------------------------
# forecast_tgarch
# ---------------------------------------------------------------------------


def test_forecast_tgarch_horizon_one_uses_eps_last():
    """h=1 doit donner sigma_1 = sqrt(omega + (alpha + gamma·I) eps² + beta·sigma²)."""
    params = {
        "model": "TGARCH",
        "omega": 1e-5, "alpha": 0.05, "gamma": 0.10, "beta": 0.80,
        "sigma2_last": 1e-4, "eps_last": -0.02,
    }
    f = forecast_tgarch(params, h=1)
    expected = np.sqrt(1e-5 + 0.05 * 0.02 ** 2 + 0.10 * 0.02 ** 2 + 0.80 * 1e-4)
    assert f.shape == (1,)
    assert f[0] == pytest.approx(expected, rel=1e-9)


def test_forecast_tgarch_multistep_converges_to_unconditional():
    """h grand → sigma converge vers sqrt(omega / (1 - alpha - gamma/2 - beta))."""
    params = {
        "model": "TGARCH",
        "omega": 1e-5, "alpha": 0.05, "gamma": 0.10, "beta": 0.80,
        "sigma2_last": 1e-4, "eps_last": 0.0,
    }
    f = forecast_tgarch(params, h=200)
    persist = 0.05 + 0.10 / 2.0 + 0.80
    sigma_uncond = np.sqrt(1e-5 / (1.0 - persist))
    assert f[-1] == pytest.approx(sigma_uncond, rel=0.05)


def test_forecast_tgarch_invalid_h():
    params = {"model": "TGARCH", "omega": 1e-5, "alpha": 0.05, "gamma": 0.05,
              "beta": 0.85, "sigma2_last": 1e-4, "eps_last": 0.0}
    assert forecast_tgarch(params, h=0).size == 0
    assert forecast_tgarch({"model": "EGARCH"}, h=5).size == 0  # mauvais modèle


# ---------------------------------------------------------------------------
# fit_egarch
# ---------------------------------------------------------------------------


def test_egarch_fits_on_returns_with_negative_skew():
    """Returns avec asymétrie négative → EGARCH gamma < 0 (effet levier)."""
    rng = np.random.default_rng(11)
    n = 3000
    eps = np.zeros(n)
    log_sig2 = np.zeros(n)
    log_sig2[0] = -8.0
    omega, alpha, gamma, beta = -0.05, 0.15, -0.10, 0.97  # gamma < 0 → leverage
    E_abs_z = np.sqrt(2.0 / np.pi)
    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_sig2[t - 1]))
        z_prev = eps[t - 1] / sigma_prev if sigma_prev > 0 else 0.0
        log_sig2[t] = omega + alpha * (abs(z_prev) - E_abs_z) + gamma * z_prev + beta * log_sig2[t - 1]
        eps[t] = rng.standard_normal() * np.sqrt(np.exp(log_sig2[t]))
    res = fit_egarch(pd.Series(eps))
    assert res["converged"]
    assert res["model"] == "EGARCH"
    assert res["gamma"] < -0.02  # effet de levier détecté
    assert res["leverage_effect"] is True
    assert res["stationary"] is True


def test_egarch_handles_short_series():
    res = fit_egarch(pd.Series(np.random.normal(0, 0.01, 20)))
    assert res["model"] == "EGARCH"
    assert res["converged"] is False
    assert res["n_obs"] == 0


# ---------------------------------------------------------------------------
# forecast_egarch (Monte Carlo)
# ---------------------------------------------------------------------------


def test_forecast_egarch_horizon_one_deterministic():
    """h=1 est déterministe (z_T connu via eps_last/sigma_T)."""
    params = {
        "model": "EGARCH",
        "omega": -0.05, "alpha": 0.15, "gamma": -0.10, "beta": 0.97,
        "log_sigma2_last": -8.0, "eps_last": -0.01,
    }
    sigma_T = np.sqrt(np.exp(-8.0))
    z_T = -0.01 / sigma_T
    E_abs_z = np.sqrt(2.0 / np.pi)
    log_sig2_1 = (
        -0.05 + 0.15 * (abs(z_T) - E_abs_z) + (-0.10) * z_T + 0.97 * (-8.0)
    )
    expected = np.sqrt(np.exp(log_sig2_1))
    f = forecast_egarch(params, h=1)
    assert f.shape == (1,)
    assert f[0] == pytest.approx(expected, rel=1e-9)


def test_forecast_egarch_multistep_returns_finite_positive():
    params = {
        "model": "EGARCH",
        "omega": -0.05, "alpha": 0.15, "gamma": -0.10, "beta": 0.97,
        "log_sigma2_last": -8.0, "eps_last": 0.0,
    }
    f = forecast_egarch(params, h=20, n_sims=500, seed=3)
    assert f.shape == (20,)
    assert np.all(np.isfinite(f))
    assert np.all(f > 0)


# ---------------------------------------------------------------------------
# detect_leverage_effect
# ---------------------------------------------------------------------------


def test_detect_leverage_effect_tgarch():
    assert detect_leverage_effect({"model": "TGARCH", "gamma": 0.10}) is True
    assert detect_leverage_effect({"model": "TGARCH", "gamma": 0.001}) is False
    assert detect_leverage_effect({"model": "TGARCH", "gamma": -0.05}) is False


def test_detect_leverage_effect_egarch():
    """EGARCH : gamma < -seuil → leverage."""
    assert detect_leverage_effect({"model": "EGARCH", "gamma": -0.10}) is True
    assert detect_leverage_effect({"model": "EGARCH", "gamma": -0.001}) is False
    assert detect_leverage_effect({"model": "EGARCH", "gamma": 0.05}) is False


def test_detect_leverage_effect_invalid_model():
    assert detect_leverage_effect({"model": "ARCH", "gamma": 0.5}) is False
    assert detect_leverage_effect({}) is False


# ---------------------------------------------------------------------------
# classify_vol_regime
# ---------------------------------------------------------------------------


def test_classify_vol_regime_thresholds():
    assert classify_vol_regime(0.5, 1.0) == "calm"
    assert classify_vol_regime(1.0, 1.0) == "normal"
    assert classify_vol_regime(2.0, 1.0) == "elevated"
    assert classify_vol_regime(3.0, 1.0) == "extreme"
    # Edge cases
    assert classify_vol_regime(1.0, 0.0) == "normal"  # baseline=0 → safe default
    assert classify_vol_regime(float("nan"), 1.0) == "normal"
