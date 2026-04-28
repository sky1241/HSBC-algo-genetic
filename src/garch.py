"""P8 / R6 — TGARCH(1,1) & EGARCH(1,1) via `arch` lib (Kevin Sheppard).

Capture le "leverage effect" (asymétrie) : chocs négatifs amplifient la
volatilité plus que les chocs positifs de même magnitude. Complément à
HAR-RV (P4, symétrique).

Migration R6 (2026-04-28)
-------------------------
Première version (P8 commit 1c10962) utilisait scipy MLE custom (Nelder-Mead
sur log-likelihood gaussien). La spec demandait `arch` lib (Kevin Sheppard).
R6 ré-architecture en wrapper de `arch.arch_model` :
  - fit_tgarch / fit_egarch retournent des dicts à API stable (omega, alpha,
    gamma, beta, sigma2_last, etc.) MAIS internellement utilisent arch_model.
  - L'objet ARCHModelResult est stocké dans la clé "_arch_fit" du dict pour
    permettre forecast multi-step via arch.

Modèles
-------
TGARCH (GJR — Glosten-Jagannathan-Runkle 1993) :
    sigma²_t = omega + alpha * eps²_{t-1} + gamma * eps²_{t-1} * I(<0) + beta * sigma²_{t-1}
    Leverage : gamma > 0  (down boost vol).

EGARCH (Nelson 1991) :
    ln(sigma²_t) = omega + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1} + beta * ln(sigma²_{t-1})
    Leverage : gamma < 0  (z négatif amplifie ln sigma²).

Per-symbol assignment (R6, conformément spec) :
    BTCUSDT -> TGARCH
    ETHUSDT -> EGARCH
    SOLUSDT -> TGARCH (default conservatif)

Références
----------
Glosten, Jagannathan, Runkle (1993). JoF 48(5) 1779-1801.
Nelson (1991). Econometrica 59(2) 347-370.
Zakoian (1994). JEDC 18(5) 931-955.
Sheppard, K. (2024). `arch` lib. github.com/bashtage/arch
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    arch_model = None  # type: ignore

# Filtre les warnings DataScaleWarning de arch (returns en fraction = scale ~1e-4)
# On rescale x100 dans le fit pour rester dans la zone optimale, puis on
# revient à l'échelle originale post-fit.
_LEVERAGE_THRESHOLD = 0.05  # cohérent avec premiere version P8

SYMBOL_GARCH_MODEL: dict[str, str] = {
    "BTCUSDT": "TGARCH",
    "ETHUSDT": "EGARCH",
    "SOLUSDT": "TGARCH",
}
DEFAULT_GARCH_MODEL = "TGARCH"


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        r = returns.dropna().to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
    return r


def _empty_result(model: str) -> dict:
    return {
        "model": model,
        "omega": 0.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "beta": 0.0,
        "mu": 0.0,
        "sigma2_last": 0.0,
        "eps_last": 0.0,
        "log_likelihood": 0.0,
        "leverage_effect": False,
        "stationary": False,
        "n_obs": 0,
        "converged": False,
        "_arch_fit": None,
        "_scale": 1.0,
    }


def _fit_arch(returns, vol: str, model_label: str, gamma_leverage_sign: int) -> dict:
    """Wrapper interne : fit arch_model, extrait dict.

    Args:
        vol: "GARCH" pour TGARCH (avec o=1), "EGARCH" pour EGARCH.
        model_label: "TGARCH" ou "EGARCH" (key dans le dict retour).
        gamma_leverage_sign: +1 si leverage = gamma > seuil (TGARCH),
            -1 si leverage = gamma < -seuil (EGARCH).
    """
    if not HAS_ARCH:
        return _empty_result(model_label)
    r = _to_array(returns)
    if len(r) < 50:
        return _empty_result(model_label)

    # Rescale x100 pour rester dans zone optimale d'optimisation arch
    scale = 100.0
    r_scaled = r * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            r_scaled, mean="Constant", vol=vol, p=1, o=1, q=1, dist="Normal",
            rescale=False,
        )
        try:
            fit = model.fit(disp="off", show_warning=False)
        except Exception:
            return _empty_result(model_label)

    # Extraction des params (noms standardisés arch lib)
    p = fit.params
    mu_scaled = float(p.get("mu", 0.0))
    omega_scaled = float(p.get("omega", 0.0))
    alpha = float(p.get("alpha[1]", 0.0))
    gamma = float(p.get("gamma[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))

    # cond_vol et returns sont en échelle scaled. On revient à l'échelle originale.
    cond_vol_scaled = np.asarray(fit.conditional_volatility)
    sigma2_last_scaled = float(cond_vol_scaled[-1] ** 2) if len(cond_vol_scaled) else 0.0
    sigma2_last = sigma2_last_scaled / (scale * scale)
    eps_last = float(r[-1] - mu_scaled / scale)
    mu = mu_scaled / scale
    omega = omega_scaled / (scale * scale)

    # Stationarité TGARCH : alpha + gamma/2 + beta < 1 (Gaussien sym)
    # EGARCH : |beta| < 1 (autoregressive log-vol)
    if model_label == "TGARCH":
        stationary = bool(alpha + gamma / 2.0 + beta < 1.0)
    else:
        stationary = bool(abs(beta) < 1.0)

    # Leverage : signe du gamma vs seuil
    if gamma_leverage_sign > 0:
        leverage = bool(gamma > _LEVERAGE_THRESHOLD)
    else:
        leverage = bool(gamma < -_LEVERAGE_THRESHOLD)

    converged = bool(getattr(fit, "convergence_flag", 0) == 0)

    return {
        "model": model_label,
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "mu": mu,
        "sigma2_last": sigma2_last,
        "eps_last": eps_last,
        "log_likelihood": float(fit.loglikelihood),
        "leverage_effect": leverage,
        "stationary": stationary,
        "n_obs": int(len(r)),
        "converged": converged,
        "_arch_fit": fit,
        "_scale": scale,
    }


def fit_tgarch(returns) -> dict:
    """TGARCH(1,1) GJR via `arch_model(vol='GARCH', o=1)` (taux et signal scaled x100)."""
    return _fit_arch(returns, vol="GARCH", model_label="TGARCH", gamma_leverage_sign=+1)


def fit_egarch(returns) -> dict:
    """EGARCH(1,1) via `arch_model(vol='EGARCH', o=1)` (taux et signal scaled x100)."""
    return _fit_arch(returns, vol="EGARCH", model_label="EGARCH", gamma_leverage_sign=-1)


def _forecast_via_arch(params: dict, h: int) -> np.ndarray:
    """Multi-step forecast via arch.fit.forecast(horizon=h). Retourne sigma (pas sigma²).

    EGARCH multi-step requires `method="simulation"` (not analytic). On essaie
    l'analytique d'abord (fast pour TGARCH), fallback simulation si non
    supporté (cas EGARCH h > 1).
    """
    fit = params.get("_arch_fit")
    if fit is None or h <= 0:
        return np.array([])
    scale = float(params.get("_scale", 1.0))
    fc = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = fit.forecast(horizon=h, reindex=False)
        except (ValueError, NotImplementedError):
            # Analytic non supporté (typique EGARCH h>1) → simulation
            try:
                fc = fit.forecast(
                    horizon=h, reindex=False,
                    method="simulation", simulations=500,
                )
            except Exception:
                return np.array([])
        except Exception:
            return np.array([])
    if fc is None:
        return np.array([])
    try:
        var_scaled = np.asarray(fc.variance.iloc[-1].values, dtype=float)
    except Exception:
        return np.array([])
    sigma_scaled = np.sqrt(np.maximum(var_scaled, 0.0))
    sigma = sigma_scaled / scale
    return sigma


def forecast_tgarch(params: dict, h: int = 1) -> np.ndarray:
    """Forecast vol h-step ahead pour TGARCH. Sigma (pas sigma²)."""
    if params.get("model") != "TGARCH":
        return np.array([])
    return _forecast_via_arch(params, h)


def forecast_egarch(params: dict, h: int = 1) -> np.ndarray:
    """Forecast vol h-step ahead pour EGARCH. Sigma (pas sigma²)."""
    if params.get("model") != "EGARCH":
        return np.array([])
    return _forecast_via_arch(params, h)


# ---------------------------------------------------------------------------
# Per-symbol assignment (R6 spec)
# ---------------------------------------------------------------------------


def fit_for_symbol(symbol: str, returns) -> dict:
    """Per-symbol GARCH model selection (R6 spec).

    BTCUSDT -> TGARCH, ETHUSDT -> EGARCH, SOLUSDT -> TGARCH.
    Symbol non listé -> TGARCH default (conservatif).
    """
    sym = str(symbol).replace("/", "").upper()
    model_type = SYMBOL_GARCH_MODEL.get(sym, DEFAULT_GARCH_MODEL)
    if model_type == "EGARCH":
        return fit_egarch(returns)
    return fit_tgarch(returns)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def detect_leverage_effect(params: dict, threshold: float = _LEVERAGE_THRESHOLD) -> bool:
    """TGARCH : gamma > +threshold ; EGARCH : gamma < -threshold."""
    if not params or "model" not in params:
        return False
    g = params.get("gamma", 0.0)
    if params["model"] == "TGARCH":
        return bool(g > threshold)
    if params["model"] == "EGARCH":
        return bool(g < -threshold)
    return False


def classify_vol_regime(sigma_forecast: float, sigma_baseline: float) -> str:
    """calm / normal / elevated / extreme vs baseline (cf P8 v1)."""
    if not np.isfinite(sigma_forecast) or sigma_baseline <= 0:
        return "normal"
    ratio = sigma_forecast / sigma_baseline
    if ratio < 0.7:
        return "calm"
    if ratio < 1.5:
        return "normal"
    if ratio < 2.5:
        return "elevated"
    return "extreme"


def garch_regime_label(forecast_sigma: float, sigma_baseline: float) -> str:
    """Mappe la classif vol GARCH vers les labels HAR-RV {low,mid,high}.

    Pour comparer GARCH vs HAR (P4) : on prend le forecast sigma 1-step
    et on classifie selon ratio à baseline.
        ratio < 0.7  -> "low"   (vs HAR "low")
        0.7-1.5      -> "mid"
        > 1.5        -> "high"
    """
    if not np.isfinite(forecast_sigma) or sigma_baseline <= 0:
        return "mid"
    ratio = forecast_sigma / sigma_baseline
    if ratio < 0.7:
        return "low"
    if ratio < 1.5:
        return "mid"
    return "high"


__all__ = [
    "HAS_ARCH",
    "SYMBOL_GARCH_MODEL",
    "DEFAULT_GARCH_MODEL",
    "fit_tgarch",
    "fit_egarch",
    "fit_for_symbol",
    "forecast_tgarch",
    "forecast_egarch",
    "detect_leverage_effect",
    "classify_vol_regime",
    "garch_regime_label",
]
