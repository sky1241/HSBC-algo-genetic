"""P8 — EGARCH(1,1) & TGARCH(1,1) : volatilité asymétrique.

Capture le "leverage effect" (effet d'asymétrie) : les chocs négatifs
amplifient la volatilité plus que les chocs positifs de même magnitude.
Complément naturel à HAR-RV (P4) qui est symétrique.

Modèles
-------
TGARCH(1,1) (Glosten-Jagannathan-Runkle 1993, Zakoian 1994):
    sigma²_t = omega + alpha * eps²_{t-1}
                     + gamma * eps²_{t-1} * I(eps_{t-1} < 0)
                     + beta  * sigma²_{t-1}
    Leverage effect : gamma > 0  (down-moves boost vol).
    Stationarité    : alpha + gamma/2 + beta < 1 (Gaussian innovations).

EGARCH(1,1) (Nelson 1991):
    ln(sigma²_t) = omega + alpha * (|z_{t-1}| - E|z|)
                         + gamma * z_{t-1}
                         + beta  * ln(sigma²_{t-1})
    avec z_t = eps_t / sigma_t  (résidu standardisé).
    Pour Gaussien   : E|z| = sqrt(2/pi) ~= 0.7979.
    Leverage effect : gamma < 0  (négatif amplifie sigma).
    Stationarité    : |beta| < 1.

API
---
    fit_tgarch(returns)              -> dict params + log_likelihood
    fit_egarch(returns)              -> dict params + log_likelihood
    forecast_tgarch(params, h)       -> array vols horizon h
    forecast_egarch(params, h)       -> array vols horizon h (Monte Carlo)
    detect_leverage_effect(params)   -> bool (test asymétrie)
    classify_vol_regime(sigma_now, sigma_baseline) -> label

Hyperparamètres figés (anti-snooping) :
    Optimisation MLE Gaussien
    Initial guess : alpha=0.05, beta=0.85, gamma=0.05
    Bornes : omega>1e-9, alpha,beta,gamma in [0,1] (TGARCH) ou (-1,1) (EGARCH gamma)
    Pas d'optim Optuna : on garde les estimées MLE par série.

Références
----------
Glosten, Jagannathan, Runkle (1993). "On the relation between the expected
    value and the volatility of the nominal excess return on stocks."
    Journal of Finance, 48(5), 1779-1801.
Nelson (1991). "Conditional heteroskedasticity in asset returns: A new
    approach." Econometrica, 59(2), 347-370.
Zakoian (1994). "Threshold heteroskedastic models." Journal of Economic
    Dynamics and Control, 18(5), 931-955.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


_E_ABS_Z_GAUSSIAN = float(np.sqrt(2.0 / np.pi))  # ~= 0.7979
_DEFAULT_INIT = {"omega": 1e-6, "alpha": 0.05, "beta": 0.85, "gamma": 0.05}
_LEVERAGE_THRESHOLD = 0.05  # gamma significatif au-dessus du noise floor MLE
# (γ=0 vrai → MLE ~0.04 par contrainte de positivité ; γ=0.10 → MLE ~0.14
# sur n=3000. Seuil 0.05 sépare proprement noise vs effet réel.)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        r = returns.dropna().to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
    return r


def _demean(r: np.ndarray) -> tuple[np.ndarray, float]:
    mu = float(np.mean(r))
    return r - mu, mu


# ---------------------------------------------------------------------------
# TGARCH(1,1) — GJR
# ---------------------------------------------------------------------------


def _tgarch_recursion(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    """Calcule sigma²_t pour t=0..T-1. sigma²_0 = var(eps)."""
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    var0 = float(np.mean(eps ** 2))
    sigma2[0] = max(var0, 1e-12)
    for t in range(1, n):
        e_prev = eps[t - 1]
        indicator = 1.0 if e_prev < 0.0 else 0.0
        sigma2[t] = (
            omega
            + alpha * e_prev * e_prev
            + gamma * e_prev * e_prev * indicator
            + beta * sigma2[t - 1]
        )
        if sigma2[t] < 1e-12:
            sigma2[t] = 1e-12
    return sigma2


def _tgarch_neg_log_likelihood(theta: np.ndarray, eps: np.ndarray) -> float:
    omega, alpha, gamma, beta = theta
    if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
        return 1e12
    if alpha + gamma / 2.0 + beta >= 0.9999:  # stationarité stricte
        return 1e12
    sigma2 = _tgarch_recursion(eps, omega, alpha, gamma, beta)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e12
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + (eps ** 2) / sigma2)
    return -float(ll)


def fit_tgarch(returns) -> dict:
    """Estime TGARCH(1,1) par MLE Gaussien.

    Returns:
        dict {
            "omega", "alpha", "gamma", "beta", "mu",
            "sigma2_last": float,        # sigma²_T (fit)
            "log_likelihood": float,
            "leverage_effect": bool,     # gamma > seuil
            "stationary": bool,
            "n_obs": int,
            "model": "TGARCH",
        }
    """
    r = _to_array(returns)
    if len(r) < 50:
        return _empty_result("TGARCH")
    eps, mu = _demean(r)
    var_r = float(np.var(eps, ddof=1))
    init = np.array([
        var_r * (1.0 - _DEFAULT_INIT["alpha"] - _DEFAULT_INIT["gamma"] / 2.0 - _DEFAULT_INIT["beta"]),
        _DEFAULT_INIT["alpha"],
        _DEFAULT_INIT["gamma"],
        _DEFAULT_INIT["beta"],
    ])
    init[0] = max(init[0], 1e-8)
    bounds = [(1e-10, None), (0.0, 0.999), (0.0, 0.999), (0.0, 0.999)]

    res = optimize.minimize(
        _tgarch_neg_log_likelihood,
        init,
        args=(eps,),
        method="L-BFGS-B",
        bounds=bounds,
    )
    omega, alpha, gamma, beta = res.x
    sigma2 = _tgarch_recursion(eps, omega, alpha, gamma, beta)
    return {
        "model": "TGARCH",
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "mu": float(mu),
        "sigma2_last": float(sigma2[-1]),
        "eps_last": float(eps[-1]),
        "log_likelihood": float(-res.fun),
        "leverage_effect": bool(gamma > _LEVERAGE_THRESHOLD),
        "stationary": bool(alpha + gamma / 2.0 + beta < 1.0),
        "n_obs": int(len(eps)),
        "converged": bool(res.success),
    }


def forecast_tgarch(params: dict, h: int = 1) -> np.ndarray:
    """Forecast vol h-step ahead. E[I(eps<0)] = 0.5 (Gaussien centré).

    Returns:
        np.array de longueur h avec sigma_t (pas sigma²).
    """
    if h <= 0 or params.get("model") != "TGARCH":
        return np.array([])
    omega = params["omega"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    beta = params["beta"]
    sigma2_last = params["sigma2_last"]
    eps_last = params["eps_last"]

    out = np.empty(h, dtype=float)
    # 1-step : on connait eps_last (réalisé)
    indicator = 1.0 if eps_last < 0.0 else 0.0
    sig2_1 = omega + alpha * eps_last ** 2 + gamma * eps_last ** 2 * indicator + beta * sigma2_last
    out[0] = float(np.sqrt(max(sig2_1, 1e-12)))

    # >=2 step : E[eps²_{t-1}] = sigma²_{t-1}, E[I·eps²] = sigma²/2 (Gaussien)
    sig2_prev = sig2_1
    persist = alpha + gamma / 2.0 + beta
    for k in range(1, h):
        sig2_k = omega + persist * sig2_prev
        out[k] = float(np.sqrt(max(sig2_k, 1e-12)))
        sig2_prev = sig2_k
    return out


# ---------------------------------------------------------------------------
# EGARCH(1,1) — Nelson 1991
# ---------------------------------------------------------------------------


def _egarch_recursion(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> np.ndarray:
    n = len(eps)
    log_sigma2 = np.empty(n, dtype=float)
    var0 = float(np.mean(eps ** 2))
    log_sigma2[0] = float(np.log(max(var0, 1e-12)))
    for t in range(1, n):
        sigma_prev = float(np.sqrt(np.exp(log_sigma2[t - 1])))
        if sigma_prev <= 0:
            sigma_prev = 1e-6
        z_prev = eps[t - 1] / sigma_prev
        log_sigma2[t] = (
            omega
            + alpha * (abs(z_prev) - _E_ABS_Z_GAUSSIAN)
            + gamma * z_prev
            + beta * log_sigma2[t - 1]
        )
        # Borne pour éviter overflow numérique
        if log_sigma2[t] > 50:
            log_sigma2[t] = 50.0
        elif log_sigma2[t] < -50:
            log_sigma2[t] = -50.0
    return log_sigma2


def _egarch_neg_log_likelihood(theta: np.ndarray, eps: np.ndarray) -> float:
    omega, alpha, gamma, beta = theta
    if abs(beta) >= 0.9999:
        return 1e12
    log_sigma2 = _egarch_recursion(eps, omega, alpha, gamma, beta)
    if not np.all(np.isfinite(log_sigma2)):
        return 1e12
    sigma2 = np.exp(log_sigma2)
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + (eps ** 2) / sigma2)
    return -float(ll)


def fit_egarch(returns) -> dict:
    """Estime EGARCH(1,1) par MLE Gaussien.

    Returns:
        dict structure similaire à fit_tgarch,
        avec "log_sigma2_last" et "leverage_effect" = (gamma < -seuil).
    """
    r = _to_array(returns)
    if len(r) < 50:
        return _empty_result("EGARCH")
    eps, mu = _demean(r)
    # Init data-aware : omega = (1-beta) * log(var) → stationarité initiale OK
    var_r = max(float(np.var(eps, ddof=1)), 1e-10)
    beta0 = 0.95
    init = np.array([(1.0 - beta0) * float(np.log(var_r)), 0.1, -0.05, beta0])
    # Nelder-Mead : robuste sur surface EGARCH non lisse
    res = optimize.minimize(
        _egarch_neg_log_likelihood,
        init,
        args=(eps,),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )
    omega, alpha, gamma, beta = res.x
    # Borne dure post-fit pour éviter |beta| >= 1
    if abs(beta) >= 0.9999:
        beta = 0.9999 * np.sign(beta) if beta != 0 else 0.95
    log_sigma2 = _egarch_recursion(eps, omega, alpha, gamma, beta)
    return {
        "model": "EGARCH",
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        "mu": float(mu),
        "log_sigma2_last": float(log_sigma2[-1]),
        "sigma2_last": float(np.exp(log_sigma2[-1])),
        "eps_last": float(eps[-1]),
        "log_likelihood": float(-res.fun),
        "leverage_effect": bool(gamma < -_LEVERAGE_THRESHOLD),
        "stationary": bool(abs(beta) < 1.0),
        "n_obs": int(len(eps)),
        "converged": bool(res.success),
    }


def forecast_egarch(params: dict, h: int = 1, n_sims: int = 2000, seed: int = 0) -> np.ndarray:
    """Forecast vol EGARCH par Monte Carlo (E[exp(log sig²)] non analytique simple).

    Returns:
        np.array de longueur h avec sigma_t médian sur n_sims trajectoires.
    """
    if h <= 0 or params.get("model") != "EGARCH":
        return np.array([])
    omega = params["omega"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    beta = params["beta"]
    log_sig2_T = params["log_sigma2_last"]
    eps_last = params["eps_last"]
    sigma_T = float(np.sqrt(np.exp(log_sig2_T)))
    z_T = eps_last / sigma_T if sigma_T > 0 else 0.0

    # 1-step : déterministe (z_T connu)
    log_sig2_1 = (
        omega
        + alpha * (abs(z_T) - _E_ABS_Z_GAUSSIAN)
        + gamma * z_T
        + beta * log_sig2_T
    )
    out = np.empty(h, dtype=float)
    out[0] = float(np.sqrt(np.exp(log_sig2_1)))
    if h == 1:
        return out

    # >=2 step : Monte Carlo
    rng = np.random.default_rng(seed)
    log_sig2 = np.full(n_sims, log_sig2_1)
    for k in range(1, h):
        z = rng.standard_normal(n_sims)
        log_sig2 = (
            omega
            + alpha * (np.abs(z) - _E_ABS_Z_GAUSSIAN)
            + gamma * z
            + beta * log_sig2
        )
        out[k] = float(np.median(np.sqrt(np.exp(log_sig2))))
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def detect_leverage_effect(params: dict, threshold: float = _LEVERAGE_THRESHOLD) -> bool:
    """Test si l'effet de levier est significatif.

    TGARCH : gamma > +threshold (down-moves amplifient vol).
    EGARCH : gamma < -threshold (z négatif amplifie ln sigma²).
    """
    if not params or "model" not in params:
        return False
    g = params.get("gamma", 0.0)
    if params["model"] == "TGARCH":
        return bool(g > threshold)
    if params["model"] == "EGARCH":
        return bool(g < -threshold)
    return False


def classify_vol_regime(sigma_forecast: float, sigma_baseline: float) -> str:
    """Classifie le régime de volatilité prévue vs baseline.

    Returns:
        "calm"      : forecast < 0.7 * baseline
        "normal"    : 0.7-1.5 * baseline
        "elevated"  : 1.5-2.5 * baseline
        "extreme"   : > 2.5 * baseline
    """
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


# ---------------------------------------------------------------------------


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
    }


__all__ = [
    "fit_tgarch",
    "forecast_tgarch",
    "fit_egarch",
    "forecast_egarch",
    "detect_leverage_effect",
    "classify_vol_regime",
]
