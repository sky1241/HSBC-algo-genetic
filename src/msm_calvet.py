"""P12 — Markov-Switching Multifractal (Calvet-Fisher 2004).

Forecasting vol long-terme (semaine+) pour confirmer régimes macro.
Complémentaire à HAR-RV (P4, court-moyen terme) et EGARCH/TGARCH (P8, asymétrie
quotidienne).

Modèle (Calvet-Fisher 2004)
---------------------------
sigma²_t = sigma_bar² × Π_{i=1}^{k_bar} M_{i,t}

avec M_{i,t} ∈ {m_0, 2 - m_0}  (Bernoulli symétrique, E[M]=1)
Probabilité de switch indép : gamma_i = 1 - (1 - gamma_k)^(b^(i - k_bar))

Paramètres MLE
- m_0       ∈ (1, 2)        amplitude du switch (2 = très volatile)
- sigma_bar > 0              vol unconditional (annualisée racine ou bps)
- gamma_k   ∈ (0, 1)         switch prob à la fréquence la plus rapide
- b         ≥ 1.5            facteur de spacing (default 2)

Forecasting
-----------
Forward filter : distribution P(état | F_t) sur 2^k_bar états.
Forecast h-step : M^h × P(état_t) → E[sigma²_{t+h} | F_t] pondéré.

Pour h grand → convergence vers sigma_bar² (régime stationnaire).
Use case : annonce "régime macro stressed" si sigma forecast(7d) >
1.5 × sigma_bar.

Limitations pragmatiques
- k_bar ≤ 8 (état space 2^8 = 256, MLE tractable)
- Innovations Gaussiennes (Calvet original supporte t-Student aussi)
- Pas d'optim Optuna (params figés post-MLE)

Référence
---------
Calvet, L. E., & Fisher, A. J. (2004). "How to Forecast Long-Run
Volatility: Regime Switching and the Estimation of Multifractal Processes."
Journal of Financial Econometrics, 2(1), 49-83.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


_DEFAULT_K_BAR = 6  # 2^6 = 64 états (tractable)
_DEFAULT_B = 2.0
_DEFAULT_INIT = {"m0": 1.4, "sigma_bar": 0.02, "gamma_k": 0.5}


def _switch_probs(gamma_k: float, b: float, k_bar: int) -> np.ndarray:
    """gamma_i = 1 - (1 - gamma_k)^(b^(i - k_bar))  pour i ∈ [1, k_bar].

    i=k_bar → gamma_i = gamma_k (most frequent)
    i=1     → gamma_i = 1 - (1 - gamma_k)^(b^(1-k_bar))  (most persistent)
    """
    gamma_k = float(np.clip(gamma_k, 1e-6, 1.0 - 1e-6))
    out = np.empty(k_bar, dtype=float)
    for idx in range(k_bar):
        i = idx + 1  # 1..k_bar
        exponent = b ** (i - k_bar)
        out[idx] = 1.0 - (1.0 - gamma_k) ** exponent
    return out


def _state_grid(k_bar: int, m0: float) -> np.ndarray:
    """Énumère les 2^k_bar états et le multiplicateur ∏M_i associé.

    State s ∈ [0, 2^k_bar) :  bit i = 1 → M_i = m0,  bit i = 0 → M_i = 2 - m0.

    Returns:
        np.ndarray shape (2^k_bar,) des produits ∏M_i pour chaque état.
    """
    n_states = 2 ** k_bar
    multipliers = np.empty(n_states, dtype=float)
    high = float(m0)
    low = float(2.0 - m0)
    for s in range(n_states):
        prod = 1.0
        for i in range(k_bar):
            if (s >> i) & 1:
                prod *= high
            else:
                prod *= low
        multipliers[s] = prod
    return multipliers


def _transition_matrix(switch_probs: np.ndarray, k_bar: int) -> np.ndarray:
    """Matrice de transition 2^k_bar × 2^k_bar.

    Composantes indépendantes : P(s'_i = s_i) = 1 - gamma_i + gamma_i/2,
    P(s'_i ≠ s_i) = gamma_i/2.  (Bernoulli symétrique on switch.)
    """
    n_states = 2 ** k_bar
    # Per-component proba de rester dans son état OU switcher vers le même
    # côté = 1 - gamma + gamma/2 = 1 - gamma/2.  Switch côté opposé = gamma/2.
    p_stay = 1.0 - switch_probs / 2.0
    p_switch = switch_probs / 2.0

    M = np.empty((n_states, n_states), dtype=float)
    for s in range(n_states):
        for s_prime in range(n_states):
            p = 1.0
            for i in range(k_bar):
                bit_old = (s >> i) & 1
                bit_new = (s_prime >> i) & 1
                p *= p_stay[i] if bit_old == bit_new else p_switch[i]
            M[s, s_prime] = p
    return M


def _stationary_distribution(k_bar: int) -> np.ndarray:
    """Distribution stationnaire = uniforme 1/2^k_bar (chaque comp Bernoulli sym)."""
    n_states = 2 ** k_bar
    return np.full(n_states, 1.0 / n_states)


# ---------------------------------------------------------------------------
# Forward filter (likelihood)
# ---------------------------------------------------------------------------


def _msm_neg_log_likelihood(
    theta: np.ndarray,
    returns: np.ndarray,
    k_bar: int,
    b: float,
) -> float:
    m0, sigma_bar, gamma_k = theta
    if not (1.0 < m0 < 2.0):
        return 1e12
    if sigma_bar <= 0:
        return 1e12
    if not (1e-6 < gamma_k < 1.0 - 1e-6):
        return 1e12

    sw = _switch_probs(gamma_k, b, k_bar)
    M = _transition_matrix(sw, k_bar)
    grid = _state_grid(k_bar, m0)

    # Sigma_t² conditionnelle à l'état : sigma_bar² × grid
    sigma2_states = sigma_bar ** 2 * grid

    # Forward filter
    pi = _stationary_distribution(k_bar)
    log_lik = 0.0
    for r in returns:
        # Likelihood per state : Gaussien(0, sigma_state)
        lik = stats.norm.pdf(r, loc=0.0, scale=np.sqrt(sigma2_states))
        # On intègre P(état_t | F_{t-1}) × likelihood
        weighted = pi * lik
        marg = float(weighted.sum())
        if marg <= 0 or not np.isfinite(marg):
            return 1e12
        log_lik += float(np.log(marg))
        # Update : P(état_t | F_t) ∝ weighted
        post = weighted / marg
        # Predict : P(état_{t+1} | F_t) = post @ M
        pi = post @ M
        if not np.all(np.isfinite(pi)) or pi.sum() <= 0:
            return 1e12
    return -log_lik


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@dataclass
class MSMFitResult:
    m0: float
    sigma_bar: float
    gamma_k: float
    b: float
    k_bar: int
    log_likelihood: float
    converged: bool
    n_obs: int
    pi_filtered: np.ndarray  # distribution finale P(état_T | F_T)


def fit_msm(
    returns,
    k_bar: int = _DEFAULT_K_BAR,
    b: float = _DEFAULT_B,
    seed: int = 0,
) -> MSMFitResult:
    """Estime MSM par MLE (Nelder-Mead, paramètres bornés).

    Args:
        returns: pd.Series ou np.array de log-returns.
        k_bar: nombre de composantes multifractales (default 6 → 64 états).
        b: spacing (default 2 = binaire).

    Returns:
        MSMFitResult avec params + pi_filtered (utile pour forecast).
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna().to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
    if len(r) < 50:
        return _empty_fit(k_bar, b)

    # Init data-aware
    sigma_init = float(np.std(r, ddof=1))
    init = np.array([
        _DEFAULT_INIT["m0"],
        max(sigma_init, 1e-4),
        _DEFAULT_INIT["gamma_k"],
    ])

    res = optimize.minimize(
        _msm_neg_log_likelihood,
        init,
        args=(r, k_bar, b),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
    )
    m0, sigma_bar, gamma_k = res.x
    # Re-run forward filter pour pi_filtered final
    sw = _switch_probs(gamma_k, b, k_bar)
    M = _transition_matrix(sw, k_bar)
    grid = _state_grid(k_bar, m0)
    sigma2_states = sigma_bar ** 2 * grid
    pi = _stationary_distribution(k_bar)
    for ri in r:
        lik = stats.norm.pdf(ri, loc=0.0, scale=np.sqrt(sigma2_states))
        weighted = pi * lik
        marg = float(weighted.sum())
        if marg <= 0:
            break
        post = weighted / marg
        pi = post @ M

    return MSMFitResult(
        m0=float(m0),
        sigma_bar=float(sigma_bar),
        gamma_k=float(gamma_k),
        b=float(b),
        k_bar=int(k_bar),
        log_likelihood=float(-res.fun),
        converged=bool(res.success),
        n_obs=int(len(r)),
        pi_filtered=pi,
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def forecast_msm_volatility(
    fit: MSMFitResult,
    horizon: int,
) -> np.ndarray:
    """Forecast E[sigma_{t+h} | F_t] pour h ∈ [1, horizon].

    E[sigma²_{t+h}] = sigma_bar² × Σ_s P(s_{t+h}) × ∏ M(s)
    où P(s_{t+h}) = pi_filtered @ M^h.
    """
    if horizon <= 0 or fit.n_obs == 0:
        return np.array([])
    sw = _switch_probs(fit.gamma_k, fit.b, fit.k_bar)
    M = _transition_matrix(sw, fit.k_bar)
    grid = _state_grid(fit.k_bar, fit.m0)

    out = np.empty(horizon, dtype=float)
    pi_h = fit.pi_filtered.copy()
    for h in range(horizon):
        pi_h = pi_h @ M
        sigma2 = float(fit.sigma_bar ** 2 * (pi_h * grid).sum())
        out[h] = float(np.sqrt(max(sigma2, 0.0)))
    return out


def classify_macro_regime(
    sigma_forecast: float,
    sigma_bar: float,
) -> str:
    """Classifie le régime macro vs sigma unconditional.

    Returns:
        "calm"      : forecast < 0.7 × sigma_bar
        "normal"    : 0.7 - 1.3 × sigma_bar
        "stressed"  : 1.3 - 2.0 × sigma_bar
        "crisis"    : > 2.0 × sigma_bar
    """
    if not np.isfinite(sigma_forecast) or sigma_bar <= 0:
        return "normal"
    ratio = sigma_forecast / sigma_bar
    if ratio < 0.7:
        return "calm"
    if ratio < 1.3:
        return "normal"
    if ratio < 2.0:
        return "stressed"
    return "crisis"


# ---------------------------------------------------------------------------
# Simulation (utile pour tests recovery)
# ---------------------------------------------------------------------------


def simulate_msm(
    n: int,
    m0: float,
    sigma_bar: float,
    gamma_k: float,
    k_bar: int = _DEFAULT_K_BAR,
    b: float = _DEFAULT_B,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simule n returns et leur sigma_t (volatilité instantanée) du MSM.

    Returns:
        (returns, sigma_t).
    """
    rng = np.random.default_rng(seed)
    sw = _switch_probs(gamma_k, b, k_bar)
    grid = _state_grid(k_bar, m0)

    # Initial state : tirage uniforme
    s = rng.integers(0, 2 ** k_bar)
    returns = np.empty(n, dtype=float)
    sigma_t = np.empty(n, dtype=float)
    for t in range(n):
        sigma_t[t] = sigma_bar * np.sqrt(grid[s])
        returns[t] = rng.standard_normal() * sigma_t[t]
        # Update : pour chaque composante i, switch avec proba gamma_i/2
        for i in range(k_bar):
            if rng.random() < sw[i] / 2.0:
                # Switch côté opposé
                s ^= (1 << i)
    return returns, sigma_t


# ---------------------------------------------------------------------------


def _empty_fit(k_bar: int, b: float) -> MSMFitResult:
    return MSMFitResult(
        m0=1.0, sigma_bar=0.0, gamma_k=0.0,
        b=b, k_bar=k_bar,
        log_likelihood=0.0, converged=False, n_obs=0,
        pi_filtered=_stationary_distribution(k_bar),
    )


__all__ = [
    "MSMFitResult",
    "fit_msm",
    "forecast_msm_volatility",
    "classify_macro_regime",
    "simulate_msm",
]
