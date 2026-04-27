"""Tail risk metrics: VaR, CVaR/ES, Cornish-Fisher, GPD POT — B17.

Couvre les besoins identifiés dans l'audit Layer-2 (2026-04-26):
- Historical VaR / CVaR (Expected Shortfall) — base
- Cornish-Fisher VaR — correction skew + kurtosis (Mark & Vaucher SSRN 4796363)
- Peaks Over Threshold (POT) + Generalized Pareto Distribution — extreme VaR
  pour les queues fat (BTC en particulier, papier MDPI Bayesian EVT 2025)

Références :
- Magdon-Ismail (2004), Maximum Drawdown
- Embrechts, Klüppelberg, Mikosch (1997), Modelling Extremal Events
- McNeil, Frey, Embrechts (2015), Quantitative Risk Management
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


# ============================================================
# Historical VaR / CVaR
# ============================================================

def historical_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk (VaR_α): quantile α des returns.

    Convention: returns négatifs = pertes. VaR > 0 indique la perte au seuil α.
    Ex: alpha=0.05 = VaR à 95% = pire 5% des cas.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 5:
        return float("nan")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    return float(-np.quantile(arr, alpha))


def historical_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR aka Expected Shortfall (ES_α): moyenne des returns ≤ VaR_α.

    Toujours ≥ VaR_α. Mesure la sévérité moyenne des pertes extrêmes.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 5:
        return float("nan")
    var_threshold = -historical_var(arr, alpha)  # quantile (négatif si pertes)
    tail = arr[arr <= var_threshold]
    if tail.size == 0:
        return float("nan")
    return float(-tail.mean())


# ============================================================
# Cornish-Fisher VaR (skew + kurtosis adjusted)
# ============================================================

def cornish_fisher_var(
    mean: float,
    std: float,
    skew: float,
    kurt_excess: float,
    alpha: float = 0.05,
) -> float:
    """Cornish-Fisher VaR avec correction skew + kurtosis (excess).

    Formule (4-moments):
        z_cf = z + (z²-1)·γ₃/6 + (z³-3z)·γ₄/24 - (2z³-5z)·γ₃²/36
    où γ₃ = skew, γ₄ = excess kurt = kurtosis - 3.

    VaR = -(mean + z_cf · std).
    """
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (z ** 2 - 1) * skew / 6.0
        + (z ** 3 - 3 * z) * kurt_excess / 24.0
        - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36.0
    )
    return float(-(mean + z_cf * std))


# ============================================================
# Peaks Over Threshold + Generalized Pareto
# ============================================================

def pot_excesses(losses: np.ndarray, threshold_quantile: float = 0.95) -> Tuple[float, np.ndarray]:
    """Extrait les excès au-dessus d'un seuil (POT method).

    Args:
        losses: array de pertes (positives = perte). Si tu as des returns, passe -returns.
        threshold_quantile: quantile pour le seuil u (e.g. 0.95).

    Returns:
        (threshold u, excesses array y_i = (loss_i - u) pour loss_i > u)
    """
    arr = np.asarray(losses, dtype=float)
    arr = arr[~np.isnan(arr)]
    u = float(np.quantile(arr, threshold_quantile))
    excesses = arr[arr > u] - u
    return u, excesses


def gpd_mle(excesses: np.ndarray) -> Tuple[float, float]:
    """MLE de Generalized Pareto Distribution sur les excesses.

    GPD CDF: F(y) = 1 - (1 + ξ·y/σ)^(-1/ξ)  pour ξ ≠ 0
              = 1 - exp(-y/σ)               pour ξ = 0

    Returns: (shape ξ, scale σ).

    Méthode des moments si MLE pose problème (échantillon court). MLE numérique
    sinon (scipy.optimize). Pour rester sans dépendance, on utilise method of
    moments comme estimateur initial robuste.
    """
    y = np.asarray(excesses, dtype=float)
    y = y[y > 0]
    if y.size < 5:
        return float("nan"), float("nan")
    mean_y = float(np.mean(y))
    var_y = float(np.var(y, ddof=1))
    if var_y <= 0 or mean_y <= 0:
        return float("nan"), float("nan")
    # Method of moments: ξ = 0.5·(1 - mean²/var), σ = mean·(1 + ξ)
    # Note: pour ξ = 0 (exponentielle) on aurait mean = std → mean²/var = 1 → ξ = 0.
    shape = 0.5 * (1.0 - mean_y ** 2 / var_y)
    scale = mean_y * (1.0 - shape)  # équivalent à 0.5·mean·(1 + mean²/var)
    return float(shape), float(scale)


def gpd_var(
    threshold: float,
    shape: float,
    scale: float,
    n_total: int,
    n_excess: int,
    alpha: float = 0.99,
) -> float:
    """VaR extreme via GPD: VaR_α = u + (σ/ξ)·[(N/n_u·(1-α))^(-ξ) - 1]

    Args:
        threshold: u utilisé dans POT.
        shape: ξ estimé.
        scale: σ estimé.
        n_total: taille totale d'échantillon.
        n_excess: nombre d'excesses au-dessus de u.
        alpha: confiance (e.g. 0.99 = VaR 99%).

    Returns:
        VaR (perte au seuil α).
    """
    if n_total <= 0 or n_excess <= 0 or scale <= 0:
        return float("nan")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    p_excess = n_excess / n_total
    quantile_term = (n_total / n_excess * (1.0 - alpha))
    if quantile_term <= 0:
        return float("nan")
    if abs(shape) < 1e-9:
        # ξ ≈ 0: VaR_α = u - σ·log(N/n_u·(1-α))
        return float(threshold - scale * np.log(quantile_term))
    return float(threshold + (scale / shape) * (quantile_term ** (-shape) - 1.0))


def gpd_es(
    threshold: float,
    shape: float,
    scale: float,
    var_alpha: float,
    alpha: float = 0.99,
) -> float:
    """Expected Shortfall via GPD: ES_α = (VaR_α + σ - ξ·u) / (1 - ξ).

    Valide pour ξ < 1.
    """
    if shape >= 1.0:
        return float("inf")  # ES indéfini si shape >= 1
    return float((var_alpha + scale - shape * threshold) / (1.0 - shape))


__all__ = [
    "historical_var",
    "historical_cvar",
    "cornish_fisher_var",
    "pot_excesses",
    "gpd_mle",
    "gpd_var",
    "gpd_es",
]
