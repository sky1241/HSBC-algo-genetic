"""P5 — Probabilistic Sharpe Ratio (PSR) live vs backtest.

Détecte rapidement (~14j vs ~6 mois pour un Sharpe statique) si l'edge live
diverge du backtest. Probabilité que le vrai Sharpe (population) dépasse
un benchmark `sr_benchmark`.

Référence (formule Eq. 9):
    Bailey, D. H., & López de Prado, M. (2012).
    "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44. SSRN 1821643.

Formule canonique:
    PSR(SR*) = Φ( (SR_hat - SR*) × sqrt(n - 1)
                  / sqrt(1 - γ3 × SR_hat + ((γ4 - 1)/4) × SR_hat^2) )

    où:
        Φ        = scipy.stats.norm.cdf
        SR_hat   = Sharpe observé (NON annualisé pour cohérence avec n)
        SR*      = Sharpe benchmark à dépasser
        n        = nombre de returns
        γ3       = skewness des returns
        γ4       = kurtosis NON-EXCESS (= 3 pour normale)

Seuils opérationnels (par défaut, override possible):
    PSR >= 0.5  : indéterminé / cohérent backtest
    PSR <  0.5  : edge live probablement INFÉRIEUR au backtest
    PSR <  0.3  : alpha decay sérieux → investigation
    PSR <  0.1  : 30j rolling → arrêt système
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _sample_stats(returns: pd.Series) -> tuple[float, float, float, int]:
    """Retourne (SR_hat, skew, kurt_non_excess, n) sur returns dropna."""
    arr = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0 or not np.isfinite(sigma):
        return float("nan"), float("nan"), float("nan"), n
    sr_hat = mu / sigma
    g3 = float(stats.skew(arr, bias=True))
    g4 = float(stats.kurtosis(arr, fisher=False, bias=True))  # non-excess
    return sr_hat, g3, g4, n


def compute_psr(returns_live, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey-LdP 2012, Eq. 9).

    Args:
        returns_live: pd.Series ou array-like des returns live (NON annualisés).
            Doit contenir au moins 2 observations valides.
        sr_benchmark: Sharpe benchmark à dépasser (NON annualisé pour cohérence).
            0.0 = "edge non nul". Pour comparer à un backtest annualisé,
            convertir d'abord (ex: SR_bt_annual / sqrt(periods_per_year)).

    Returns:
        P(SR_true > sr_benchmark) ∈ [0, 1]. 0.5 si returns dégénéré ou
        sample insuffisant. Asymptote 0/1 si la stat est extrême.
    """
    sr_hat, g3, g4, n = _sample_stats(returns_live)
    if not all(np.isfinite([sr_hat, g3, g4])) or n < 2:
        return 0.5
    var_factor = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * (sr_hat ** 2)
    if var_factor <= 0 or not np.isfinite(var_factor):
        # Cas pathologique : on retourne 0.5 (indéterminé) plutôt que NaN
        return 0.5
    z = (sr_hat - float(sr_benchmark)) * math.sqrt(n - 1) / math.sqrt(var_factor)
    return float(stats.norm.cdf(z))


def min_track_record_length(
    sr_hat: float,
    sr_benchmark: float = 0.0,
    g3: float = 0.0,
    g4: float = 3.0,
    alpha: float = 0.05,
) -> int:
    """Minimum returns nécessaires pour confirmer SR > benchmark avec confiance 1-α.

    Bailey-LdP 2012, Eq. 12 (réécrite):
        MinTRL = 1 + (1 - γ3 × SR_hat + ((γ4-1)/4) × SR_hat^2) × (z_(1-α) / (SR_hat - SR*))^2

    Args:
        sr_hat: Sharpe observé (NON annualisé).
        sr_benchmark: Sharpe à dépasser.
        g3: skewness (default 0 = symétrique).
        g4: kurtosis non-excess (default 3 = normale).
        alpha: niveau test (default 5%).

    Returns:
        Nombre entier de returns minimum nécessaires. Renvoie un grand
        nombre (10**9) si SR_hat <= sr_benchmark (cas dégénéré inférable).
    """
    if sr_hat <= sr_benchmark:
        return 10 ** 9
    z = float(stats.norm.ppf(1.0 - alpha))
    var_factor = 1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * (sr_hat ** 2)
    if var_factor <= 0 or not np.isfinite(var_factor):
        return 10 ** 9
    diff = sr_hat - sr_benchmark
    return int(1 + var_factor * (z / diff) ** 2)


def classify_psr_alert(psr: float) -> str:
    """Classifie la PSR selon les seuils opérationnels P5.

    Returns:
        "ok"           : PSR >= 0.5 (cohérent backtest)
        "warn"         : 0.3 <= PSR < 0.5 (edge < backtest)
        "alpha_decay"  : 0.1 <= PSR < 0.3 (sérieux)
        "kill_system"  : PSR < 0.1 (arrêt audit complet)
    """
    if not np.isfinite(psr):
        return "ok"  # safe : NaN -> indéterminé, pas d'alerte
    if psr >= 0.5:
        return "ok"
    if psr >= 0.3:
        return "warn"
    if psr >= 0.1:
        return "alpha_decay"
    return "kill_system"


__all__ = [
    "compute_psr",
    "min_track_record_length",
    "classify_psr_alert",
]
