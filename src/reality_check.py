"""White's Reality Check + Hansen Superior Predictive Ability (SPA) test.

Tests de significativité statistique de l'edge d'une (ou un panel de) stratégie(s)
de trading, après correction pour data snooping.

Références:
- White (2000) — A Reality Check for Data Snooping
  https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf
- Hansen (2005) — A Test for Superior Predictive Ability
  J. Business & Economic Statistics 23(4): 365–380
- Politis & Romano (1994) — The Stationary Bootstrap, JASA 89: 1303–1313

Usage typique:
    >>> # K stratégies vs benchmark, n returns chacun
    >>> import numpy as np
    >>> from src.reality_check import whites_reality_check, hansen_spa_test
    >>> # excess_returns shape (n, K) — chaque colonne = returns_k - returns_benchmark
    >>> p = whites_reality_check(excess_returns, n_bootstrap=2000, block_size_mean=20)
    >>> # p < 0.05 ⇒ on rejette H0 "best strategy n'a pas d'edge"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class RealityCheckResult:
    """Résultat de White's Reality Check."""
    test_statistic: float       # V = max_k √n * f̄_k
    p_value: float              # P(V* >= V | H0)
    best_strategy: int          # arg max_k f̄_k
    n_strategies: int
    n_bootstrap: int


@dataclass(slots=True)
class SPAResult:
    """Résultat du Hansen SPA test."""
    test_statistic: float       # V = max_k √n * f̄_k
    p_value_consistent: float   # SPA_c (centered)
    p_value_lower: float        # SPA_l (lower bound, plus conservateur)
    p_value_upper: float        # SPA_u (upper bound, moins conservateur)
    best_strategy: int
    n_strategies: int
    n_bootstrap: int


def stationary_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    block_size_mean: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Génère n_bootstrap séquences d'indices via le stationary bootstrap (Politis-Romano).

    Args:
        n: longueur de la série originale.
        n_bootstrap: nombre de répliques.
        block_size_mean: longueur de bloc moyenne (E[L] = 1/p, p = 1/block_size_mean).
        rng: numpy Generator pour reproductibilité.

    Returns:
        Array (n_bootstrap, n) d'indices à valeurs dans [0, n-1].
    """
    if rng is None:
        rng = np.random.default_rng()
    if n <= 0:
        raise ValueError("n must be > 0")
    if block_size_mean <= 1.0:
        raise ValueError("block_size_mean must be > 1")
    p = 1.0 / block_size_mean
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    for b in range(n_bootstrap):
        # Premier index aléatoire
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        # Pour chaque step : avec proba p on saute à un nouveau random index;
        # sinon on continue le bloc en avançant de 1 (modulo n pour stationarité).
        rand_uniform = rng.random(n - 1)
        rand_starts = rng.integers(0, n, size=n - 1)
        for t in range(1, n):
            if rand_uniform[t - 1] < p:
                idx[t] = rand_starts[t - 1]
            else:
                idx[t] = (idx[t - 1] + 1) % n
        out[b] = idx
    return out


def whites_reality_check(
    excess_returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size_mean: float = 10.0,
    seed: Optional[int] = None,
) -> RealityCheckResult:
    """White's Reality Check (White 2000).

    Test si la MEILLEURE stratégie d'un panel a une performance vraiment > benchmark
    après correction pour le fait qu'on a testé K stratégies.

    H0: max_k E[f_k] <= 0  (aucune stratégie n'a d'edge)
    H1: max_k E[f_k] > 0   (au moins une bat le benchmark)

    Args:
        excess_returns: array (n, K) — chaque colonne k = returns_k - returns_benchmark
            sur les mêmes n périodes. (n returns, K stratégies à comparer.)
        n_bootstrap: nombre de répliques bootstrap (recommandé >= 1000).
        block_size_mean: longueur de bloc moyenne pour stationary bootstrap.
            Règle de pouce: ~ n^(1/3). Pour n=1000 → ~10.
        seed: seed RNG pour reproductibilité.

    Returns:
        RealityCheckResult — p_value < 0.05 ⇒ rejet H0 (au moins une stratégie a un edge réel).
    """
    if excess_returns.ndim == 1:
        excess_returns = excess_returns.reshape(-1, 1)
    n, K = excess_returns.shape
    if n < 5 or K == 0:
        raise ValueError("Need at least 5 obs and 1 strategy")

    rng = np.random.default_rng(seed)

    # f̄_k pour chaque stratégie (mean excess return)
    f_bar = excess_returns.mean(axis=0)
    # Test statistic V = √n * max_k f̄_k
    sqrt_n = np.sqrt(n)
    V = sqrt_n * np.max(f_bar)
    best_k = int(np.argmax(f_bar))

    # Bootstrap: générer V*_b et comparer à V
    indices = stationary_bootstrap_indices(n, n_bootstrap, block_size_mean, rng)
    V_star = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = excess_returns[indices[b]]
        f_bar_star = sample.mean(axis=0)
        # Re-centrage sous H0: f̄*_k - f̄_k pour que la distribution sous H0 soit centrée
        V_star[b] = sqrt_n * np.max(f_bar_star - f_bar)

    p_value = float(np.mean(V_star >= V))
    return RealityCheckResult(
        test_statistic=float(V),
        p_value=p_value,
        best_strategy=best_k,
        n_strategies=K,
        n_bootstrap=n_bootstrap,
    )


def hansen_spa_test(
    excess_returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size_mean: float = 10.0,
    seed: Optional[int] = None,
) -> SPAResult:
    """Hansen's Superior Predictive Ability test (Hansen 2005).

    Plus puissant que White's RC car élimine les modèles clairement inférieurs
    (poor performers) du calcul de la distribution sous H0.

    Calcule trois p-values:
    - SPA_l: lower (suppose tous les modèles sont sur la frontière) — conservateur.
    - SPA_c: consistent (centered) — le standard recommandé par Hansen.
    - SPA_u: upper (ignore poor performers) — moins conservateur.

    Args:
        excess_returns: array (n, K). Cf whites_reality_check.

    Returns:
        SPAResult.
    """
    if excess_returns.ndim == 1:
        excess_returns = excess_returns.reshape(-1, 1)
    n, K = excess_returns.shape
    if n < 5 or K == 0:
        raise ValueError("Need at least 5 obs and 1 strategy")
    rng = np.random.default_rng(seed)

    sqrt_n = np.sqrt(n)
    f_bar = excess_returns.mean(axis=0)
    sigma_k = excess_returns.std(axis=0, ddof=0)
    sigma_k = np.where(sigma_k > 0, sigma_k, 1e-12)
    V = sqrt_n * np.max(f_bar)
    best_k = int(np.argmax(f_bar))

    # Threshold A_n = -sigma_k * sqrt(2 log log n) / sqrt(n)
    # Modèles avec f̄_k < A_n sont considérés "poor performers" (exclus de SPA_c et SPA_u).
    # ln(ln(n)) demande n >= 3 pour être > 0.
    log_log_n = np.log(np.log(max(n, 4)))
    A_n = -sigma_k * np.sqrt(2.0 * log_log_n) / sqrt_n  # array shape (K,)

    # Mu hat sous H0 par version du test:
    mu_l = np.zeros(K)                                    # lower
    mu_c = np.minimum(f_bar, 0.0)                         # consistent (centered)
    mu_u = np.where(f_bar >= A_n, np.minimum(f_bar, 0.0), f_bar)  # upper

    indices = stationary_bootstrap_indices(n, n_bootstrap, block_size_mean, rng)

    V_star_l = np.empty(n_bootstrap)
    V_star_c = np.empty(n_bootstrap)
    V_star_u = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = excess_returns[indices[b]]
        f_bar_star = sample.mean(axis=0)
        # Re-centré: (f̄*_k - f̄_k) + mu^_k_version, puis on garde max_k
        V_star_l[b] = sqrt_n * np.max(f_bar_star - f_bar + mu_l)
        V_star_c[b] = sqrt_n * np.max(f_bar_star - f_bar + mu_c)
        V_star_u[b] = sqrt_n * np.max(f_bar_star - f_bar + mu_u)

    p_l = float(np.mean(V_star_l >= V))
    p_c = float(np.mean(V_star_c >= V))
    p_u = float(np.mean(V_star_u >= V))

    return SPAResult(
        test_statistic=float(V),
        p_value_lower=p_l,
        p_value_consistent=p_c,
        p_value_upper=p_u,
        best_strategy=best_k,
        n_strategies=K,
        n_bootstrap=n_bootstrap,
    )


__all__ = [
    "RealityCheckResult",
    "SPAResult",
    "stationary_bootstrap_indices",
    "whites_reality_check",
    "hansen_spa_test",
]
