"""CSCV / PBO — Combinatorially Symmetric Cross-Validation & Probability of Backtest Overfitting.

Implementation of Bailey, Borwein, Lopez de Prado & Zhu (2013):
"The Probability of Backtest Overfitting"

Usage:
  from optimizers.cscv_pbo import compute_pbo

  # trials_matrix: N strategies × T time periods (returns or Sharpe per period)
  pbo, logits, details = compute_pbo(trials_matrix, n_splits=8)
  print(f"PBO = {pbo:.2%}")
  # PBO > 5% = warning overfitting
  # PBO > 25% = high risk
  # PBO > 50% = likely overfit

Reference:
  Bailey, D.H., Borwein, J.M., Lopez de Prado, M. & Zhu, Q.J. (2013).
  "The Probability of Backtest Overfitting."
  Notices of the AMS (ScholarWorks WMU/42).
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _sharpe_from_returns(returns: np.ndarray) -> float:
    """Compute Sharpe ratio from a returns array (annualized, 252 periods)."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


def compute_pbo(
    trials_matrix: np.ndarray,
    n_splits: int = 8,
    metric: str = "sharpe",
    max_combos: int = 5000,
    seed: int = 42,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Compute Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    trials_matrix : ndarray, shape (N, T)
        N strategies (parameter sets) × T time periods.
        Each row = returns series of one strategy across T periods.
    n_splits : int
        Number of even splits (must be even). Typical: 8-16.
        More splits = more combinations = better estimate but slower.
    metric : str
        Performance metric: "sharpe" (default) or "mean" (mean return).
    max_combos : int
        Cap on number of combinations to evaluate (for large S, C(S,S/2) can be huge).
        If exceeded, randomly sample max_combos combinations.
    seed : int
        Random seed for reproducible subsampling of combinations.

    Returns
    -------
    pbo : float
        Probability of Backtest Overfitting [0, 1].
        PBO > 0.05 = warning. PBO > 0.25 = high risk. PBO > 0.50 = likely overfit.
    logits : ndarray
        Logit of the rank for each combination. Mean logit < 0 → overfitting.
    details : dict
        Additional diagnostics: n_combos, rank_distribution, degradation stats.
    """
    N, T = trials_matrix.shape
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if N < 2:
        raise ValueError(f"Need at least 2 strategies, got {N}")
    if T < n_splits:
        raise ValueError(f"T={T} too small for {n_splits} splits (need T >= n_splits)")

    # Step 1: Split time axis into S equal chunks
    split_size = T // n_splits
    # Trim if T not exactly divisible
    T_trimmed = split_size * n_splits
    matrix = trials_matrix[:, :T_trimmed]

    splits = []
    for s in range(n_splits):
        splits.append(matrix[:, s * split_size : (s + 1) * split_size])
    # splits[s] has shape (N, split_size)

    # Step 2: Enumerate C(S, S/2) combinations
    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))
    total_combos = len(all_combos)

    rng = np.random.default_rng(seed)
    if total_combos > max_combos:
        logger.info("CSCV: %d combinations > max %d, subsampling.", total_combos, max_combos)
        indices = rng.choice(total_combos, size=max_combos, replace=False)
        combos = [all_combos[i] for i in sorted(indices)]
    else:
        combos = all_combos

    # Step 3: For each combination, compute IS/OOS performance
    under_median = 0
    ranks = []
    logit_values = []
    degradations = []

    metric_fn = _sharpe_from_returns if metric == "sharpe" else lambda r: float(np.mean(r))

    for combo in combos:
        oos_indices = sorted(set(range(n_splits)) - set(combo))

        # IS returns: concatenate selected splits
        is_returns = np.concatenate([splits[s] for s in combo], axis=1)  # (N, half*split_size)
        oos_returns = np.concatenate([splits[s] for s in oos_indices], axis=1)

        # Compute metric for each strategy
        is_scores = np.array([metric_fn(is_returns[i]) for i in range(N)])
        oos_scores = np.array([metric_fn(oos_returns[i]) for i in range(N)])

        # Best IS strategy
        best_is_idx = np.argmax(is_scores)
        best_is_score = is_scores[best_is_idx]
        best_oos_score = oos_scores[best_is_idx]

        # Rank: what fraction of OOS strategies does the IS-best beat?
        rank = float(np.mean(oos_scores <= best_oos_score))  # percentile rank
        ranks.append(rank)

        # Logit of rank (avoid log(0) and log(inf))
        rank_clipped = np.clip(rank, 0.01, 0.99)
        logit = float(np.log(rank_clipped / (1.0 - rank_clipped)))
        logit_values.append(logit)

        # Performance degradation: IS best score vs its OOS score
        if abs(best_is_score) > 1e-12:
            degradation = (best_is_score - best_oos_score) / abs(best_is_score)
        else:
            degradation = 0.0
        degradations.append(degradation)

        # Under median?
        if rank <= 0.5:
            under_median += 1

    # Step 4: Compute PBO
    pbo = float(under_median / len(combos))
    logits = np.array(logit_values)

    details = {
        "pbo": pbo,
        "n_strategies": N,
        "n_periods": T,
        "n_splits": n_splits,
        "n_combos_evaluated": len(combos),
        "n_combos_total": total_combos,
        "mean_logit": float(np.mean(logits)),
        "mean_rank": float(np.mean(ranks)),
        "mean_degradation": float(np.mean(degradations)),
        "std_degradation": float(np.std(degradations)),
        "interpretation": (
            "LOW RISK" if pbo < 0.05 else
            "MODERATE RISK" if pbo < 0.25 else
            "HIGH RISK" if pbo < 0.50 else
            "LIKELY OVERFIT"
        ),
    }

    logger.info(
        "CSCV/PBO: PBO=%.2f%% (%s), %d combos, mean_rank=%.2f, mean_degradation=%.1f%%",
        pbo * 100, details["interpretation"], len(combos),
        details["mean_rank"], details["mean_degradation"] * 100,
    )

    return pbo, logits, details


def build_trials_matrix_from_wfa(
    archive_entries: list,
    df: Any,
    backtest_fn: Any,
    n_periods: int = 52,
    **backtest_kw: Any,
) -> np.ndarray:
    """Build a trials matrix from ACO archive for CSCV analysis.

    Takes the top-K solutions from the ACO archive, backtests each on the full data,
    and splits the returns into n_periods (e.g. 52 weeks).

    Parameters
    ----------
    archive_entries : list
        List of dicts with 'params' key (decoded Ichimoku params).
    df : DataFrame
        Full OHLCV data.
    backtest_fn : callable
        backtest_long_short function.
    n_periods : int
        Number of time periods to split returns into (e.g. 52 for weekly).

    Returns
    -------
    ndarray, shape (N, n_periods)
        N strategies × n_periods. Each cell = cumulative return in that period.
    """
    from .fitness import run_backtest

    N = len(archive_entries)
    T = len(df)
    period_size = T // n_periods

    # Get per-bar returns for each strategy via simple simulation
    # (using risk_sizing.simulate_strategy for speed)
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.risk_sizing import simulate_strategy

    matrix = np.zeros((N, n_periods))
    for i, entry in enumerate(archive_entries):
        params = entry if isinstance(entry, dict) else entry.get("params", entry)
        returns = simulate_strategy(df, params)
        # Split into n_periods chunks and sum returns per chunk
        returns_arr = returns.values[:period_size * n_periods]
        for p in range(n_periods):
            chunk = returns_arr[p * period_size : (p + 1) * period_size]
            matrix[i, p] = float(np.sum(chunk))

    return matrix
