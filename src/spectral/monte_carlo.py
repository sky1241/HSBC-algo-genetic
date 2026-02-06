#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.5 - Monte Carlo Robust Validation

Provides robust validation via Monte Carlo simulation:
- Bootstrap resampling of trades/returns
- Percentile statistics (p5, p50, p95)
- Worst-case analysis
- Confidence intervals

Usage:
    from src.spectral.monte_carlo import MonteCarloValidator

    validator = MonteCarloValidator(n_simulations=1000)
    results = validator.run(trades_df)

    print(f"Median equity: {results.equity_p50:.2f}")
    print(f"5th percentile (worst case): {results.equity_p5:.2f}")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class MCResults:
    """Monte Carlo simulation results."""
    # Equity statistics
    equity_p5: float       # 5th percentile (worst case)
    equity_p25: float      # 25th percentile
    equity_p50: float      # Median
    equity_p75: float      # 75th percentile
    equity_p95: float      # 95th percentile

    # Drawdown statistics
    mdd_p5: float          # 5th percentile MDD (best case)
    mdd_p50: float         # Median MDD
    mdd_p95: float         # 95th percentile MDD (worst case)

    # Trade statistics
    trades_mean: float
    win_rate_p50: float

    # Risk metrics
    calmar_p5: float       # 5th percentile Calmar
    calmar_p50: float      # Median Calmar
    sharpe_p50: float      # Median Sharpe

    # Stability metrics
    variance_inter_seed: float  # Variance between different seeds
    survival_rate: float        # % of simulations that survive (equity > 0.5x)

    # Raw data for further analysis
    equity_distribution: np.ndarray
    mdd_distribution: np.ndarray

    def to_dict(self) -> Dict[str, float]:
        return {
            "equity_p5": self.equity_p5,
            "equity_p25": self.equity_p25,
            "equity_p50": self.equity_p50,
            "equity_p75": self.equity_p75,
            "equity_p95": self.equity_p95,
            "mdd_p5": self.mdd_p5,
            "mdd_p50": self.mdd_p50,
            "mdd_p95": self.mdd_p95,
            "trades_mean": self.trades_mean,
            "win_rate_p50": self.win_rate_p50,
            "calmar_p5": self.calmar_p5,
            "calmar_p50": self.calmar_p50,
            "sharpe_p50": self.sharpe_p50,
            "variance_inter_seed": self.variance_inter_seed,
            "survival_rate": self.survival_rate,
        }


def _bootstrap_equity_curve(
    returns: np.ndarray,
    n_trades: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Single bootstrap simulation.

    Returns: (final_equity, max_drawdown, calmar, sharpe)
    """
    rng = np.random.default_rng(seed)

    # Resample returns with replacement
    sampled_idx = rng.choice(len(returns), size=n_trades, replace=True)
    sampled_returns = returns[sampled_idx]

    # Build equity curve
    equity = np.cumprod(1 + sampled_returns)

    # Final equity (as multiplier)
    final_eq = float(equity[-1]) if len(equity) > 0 else 1.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    mdd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Annualized metrics (assuming H2 = 12 bars/day)
    n_bars = len(equity)
    years = max(0.1, n_bars / (12 * 365))
    cagr = (final_eq ** (1 / years)) - 1 if final_eq > 0 else -1.0

    # Calmar ratio
    calmar = cagr / max(0.01, abs(mdd))

    # Sharpe proxy (annualized)
    if len(sampled_returns) > 1:
        mean_ret = np.mean(sampled_returns)
        std_ret = np.std(sampled_returns)
        sharpe = (mean_ret / max(1e-8, std_ret)) * np.sqrt(12 * 365)  # Annualized
    else:
        sharpe = 0.0

    return final_eq, mdd, calmar, sharpe


class MonteCarloValidator:
    """Monte Carlo validation for trading strategies.

    Performs bootstrap resampling to estimate:
    - Best/worst case scenarios (percentiles)
    - Parameter stability
    - Risk metrics distribution
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        n_jobs: int = 1,
        survival_threshold: float = 0.5,
    ):
        """
        Args:
            n_simulations: Number of bootstrap simulations
            n_jobs: Parallel jobs (1 = sequential)
            survival_threshold: Equity multiplier below which = "ruin"
        """
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs
        self.survival_threshold = survival_threshold

    def run_from_returns(
        self,
        returns: np.ndarray,
        n_trades: Optional[int] = None,
        seed: int = 42,
    ) -> MCResults:
        """Run Monte Carlo from trade returns array.

        Args:
            returns: Array of trade returns (as fractions, e.g., 0.05 = 5%)
            n_trades: Number of trades per simulation (default: len(returns))
            seed: Random seed for reproducibility

        Returns:
            MCResults with all statistics
        """
        if n_trades is None:
            n_trades = len(returns)

        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return self._empty_results()

        # Run simulations
        equities: List[float] = []
        mdds: List[float] = []
        calmars: List[float] = []
        sharpes: List[float] = []

        if self.n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        _bootstrap_equity_curve, returns, n_trades, seed + i
                    ): i
                    for i in range(self.n_simulations)
                }
                for future in as_completed(futures):
                    eq, mdd, calmar, sharpe = future.result()
                    equities.append(eq)
                    mdds.append(mdd)
                    calmars.append(calmar)
                    sharpes.append(sharpe)
        else:
            # Sequential execution
            for i in range(self.n_simulations):
                eq, mdd, calmar, sharpe = _bootstrap_equity_curve(
                    returns, n_trades, seed + i
                )
                equities.append(eq)
                mdds.append(mdd)
                calmars.append(calmar)
                sharpes.append(sharpe)

        # Convert to arrays
        equities = np.array(equities)
        mdds = np.array(mdds)
        calmars = np.array(calmars)
        sharpes = np.array(sharpes)

        # Calculate win rate
        win_rate = float(np.mean(returns > 0))

        # Survival rate
        survival = float(np.mean(equities >= self.survival_threshold))

        return MCResults(
            equity_p5=float(np.percentile(equities, 5)),
            equity_p25=float(np.percentile(equities, 25)),
            equity_p50=float(np.percentile(equities, 50)),
            equity_p75=float(np.percentile(equities, 75)),
            equity_p95=float(np.percentile(equities, 95)),
            mdd_p5=float(np.percentile(mdds, 5)),
            mdd_p50=float(np.percentile(mdds, 50)),
            mdd_p95=float(np.percentile(mdds, 95)),
            trades_mean=float(n_trades),
            win_rate_p50=win_rate,
            calmar_p5=float(np.percentile(calmars, 5)),
            calmar_p50=float(np.percentile(calmars, 50)),
            sharpe_p50=float(np.percentile(sharpes, 50)),
            variance_inter_seed=float(np.var(equities)),
            survival_rate=survival,
            equity_distribution=equities,
            mdd_distribution=mdds,
        )

    def run_from_trades(
        self,
        trades_df: pd.DataFrame,
        return_col: str = "pnl_pct",
        seed: int = 42,
    ) -> MCResults:
        """Run Monte Carlo from trades DataFrame.

        Args:
            trades_df: DataFrame with trade results
            return_col: Column name for returns (as %)
            seed: Random seed

        Returns:
            MCResults
        """
        if return_col not in trades_df.columns:
            # Try alternatives
            for alt in ["return", "pnl", "profit", "gain"]:
                if alt in trades_df.columns:
                    return_col = alt
                    break

        returns = trades_df[return_col].values / 100.0  # Convert % to fraction
        return self.run_from_returns(returns, seed=seed)

    def _empty_results(self) -> MCResults:
        """Return empty results for insufficient data."""
        empty = np.array([1.0])
        return MCResults(
            equity_p5=1.0, equity_p25=1.0, equity_p50=1.0, equity_p75=1.0, equity_p95=1.0,
            mdd_p5=0.0, mdd_p50=0.0, mdd_p95=0.0,
            trades_mean=0, win_rate_p50=0.5,
            calmar_p5=0.0, calmar_p50=0.0, sharpe_p50=0.0,
            variance_inter_seed=0.0, survival_rate=1.0,
            equity_distribution=empty, mdd_distribution=empty,
        )


def aggregate_seed_results(
    seed_results: List[Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate results from multiple seeds.

    Args:
        seed_results: List of result dicts from different seeds
        metric_keys: Which metrics to aggregate (default: all numeric)

    Returns:
        Dict with aggregated statistics per metric
    """
    if not seed_results:
        return {}

    if metric_keys is None:
        # Auto-detect numeric keys
        metric_keys = [
            k for k, v in seed_results[0].items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        ]

    result = {}
    for key in metric_keys:
        values = [r.get(key, np.nan) for r in seed_results]
        values = [v for v in values if np.isfinite(v)]
        if values:
            result[key] = {
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0,
            }

    return result


def worst_case_decision(
    mc_results: MCResults,
    min_equity_p5: float = 0.8,
    max_mdd_p95: float = 0.30,
    min_calmar_p5: float = 0.3,
    min_survival: float = 0.95,
) -> Tuple[bool, List[str]]:
    """Make accept/reject decision based on worst-case metrics.

    Args:
        mc_results: Monte Carlo results
        min_equity_p5: Minimum 5th percentile equity (e.g., 0.8 = 80% of initial)
        max_mdd_p95: Maximum 95th percentile MDD (e.g., 0.30 = 30%)
        min_calmar_p5: Minimum 5th percentile Calmar ratio
        min_survival: Minimum survival rate (e.g., 0.95 = 95%)

    Returns:
        (passed: bool, violations: list of string descriptions)
    """
    violations = []

    if mc_results.equity_p5 < min_equity_p5:
        violations.append(
            f"equity_p5={mc_results.equity_p5:.2f} < {min_equity_p5}"
        )

    if mc_results.mdd_p95 > max_mdd_p95:
        violations.append(
            f"mdd_p95={mc_results.mdd_p95:.2%} > {max_mdd_p95:.2%}"
        )

    if mc_results.calmar_p5 < min_calmar_p5:
        violations.append(
            f"calmar_p5={mc_results.calmar_p5:.2f} < {min_calmar_p5}"
        )

    if mc_results.survival_rate < min_survival:
        violations.append(
            f"survival={mc_results.survival_rate:.2%} < {min_survival:.2%}"
        )

    passed = len(violations) == 0
    return passed, violations
