"""Parameter search utilities for the phase-aware strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from . import risk_sizing


BASELINE_PARAMS: dict[str, float] = {
    "tenkan": 9,
    "kijun": 26,
    "senkou_b": 52,
    "shift": 26,
    "atr_mult": 2.0,
}


@dataclass(slots=True)
class OptimiserConfig:
    n_trials: int = 25
    seed: int | None = None


def _sample_params(rng: np.random.Generator) -> dict[str, float]:
    tenkan = int(rng.integers(5, 13))
    kijun = int(rng.integers(max(tenkan + 5, 20), 80))
    kijun = max(kijun, tenkan + 1)
    senkou_b = int(rng.integers(max(kijun + 5, 40), 160))
    shift = int(rng.integers(20, 35))
    atr_mult = float(rng.uniform(1.0, 5.0))
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_b": senkou_b,
        "shift": shift,
        "atr_mult": atr_mult,
    }


def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return float("nan")
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std <= 1e-12:
        return float("nan")
    return float(np.sqrt(periods_per_year) * mean / std)


def optimise_phase_parameters(
    df: pd.DataFrame,
    phases: pd.Series,
    config: OptimiserConfig,
    periods_per_year: int,
    baseline: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Random-search optimisation for each phase.

    Parameters
    ----------
    df:
        Training dataframe containing OHLCV data.
    phases:
        Phase labels aligned with ``df``.
    config:
        Optimiser configuration (number of trials and random seed).
    periods_per_year:
        Used for annualising the Sharpe ratio.
    baseline:
        Parameters used when a phase does not have enough observations.
    """

    if baseline is None:
        baseline = BASELINE_PARAMS.copy()
    rng = np.random.default_rng(config.seed)
    params_by_phase: dict[str, dict[str, float]] = {}
    valid_phases = [p for p in phases.dropna().unique()]
    for phase in valid_phases:
        mask = phases == phase
        df_phase = df.loc[mask]
        if len(df_phase) < 50:
            params_by_phase[phase] = baseline.copy()
            continue
        best_score = -np.inf
        best_params = baseline.copy()
        for _ in range(int(config.n_trials)):
            candidate = _sample_params(rng)
            returns = risk_sizing.simulate_strategy(df_phase, candidate)
            score = _sharpe_ratio(returns, periods_per_year)
            if np.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_params = candidate
        params_by_phase[phase] = best_params
    if not params_by_phase:
        params_by_phase["global"] = baseline.copy()
    return params_by_phase


__all__ = ["BASELINE_PARAMS", "OptimiserConfig", "optimise_phase_parameters"]
