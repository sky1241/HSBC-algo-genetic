"""Walk-forward analysis orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import features_fourier, optimizer, regime_hmm, risk_sizing, stats_eval


@dataclass(slots=True)
class WalkForwardConfig:
    n_states: int = 3
    n_trials: int = 25
    feature_window: int = 128
    lfp_cutoff: float = 0.1
    min_train_size: int = 200
    min_train_years: int = 1
    start_year: int | None = None
    end_year: int | None = None
    baseline_params: dict[str, float] = field(
        default_factory=lambda: optimizer.BASELINE_PARAMS.copy()
    )
    periods_per_year: int = 252


@dataclass(slots=True)
class WFAResult:
    returns: pd.DataFrame
    metrics: pd.DataFrame
    params: pd.DataFrame
    skipped_folds: list[dict[str, object]]
    config: WalkForwardConfig


def _prepare_phases(series: pd.Series) -> pd.Series:
    mapped = series.copy()
    mapped = mapped.map(lambda x: f"phase_{int(x)}" if pd.notna(x) else np.nan)
    return mapped.astype("object")


def _complete_params(
    params_by_phase: dict[str, dict[str, float]],
    phases: Iterable[str],
    baseline: dict[str, float],
) -> dict[str, dict[str, float]]:
    completed = {phase: params.copy() for phase, params in params_by_phase.items()}
    for phase in set(phases):
        if phase is None or (isinstance(phase, float) and np.isnan(phase)):
            continue
        if phase not in completed:
            completed[phase] = baseline.copy()
    return completed


def run_walk_forward(
    df: pd.DataFrame,
    seeds: Sequence[int],
    config: WalkForwardConfig,
) -> WFAResult:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être de type DatetimeIndex")
    df = df.sort_index()
    features = features_fourier.compute_fourier_features(
        df,
        features_fourier.FourierConfig(
            window=config.feature_window,
            lfp_cutoff=config.lfp_cutoff,
            price_col="close",
        ),
    )
    years = sorted(df.index.year.unique())
    if config.start_year is not None:
        years = [y for y in years if y >= config.start_year]
    if config.end_year is not None:
        years = [y for y in years if y <= config.end_year]
    if len(years) <= config.min_train_years:
        raise ValueError("Pas assez d'années pour exécuter le walk-forward")

    all_returns: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    params_records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for seed in seeds:
        hmm_cfg = regime_hmm.HMMConfig(n_states=config.n_states)
        opt_cfg = optimizer.OptimiserConfig(n_trials=config.n_trials, seed=seed)
        for idx in range(config.min_train_years, len(years)):
            test_year = years[idx]
            train_years = years[:idx]
            train_df = df[df.index.year.isin(train_years)]
            if len(train_df) < config.min_train_size:
                skipped.append({"seed": seed, "year": test_year, "reason": "train_size"})
                continue
            test_df = df[df.index.year == test_year]
            train_features = features.loc[train_df.index]
            try:
                model = regime_hmm.fit_regime_model(train_features, hmm_cfg, random_state=seed)
            except ValueError:
                skipped.append({"seed": seed, "year": test_year, "reason": "hmm_fit"})
                continue
            train_states = regime_hmm.predict_regimes(model, train_features)
            train_phases = _prepare_phases(train_states)
            test_features = features.loc[test_df.index]
            test_states = regime_hmm.predict_regimes(model, test_features)
            test_phases = _prepare_phases(test_states)
            if test_phases.isna().all():
                skipped.append({"seed": seed, "year": test_year, "reason": "no_phases"})
                continue
            test_phases = test_phases.fillna(method="ffill").fillna(method="bfill").fillna("phase_unknown")
            params_by_phase = optimizer.optimise_phase_parameters(
                train_df,
                train_phases,
                opt_cfg,
                config.periods_per_year,
                config.baseline_params,
            )
            params_by_phase = _complete_params(
                params_by_phase,
                test_phases.unique(),
                config.baseline_params,
            )
            global_returns, per_phase_returns = risk_sizing.run_phase_strategy(
                test_df,
                test_phases,
                params_by_phase,
            )
            baseline_returns = risk_sizing.simulate_strategy(test_df, config.baseline_params)
            records = pd.DataFrame(
                {
                    "timestamp": test_df.index,
                    "return": global_returns.values,
                    "seed": seed,
                    "year": test_year,
                    "strategy": "phaseaware",
                    "phase": test_phases.values,
                }
            )
            baseline_records = pd.DataFrame(
                {
                    "timestamp": test_df.index,
                    "return": baseline_returns.values,
                    "seed": seed,
                    "year": test_year,
                    "strategy": "baseline",
                    "phase": test_phases.values,
                }
            )
            all_returns.extend([records, baseline_records])
            for phase, params in params_by_phase.items():
                params_record = {
                    "seed": seed,
                    "year": test_year,
                    "phase": phase,
                    **params,
                }
                params_records.append(params_record)
            metrics_phaseaware = stats_eval.compute_metrics(
                global_returns, config.periods_per_year
            )
            metrics_phaseaware.update(
                {
                    "strategy": "phaseaware",
                    "seed": seed,
                    "phase": "global",
                    "year": test_year,
                }
            )
            all_metrics.append(metrics_phaseaware)
            baseline_metrics = stats_eval.compute_metrics(
                baseline_returns, config.periods_per_year
            )
            baseline_metrics.update(
                {
                    "strategy": "baseline",
                    "seed": seed,
                    "phase": "global",
                    "year": test_year,
                }
            )
            all_metrics.append(baseline_metrics)
            for phase, returns in per_phase_returns.items():
                metrics = stats_eval.compute_metrics(
                    returns.loc[test_df.index], config.periods_per_year
                )
                metrics.update(
                    {
                        "strategy": "phaseaware",
                        "seed": seed,
                        "phase": phase,
                        "year": test_year,
                    }
                )
                all_metrics.append(metrics)
            for phase in test_phases.unique():
                mask = test_phases == phase
                returns = baseline_returns.loc[mask]
                metrics = stats_eval.compute_metrics(returns, config.periods_per_year)
                metrics.update(
                    {
                        "strategy": "baseline",
                        "seed": seed,
                        "phase": phase,
                        "year": test_year,
                    }
                )
                all_metrics.append(metrics)
    returns_df = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
    metrics_df = pd.DataFrame.from_records(all_metrics)
    params_df = pd.DataFrame.from_records(params_records)
    return WFAResult(
        returns=returns_df,
        metrics=metrics_df,
        params=params_df,
        skipped_folds=skipped,
        config=config,
    )


__all__ = ["WalkForwardConfig", "WFAResult", "run_walk_forward"]
