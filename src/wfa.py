"""Walk-forward analysis orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import features_fourier, optimizer, regime_hmm, risk_sizing, stats_eval


@dataclass(slots=True)
class WalkForwardConfig:
    n_states: int | None = 3
    hmm_state_candidates: Sequence[int] = (2, 3, 4)
    hmm_feature_cols: Sequence[str] = ("P1_period", "LFP_ratio", "volatility")
    hmm_min_duration_frac: float = 0.05
    hmm_min_trade_frac: float = 0.05
    hmm_min_trades: float = 5.0
    hmm_min_obs: int = 15
    hmm_duration_col: str | None = None
    hmm_trade_count_col: str | None = None
    hmm_quantiles: tuple[float, float] = (0.25, 0.75)
    hmm_fourier_p1_col: str = "P1_period"
    hmm_fourier_lfp_col: str = "LFP_ratio"
    n_trials: int = 25
    welch_nperseg_grid: Sequence[int] = (128, 256, 512)
    welch_noverlap: float = 0.5
    welch_window: str = "hann"
    lfp_horizon_days: float = 5.0
    volatility_window: int = 96
    fs_per_day: float | None = None
    price_col: str = "close"
    min_train_size: int = 200
    min_train_years: int = 1
    start_year: int | None = None
    end_year: int | None = None
    baseline_params: dict[str, float] = field(
        default_factory=lambda: optimizer.BASELINE_PARAMS.copy()
    )
    periods_per_year: int = 252
    taker_fee: float = 0.0004
    max_drawdown: float = 0.5


@dataclass(slots=True)
class WFAResult:
    returns: pd.DataFrame
    metrics: pd.DataFrame
    params: pd.DataFrame
    skipped_folds: list[dict[str, object]]
    config: WalkForwardConfig


def _prepare_phases(series: pd.Series) -> pd.Series:
    def _map(value: object) -> object:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (str, bytes)):
            return str(value)
        if isinstance(value, (np.integer, int)):
            return f"phase_{int(value)}"
        if isinstance(value, (np.floating, float)) and np.isfinite(value):
            return f"phase_{int(value)}"
        return str(value)

    mapped = series.map(_map)
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
    *,
    funding: pd.Series | None = None,
) -> WFAResult:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être de type DatetimeIndex")
    df = df.sort_index()
    fs_per_day = (
        float(config.fs_per_day)
        if config.fs_per_day is not None
        else features_fourier.estimate_fs_per_day(df.index)
    )
    features_cfg = features_fourier.FourierConfig(
        price_col=config.price_col,
        fs_per_day=fs_per_day,
        window=config.welch_window,
        nperseg_grid=config.welch_nperseg_grid,
        noverlap_ratio=config.welch_noverlap,
        lfp_horizon_days=config.lfp_horizon_days,
        volatility_window=config.volatility_window,
    )
    features = features_fourier.compute_fourier_features(df, features_cfg)
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
        if config.hmm_state_candidates:
            candidate_states = tuple(
                dict.fromkeys(int(s) for s in config.hmm_state_candidates if int(s) > 0)
            )
        else:
            candidate_states = ()
        if config.n_states is not None:
            candidate_states = (int(config.n_states),)
        if not candidate_states:
            candidate_states = (2, 3, 4)
        hmm_cfg: dict[str, object] = {
            "feature_cols": list(config.hmm_feature_cols),
            "K": candidate_states,
            "seed": seed,
            "min_duration_frac": config.hmm_min_duration_frac,
            "min_trade_frac": config.hmm_min_trade_frac,
            "min_trades": config.hmm_min_trades,
            "min_obs": config.hmm_min_obs,
            "fourier_cols": {
                "P1": config.hmm_fourier_p1_col,
                "LFP": config.hmm_fourier_lfp_col,
            },
            "quantiles": config.hmm_quantiles,
            "return_train_states": True,
        }
        if config.hmm_duration_col:
            hmm_cfg["duration_col"] = config.hmm_duration_col
        if config.hmm_trade_count_col:
            hmm_cfg["trade_count_col"] = config.hmm_trade_count_col

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
            test_features = features.loc[test_df.index]
            try:
                hmm_result = regime_hmm.apply_hmm(train_features, test_features, hmm_cfg)
            except ValueError:
                skipped.append({"seed": seed, "year": test_year, "reason": "hmm_fit"})
                continue
            test_states = hmm_result["oos_states"].reindex(test_df.index)
            if "train_states" in hmm_result:
                train_states = hmm_result["train_states"].reindex(train_df.index)
            elif hmm_result.get("model") is None:
                train_states = regime_hmm.rules_fourier(train_features, train_features, hmm_cfg)
                train_states = train_states.reindex(train_df.index)
            else:
                train_states = pd.Series(np.nan, index=train_df.index, name="state")
            train_phases = _prepare_phases(train_states).reindex(train_df.index)
            test_phases = _prepare_phases(test_states).reindex(test_df.index)
            if test_phases.isna().all():
                skipped.append({"seed": seed, "year": test_year, "reason": "no_phases"})
                continue
            test_phases = test_phases.ffill().bfill().fillna("phase_unknown")
            funding_slice = None
            if funding is not None:
                funding_slice = funding.reindex(test_df.index).fillna(0.0)
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
                fee=config.taker_fee,
                funding=funding_slice,
                max_drawdown=config.max_drawdown,
            )
            baseline_returns = risk_sizing.simulate_strategy(
                test_df,
                config.baseline_params,
                fee=config.taker_fee,
                funding=funding_slice,
                max_drawdown=config.max_drawdown,
            )
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
