"""Hidden Markov Model helpers with phasenaware constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(slots=True)
class HMMConfig:
    n_states: int | None = None
    state_candidates: Sequence[int] = (2, 3, 4)
    covariance_type: str = "full"
    n_iter: int = 200
    min_state_fraction: float = 0.05
    min_total_length: int = 30
    feature_columns: Sequence[str] = ("P1_period", "LFP_ratio", "volatility")


def _prepare_matrix(features: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, pd.Index]:
    matrix = features.loc[:, list(columns)].dropna()
    return matrix.values, matrix.index


def _check_state_constraints(
    states: np.ndarray,
    n_components: int,
    min_fraction: float,
    min_total_length: int,
) -> bool:
    if states.size == 0:
        return False
    unique, counts = np.unique(states, return_counts=True)
    if len(unique) != n_components:
        return False
    total = states.size
    min_obs = max(int(np.ceil(total * min_fraction)), int(min_total_length / n_components))
    if np.any(counts < max(min_obs, 1)):
        return False
    return True


def fit_regime_model(
    features: pd.DataFrame,
    config: HMMConfig | None = None,
    random_state: int | None = None,
) -> GaussianHMM:
    """Fit a Gaussian HMM on the provided feature matrix."""

    cfg = config or HMMConfig()
    columns = list(cfg.feature_columns)
    X, _ = _prepare_matrix(features, columns)
    if len(X) < 10:
        raise ValueError("Pas assez d'observations pour entraîner le HMM")
    candidates = (
        [int(cfg.n_states)] if cfg.n_states is not None else [int(c) for c in cfg.state_candidates]
    )
    candidates = [c for c in candidates if 2 <= c <= 6]
    if not candidates:
        raise ValueError("Aucun nombre d'états valide fourni")
    best_model: GaussianHMM | None = None
    best_score = -np.inf
    for n_states in candidates:
        if len(X) <= n_states:
            continue
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                random_state=random_state,
            )
            model.fit(X)
            states = model.predict(X)
        except ValueError:
            continue
        if not _check_state_constraints(states, n_states, cfg.min_state_fraction, cfg.min_total_length):
            continue
        score = model.score(X) - 0.5 * n_states * np.log(len(X))
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise ValueError("Impossible de trouver un HMM respectant les contraintes")
    setattr(best_model, "_feature_columns", tuple(columns))
    return best_model


def predict_regimes(
    model: GaussianHMM,
    features: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.Series:
    """Predict the hidden state sequence for ``features`` using ``model``."""

    if columns is None:
        columns = getattr(model, "_feature_columns", ("P1_period", "LFP_ratio", "volatility"))
    X, idx = _prepare_matrix(features, columns)
    if len(X) == 0:
        return pd.Series(dtype=float, index=features.index)
    states = model.predict(X)
    series = pd.Series(np.nan, index=features.index, dtype=float)
    series.loc[idx] = states
    return series


__all__ = ["HMMConfig", "fit_regime_model", "predict_regimes"]
