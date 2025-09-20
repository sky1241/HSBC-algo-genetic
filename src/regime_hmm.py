"""Hidden Markov Model helpers for phase/regime identification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(slots=True)
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 200


def _prepare_matrix(features: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, pd.Index]:
    matrix = features.loc[:, list(columns)].dropna()
    return matrix.values, matrix.index


def fit_regime_model(
    features: pd.DataFrame,
    config: HMMConfig | None = None,
    random_state: int | None = None,
) -> GaussianHMM:
    """Fit a Gaussian HMM on the provided feature matrix."""

    if config is None:
        config = HMMConfig()
    columns = ["dominant_period", "lfp_ratio", "volatility"]
    X, _ = _prepare_matrix(features, columns)
    if len(X) < config.n_states:
        raise ValueError("Pas assez d'observations pour entraîner le HMM")
    model = GaussianHMM(
        n_components=int(config.n_states),
        covariance_type=config.covariance_type,
        n_iter=int(config.n_iter),
        random_state=random_state,
    )
    model.fit(X)
    return model


def predict_regimes(
    model: GaussianHMM,
    features: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.Series:
    """Predict the hidden state sequence for ``features`` using ``model``."""

    if columns is None:
        columns = ("dominant_period", "lfp_ratio", "volatility")
    X, idx = _prepare_matrix(features, columns)
    if len(X) == 0:
        return pd.Series(dtype=float, index=features.index)
    states = model.predict(X)
    series = pd.Series(np.nan, index=features.index, dtype=float)
    series.loc[idx] = states
    return series


__all__ = ["HMMConfig", "fit_regime_model", "predict_regimes"]
