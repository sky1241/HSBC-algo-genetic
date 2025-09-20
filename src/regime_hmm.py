"""Utility functions for fitting and applying HMM-based regime filters.

The module implements three main entry points:

```
fit_hmm(train_features, K=[2, 3, 4], seed=None)
predict_hmm(model, features)
apply_hmm(train_df, test_df, cfg)
```

`fit_hmm` trains Gaussian HMM candidates and rejects configurations where
some regimes are too rare when measured both in cumulative duration and in the
number of trades. `predict_hmm` simply decodes a feature matrix with a frozen
model, while `apply_hmm` orchestrates the workflow: scale the training features,
fit the best model and decode only the out-of-sample rows. If every candidate is
rejected, a Fourier based fallback (`rules_fourier`) is used instead. The
functions are written to avoid lookahead: all parameters (scalers, quantiles,
model weights) are derived from the in-sample portion and then frozen before
being applied to the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

__all__ = [
    "HMMFitResult",
    "fit_hmm",
    "predict_hmm",
    "apply_hmm",
    "rules_fourier",
]


@dataclass(frozen=True)
class HMMFitResult:
    """Bundle returned by :func:`fit_hmm`.

    Attributes
    ----------
    model:
        The GaussianHMM instance selected according to the BIC criterion.
    K:
        Number of hidden states used by ``model``.
    bic:
        Bayesian information criterion computed on the training window.
    state_summary:
        DataFrame indexed by state id with coverage statistics
        (count, cumulative duration/trades and the associated fractions).
    train_states:
        Hidden state sequence decoded on the training window. It is provided for
        diagnostics only and is never used for out-of-sample predictions in
        :func:`apply_hmm`.
    """

    model: GaussianHMM
    K: int
    bic: float
    state_summary: pd.DataFrame
    train_states: pd.Series


def _as_dataframe(data: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> pd.DataFrame:
    """Return *data* as a numeric :class:`~pandas.DataFrame`.

    Non numeric columns are dropped. A :class:`ValueError` is raised when no
    numeric feature is left after sanitisation.
    """

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("train_features must contain at least one numeric column")
    return numeric_df


def _prepare_measure(series: Optional[Iterable[float]], index: pd.Index, default: float) -> pd.Series:
    """Convert ``series`` to a float :class:`~pandas.Series` aligned on *index*.

    When ``series`` is *None*, a constant series filled with ``default`` is
    returned. The helper ensures that no negative entries slip into the
    downstream computations.
    """

    if series is None:
        values = np.full(len(index), float(default), dtype=float)
        return pd.Series(values, index=index, dtype=float)
    series_pd = pd.Series(series, index=index, dtype=float)
    if (series_pd < 0).any():
        raise ValueError("Duration/trade count series must be non-negative")
    return series_pd


def _bic(model: GaussianHMM, X: np.ndarray) -> float:
    """Compute the Bayesian information criterion for ``model`` on ``X``."""

    n, n_features = X.shape
    ll = float(model.score(X))
    # Number of free parameters: means + diag covariances + transitions + startprob
    params = model.n_components * n_features  # means
    params += model.n_components * n_features  # diag covariances
    params += model.n_components * (model.n_components - 1)  # transition matrix
    params += (model.n_components - 1)  # start probabilities
    return params * np.log(max(n, 1)) - 2.0 * ll


def fit_hmm(
    train_features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    K: Sequence[int] | None = (2, 3, 4),
    seed: Optional[int] = None,
    *,
    durations: Optional[Iterable[float]] = None,
    trade_counts: Optional[Iterable[float]] = None,
    min_duration_frac: float = 0.05,
    min_trade_frac: float = 0.05,
    min_trades: float = 5.0,
    min_obs: int = 15,
) -> HMMFitResult:
    """Estimate an HMM on the *train_features* matrix.

    Parameters
    ----------
    train_features:
        Feature matrix used for the fit. It can be a :class:`~pandas.DataFrame`,
        a two-dimensional :class:`numpy.ndarray` or any sequence convertible to a
        DataFrame. The function drops non-numeric columns.
    K:
        Iterable of candidate numbers of regimes. Candidates producing *rare*
        states are skipped. A state is marked as rare when its cumulative
        duration or the number of trades represents less than the respective
        fraction thresholds or when it contains fewer than ``min_obs`` samples.
    seed:
        Random seed forwarded to :class:`hmmlearn.hmm.GaussianHMM` for
        reproducibility.
    durations, trade_counts:
        Optional iterables aligned with ``train_features``. They are used to
        measure the coverage of each state. When omitted, unit weights are
        assumed.
    min_duration_frac, min_trade_frac:
        Fractional coverage thresholds (in [0, 1]). If a state represents less
        than the threshold of the total duration/trade volume, the model is
        rejected.
    min_trades:
        Minimal absolute number of trades required for each state. It acts as a
        safeguard when the total number of trades is small.
    min_obs:
        Minimal number of samples assigned to each hidden state.

    Returns
    -------
    HMMFitResult
        Selected model and diagnostics.

    Raises
    ------
    ValueError
        If every candidate in ``K`` is rejected because of rare states or
        because the input matrix is empty.
    """

    features = _as_dataframe(train_features).dropna()
    if features.empty:
        raise ValueError("Cannot fit an HMM on an empty feature matrix")

    index = features.index if isinstance(features, pd.DataFrame) else pd.RangeIndex(len(features))
    durations_series = _prepare_measure(durations, index, default=1.0)
    trades_series = _prepare_measure(trade_counts, index, default=1.0)

    X = features.to_numpy(dtype=float)
    valid_results: List[HMMFitResult] = []
    seen_k = [] if K is None else list(dict.fromkeys(int(k) for k in K if int(k) > 0))
    if not seen_k:
        seen_k = [2, 3, 4]

    for k in seen_k:
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
        )
        model.fit(X)
        states = pd.Series(model.predict(X), index=index, name="state")
        counts = states.value_counts().sort_index()
        duration_by_state = durations_series.groupby(states).sum().sort_index()
        trade_by_state = trades_series.groupby(states).sum().sort_index()
        duration_frac = duration_by_state / duration_by_state.sum() if duration_by_state.sum() else duration_by_state * 0.0
        trade_frac = trade_by_state / trade_by_state.sum() if trade_by_state.sum() else trade_by_state * 0.0

        rare_mask = (
            (duration_frac < min_duration_frac)
            | (trade_frac < min_trade_frac)
            | (trade_by_state < min_trades)
            | (counts < min_obs)
        )
        if bool(rare_mask.any()):
            continue

        summary = pd.DataFrame(
            {
                "count": counts,
                "duration": duration_by_state,
                "duration_frac": duration_frac,
                "trades": trade_by_state,
                "trade_frac": trade_frac,
            }
        )
        result = HMMFitResult(
            model=model,
            K=k,
            bic=_bic(model, X),
            state_summary=summary,
            train_states=states,
        )
        valid_results.append(result)

    if not valid_results:
        raise ValueError("All HMM candidates were rejected due to rare states")

    best = min(valid_results, key=lambda res: (res.bic, res.K))
    return best


def predict_hmm(model: GaussianHMM, features: pd.DataFrame | np.ndarray) -> pd.Series:
    """Decode hidden states for ``features`` using a frozen ``model``.

    Parameters
    ----------
    model:
        Previously fitted :class:`hmmlearn.hmm.GaussianHMM`.
    features:
        Feature matrix matching the training layout. It must already be scaled
        with the statistics computed on the training window.

    Returns
    -------
    pandas.Series
        Sequence of hidden states aligned with the rows of ``features``. The
        function never mutates or refits ``model`` which guarantees that the
        parameters stay frozen when the series is applied out-of-sample.
    """

    df = _as_dataframe(features)
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    X = df.to_numpy(dtype=float)
    states = model.predict(X)
    return pd.Series(states, index=df.index, name="state")


def rules_fourier(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict[str, object]) -> pd.Series:
    """Fallback regime classifier based on Fourier quantiles.

    The rule maps each out-of-sample row to ``{"trend", "range", "transition"}``
    according to the quantiles (defaults to 25/75%) of ``P1`` and ``LFP``
    measured on the training window. Missing Fourier columns trigger a
    :class:`ValueError` to make the failure explicit.
    """

    fourier_cols = cfg.get("fourier_cols") if isinstance(cfg, dict) else None
    if not isinstance(fourier_cols, dict):
        raise ValueError("cfg must define a 'fourier_cols' mapping for fallback")
    p1_col = fourier_cols.get("P1")
    lfp_col = fourier_cols.get("LFP")
    if p1_col is None or lfp_col is None:
        raise ValueError("fourier_cols must contain 'P1' and 'LFP' keys")
    if p1_col not in train_df or lfp_col not in train_df:
        raise ValueError("Training dataframe is missing required Fourier columns")
    if p1_col not in test_df or lfp_col not in test_df:
        raise ValueError("Test dataframe is missing required Fourier columns")

    quantiles = cfg.get("quantiles", (0.25, 0.75)) if isinstance(cfg, dict) else (0.25, 0.75)
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 2:
        raise ValueError("quantiles must be a length-2 tuple/list")
    q_low, q_high = float(min(quantiles)), float(max(quantiles))

    p1_train = train_df[p1_col].astype(float)
    lfp_train = train_df[lfp_col].astype(float)
    p1_low, p1_high = p1_train.quantile(q_low), p1_train.quantile(q_high)
    lfp_low, lfp_high = lfp_train.quantile(q_low), lfp_train.quantile(q_high)

    p1_test = test_df[p1_col].astype(float)
    lfp_test = test_df[lfp_col].astype(float)
    unknown_mask = p1_test.isna() | lfp_test.isna()
    states = np.full(len(test_df), "transition", dtype=object)
    states[(lfp_test >= lfp_high) & (p1_test >= p1_high)] = "trend"
    states[(lfp_test <= lfp_low) | (p1_test <= p1_low)] = "range"
    states[unknown_mask] = "unknown"
    return pd.Series(states, index=test_df.index, name="regime")


def apply_hmm(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict[str, object]) -> Dict[str, object]:
    """Fit an HMM on ``train_df`` and decode the frozen model on ``test_df``.

    Parameters
    ----------
    train_df, test_df:
        DataFrames split chronologically. Only ``train_df`` is used to fit the
        scaling statistics and the HMM parameters. ``test_df`` is never leaked
        into the calibration step which prevents lookahead bias.
    cfg:
        Dictionary describing the configuration. The most relevant keys are:

        - ``feature_cols`` (list): feature names used by the HMM.
        - ``duration_col`` / ``trade_count_col`` (str, optional): cumulative
          measures used to reject rare states.
        - ``K`` (iterable of int): candidate number of regimes.
        - ``seed`` (int, optional): random seed for reproducibility.
        - ``min_duration_frac`` / ``min_trade_frac`` / ``min_trades`` /
          ``min_obs``: thresholds forwarded to :func:`fit_hmm`.
        - ``fourier_cols`` and ``quantiles``: parameters for
          :func:`rules_fourier` when the HMM is rejected.
        - ``return_train_states`` (bool): whether to include the decoded train
          sequence for diagnostics. It defaults to ``False`` to emphasise that
          the production workflow should only consume the out-of-sample states.

    Returns
    -------
    dict
        Always contains ``oos_states`` (out-of-sample sequence as a
        :class:`pandas.Series`) and ``scaler`` (median/MAD statistics). When the
        HMM succeeds, ``model``, ``train_summary`` and metadata (``K``, ``bic``)
        are provided. Otherwise, ``model`` is ``None`` and ``fallback`` equals
        ``"rules_fourier"``. In both cases the predictions are generated using a
        model whose parameters are frozen on the training slice only.
    """

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise TypeError("train_df and test_df must be pandas DataFrames")
    if train_df.empty or test_df.empty:
        raise ValueError("Both train_df and test_df must be non-empty")

    feature_cols = cfg.get("feature_cols") if isinstance(cfg, dict) else None
    if not isinstance(feature_cols, (list, tuple)) or not feature_cols:
        raise ValueError("cfg must define a non-empty 'feature_cols' list")
    missing_train = [col for col in feature_cols if col not in train_df]
    missing_test = [col for col in feature_cols if col not in test_df]
    if missing_train:
        raise ValueError(f"train_df is missing feature columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test_df is missing feature columns: {missing_test}")

    train_features = train_df[feature_cols].astype(float)
    test_features = test_df[feature_cols].astype(float)

    train_valid = train_features.dropna()
    if train_valid.empty:
        raise ValueError("Training features contain only NaN values")
    test_valid = test_features.dropna()

    median = train_valid.median()
    mad = (train_valid - median).abs().median()
    mad = mad.replace(0.0, 1.0)
    train_scaled_full = (train_features - median) / mad
    test_scaled_full = (test_features - median) / mad
    train_scaled = train_scaled_full.dropna()
    test_scaled = test_scaled_full.dropna()

    duration_col = cfg.get("duration_col") if isinstance(cfg, dict) else None
    trade_col = cfg.get("trade_count_col") if isinstance(cfg, dict) else None
    durations = None
    trades = None
    if duration_col is not None:
        if duration_col not in train_df:
            raise ValueError(f"duration_col '{duration_col}' missing from training dataframe")
        durations = train_df.loc[train_scaled.index, duration_col]
    if trade_col is not None:
        if trade_col not in train_df:
            raise ValueError(f"trade_count_col '{trade_col}' missing from training dataframe")
        trades = train_df.loc[train_scaled.index, trade_col]

    fit_kwargs = dict(
        durations=durations,
        trade_counts=trades,
        min_duration_frac=cfg.get("min_duration_frac", 0.05),
        min_trade_frac=cfg.get("min_trade_frac", 0.05),
        min_trades=cfg.get("min_trades", 5.0),
        min_obs=cfg.get("min_obs", 15),
    )

    candidates = cfg.get("K", (2, 3, 4)) if isinstance(cfg, dict) else (2, 3, 4)
    try:
        fit_result = fit_hmm(
            train_scaled,
            K=candidates,
            seed=cfg.get("seed") if isinstance(cfg, dict) else None,
            **fit_kwargs,
        )
        train_states = predict_hmm(fit_result.model, train_scaled)
        train_states = train_states.reindex(train_features.index)
        if test_scaled.empty:
            oos_states = pd.Series(np.nan, index=test_features.index, name="state")
        else:
            oos_states = predict_hmm(fit_result.model, test_scaled)
            oos_states = oos_states.reindex(test_features.index)
        result: Dict[str, object] = {
            "model": fit_result.model,
            "oos_states": oos_states,
            "train_summary": fit_result.state_summary,
            "scaler": {"median": median, "mad": mad},
            "K": fit_result.K,
            "bic": fit_result.bic,
        }
        if cfg.get("return_train_states"):
            result["train_states"] = train_states
        return result
    except ValueError as err:
        fallback_states = rules_fourier(train_df, test_df, cfg)
        result = {
            "model": None,
            "oos_states": fallback_states,
            "train_summary": None,
            "scaler": {"median": median, "mad": mad},
            "fallback": "rules_fourier",
            "reason": str(err),
        }
        if cfg.get("return_train_states"):
            result["train_states"] = rules_fourier(train_df, train_df, cfg)
        return result
