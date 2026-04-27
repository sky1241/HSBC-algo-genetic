"""Regime-Aware LightGBM Directional Classifier — ALPHA-3.

Pipeline:
    1. HMM Gaussien 2 etats sur (log_returns, realized_vol) -> regime probability.
    2. Features tabulaires (ALPHA-1 + funding + temporal cyclic + momentum).
    3. LightGBM multi-class {-1, 0, +1} avec regime probability comme feature.
    4. Walk-forward strict avec purge (= horizon de label) entre train/test.

Reference:
    Karpacz, Vovk & Schmidt (2025) "Regime-Aware LightGBM Walk-Forward
    Framework", MDPI Electronics 15(6) 1334. Sharpe portfolio crypto = 1.18
    (CI 95% [0.53, 1.84]) — IC large => edge fragile.

Conventions strictes:
    - Pas de lookahead pour les FEATURES: tout est .shift(0) ou rolling sur
      data <= t.
    - LOOKAHEAD legitime pour le LABEL (forward return horizon=3 H2 = 6h),
      strictement utilise au training.
    - Le signal est genere sur t et applique au PnL[t+1] (cf. simulate_returns).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False
    lgb = None

from hmmlearn.hmm import GaussianHMM

from . import range_detector as rd


# ============================================================
# 0. Helpers — fast Hurst (vectorized) for backtest scale
# ============================================================

def _fast_rolling_hurst(prices: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Vectorized rolling Hurst exponent (R/S log-log slope on log-prices).

    Equivalent (modulo numerical precision) a `range_detector.rolling_hurst`,
    mais ~50x plus rapide grace au fait qu'on ne fait pas un polyfit Python
    par fenetre — on utilise le standard deviation rolling de differences a
    `lag` fixe, puis polyfit unique sur (log_lag, log_std_per_window).

    Astuce : pour chaque lag L, std(log_p[t-window+1..t] differences L) =
    std(log_p[t-window+1..t-L+1] - log_p[t-window+1+L..t]). On le calcule via
    rolling std d'une serie de differences de log-prices sur la fenetre
    (window - L + 1).
    """
    lp = np.log(prices.replace(0, np.nan).astype(float))
    lags = np.arange(2, max_lag + 1)
    # diffs[L] est une serie de diff(L) — ses bars sont l'index original
    # Pour chaque bar t, on veut std des differences L sur les ~window obs
    # se terminant en t. On utilise rolling std sur la serie diff(L).
    log_taus = np.full((len(lp), len(lags)), np.nan, dtype=float)
    for j, L in enumerate(lags):
        d = lp.diff(L)
        # rolling std sur les `window - L` derniers diffs
        roll_std = d.rolling(window=window - L, min_periods=max(10, window - L)).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_taus[:, j] = np.log(roll_std.values)

    # Pour chaque bar t : slope = polyfit(log(lags), log_taus[t, :], 1)
    log_lags = np.log(lags.astype(float))
    H = np.full(len(lp), np.nan, dtype=float)
    # Vectorise: pour chaque ligne t, slope = cov / var sur (log_lags, log_taus[t])
    # mais on doit gerer NaN par ligne -> mask
    valid_mask = np.isfinite(log_taus).all(axis=1)
    if valid_mask.any():
        x = log_lags
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        Y = log_taus[valid_mask]  # (n_valid, n_lags)
        y_mean = Y.mean(axis=1, keepdims=True)
        cov = ((x - x_mean) * (Y - y_mean)).sum(axis=1)
        slopes = cov / x_var
        H[valid_mask] = slopes
    return pd.Series(H, index=prices.index)


# ============================================================
# 1. HMM regime fitting
# ============================================================

def fit_hmm_regimes(
    log_returns: pd.Series,
    realized_vol: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    random_state: int = 42,
) -> Tuple[GaussianHMM, pd.Series]:
    """Fit Gaussian HMM sur (returns, vol) 2D.

    Returns
    -------
    (model, regime_prob)
        regime_prob = P(state = "high vol regime") par bar. Pour n_states=2 on
        identifie l'etat "high vol" comme celui dont la moyenne de la 2eme
        feature (vol) est la plus grande, et on retourne sa proba posterieure.
        Pour n_states=3, on retourne la proba de l'etat le plus volatile.
    """
    if not isinstance(log_returns, pd.Series) or not isinstance(realized_vol, pd.Series):
        raise TypeError("log_returns and realized_vol must be pd.Series")

    df = pd.concat([log_returns.rename("ret"), realized_vol.rename("vol")], axis=1).dropna()
    if len(df) < n_states * 10:
        raise ValueError(f"Not enough observations ({len(df)}) to fit HMM with {n_states} states")

    X = df.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-3,
        )
        model.fit(X)

    # Posterior probability of each state per observation
    posteriors = model.predict_proba(X)  # shape (T, n_states)

    # Identify "high vol" state = state with max mean of vol feature (col 1)
    vol_means = model.means_[:, 1]
    high_vol_state = int(np.argmax(vol_means))

    prob_series = pd.Series(np.nan, index=log_returns.index, dtype=float)
    prob_series.loc[df.index] = posteriors[:, high_vol_state]

    return model, prob_series


def predict_hmm_regimes(model: GaussianHMM, log_returns: pd.Series, realized_vol: pd.Series) -> pd.Series:
    """Predict regime probability on new data using a pre-fit HMM."""
    df = pd.concat([log_returns.rename("ret"), realized_vol.rename("vol")], axis=1).dropna()
    if df.empty:
        return pd.Series(np.nan, index=log_returns.index, dtype=float)
    X = df.values.astype(float)
    posteriors = model.predict_proba(X)
    vol_means = model.means_[:, 1]
    high_vol_state = int(np.argmax(vol_means))
    out = pd.Series(np.nan, index=log_returns.index, dtype=float)
    out.loc[df.index] = posteriors[:, high_vol_state]
    return out


# ============================================================
# 2. Feature engineering (NO LOOKAHEAD)
# ============================================================

def build_features(
    df: pd.DataFrame,
    funding_h8: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Construit features a partir d'un DataFrame OHLCV.

    Toutes les features utilisent uniquement data jusqu'a t (close inclus).
    Les rolling/shift sont causaux. Pas d'usage de close.shift(-N).

    Returns
    -------
    DataFrame index=df.index, colonnes:
        ret_1, ret_4, ret_12, vol_20, range_pct,
        adx_14, er_30, bbw_20, bbw_squeeze, hurst_100,
        hour_sin, hour_cos, dow_sin, dow_cos,
        funding_current, funding_change_8h,
        mom_6, mom_12, mom_24
    """
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("df must contain open/high/low/close columns")

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # --- Returns / vol / range (causaux)
    log_close = np.log(close.replace(0, np.nan))
    ret_1 = log_close.diff()
    out["ret_1"] = ret_1
    out["ret_4"] = log_close.diff(4)
    out["ret_12"] = log_close.diff(12)
    out["vol_20"] = ret_1.rolling(20).std()
    out["range_pct"] = (high - low) / close.replace(0, np.nan)

    # --- ALPHA-1 indicators
    out["adx_14"] = rd.adx(high, low, close, period=14)
    out["er_30"] = rd.efficiency_ratio(close, period=30)
    out["bbw_20"] = rd.bollinger_band_width(close, period=20)
    out["bbw_squeeze"] = rd.bbw_squeeze(close, period=20).astype(float)
    # Hurst rolling: on utilise la version vectorisee locale (~50x faster)
    out["hurst_100"] = _fast_rolling_hurst(close, window=100, max_lag=20)

    # --- Temporal cyclic encoding
    if isinstance(df.index, pd.DatetimeIndex):
        hod = df.index.hour.values.astype(float)
        dow = df.index.dayofweek.values.astype(float)
        out["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 1.0
        out["dow_sin"] = 0.0
        out["dow_cos"] = 1.0

    # --- Funding (no lookahead: ffill sur valeurs passees; shift d'1 bar pour
    #     etre sur que la valeur observable a t est la derniere VRAIMENT publiee)
    if funding_h8 is not None and len(funding_h8) > 0:
        f_aligned = funding_h8.reindex(df.index, method="ffill")
        # shift(1) garde-fou contre publication a la meme bar t
        out["funding_current"] = f_aligned.shift(1)
        out["funding_change_8h"] = f_aligned.diff().shift(1)
    else:
        out["funding_current"] = 0.0
        out["funding_change_8h"] = 0.0

    # --- Momentum (close / close.shift(N) - 1) — causal
    for n in (6, 12, 24):
        out[f"mom_{n}"] = close / close.shift(n) - 1.0

    return out


# ============================================================
# 3. Directional label (LOOKAHEAD legitime pour le label)
# ============================================================

def label_directional(
    close: pd.Series,
    horizon: int = 3,
    flat_threshold: float = 0.002,
) -> pd.Series:
    """Label trinaire base sur forward return.

    Y[t] = sign(close[t+horizon] / close[t] - 1) si |change| > flat_threshold,
    sinon 0.

    -1 = down, 0 = flat (faible amplitude), +1 = up.

    LOOKAHEAD volontaire (label seulement); les features doivent etre causales.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd = close.shift(-horizon) / close - 1.0
    y = pd.Series(0, index=close.index, dtype=int)
    y[fwd > flat_threshold] = 1
    y[fwd < -flat_threshold] = -1
    # NaN au queue (shift(-N) cree des NaN)
    y[fwd.isna()] = 0
    return y


# ============================================================
# 4. RegimeLightGBM classifier wrapper
# ============================================================

@dataclass
class RegimeLGBMConfig:
    objective: str = "multiclass"
    num_class: int = 3  # {-1, 0, +1} mappe sur {0, 1, 2}
    learning_rate: float = 0.05
    num_leaves: int = 31
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 30
    num_boost_round: int = 200
    early_stopping_rounds: int = 30
    seed: int = 42
    verbose: int = -1


class RegimeLightGBM:
    """LightGBM directional classifier avec feature regime-aware.

    Le `regime` (P(high-vol)) est ajoute comme feature au moment du fit/predict.
    """

    LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
    CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: Optional[RegimeLGBMConfig] = None):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")
        self.config = config or RegimeLGBMConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_names_: list[str] = []

    def _make_X(self, X: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
        Xc = X.copy()
        Xc["regime_p"] = regime.reindex(Xc.index)
        return Xc

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        regime_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        regime_val: Optional[pd.Series] = None,
    ) -> "RegimeLightGBM":
        Xtr = self._make_X(X_train, regime_train)
        # Map labels -1/0/1 -> 0/1/2
        ytr = y_train.map(self.LABEL_TO_CLASS).astype(int)
        # Drop NaN rows (au cas ou)
        mask = Xtr.notna().all(axis=1) & ytr.notna()
        Xtr = Xtr[mask]
        ytr = ytr[mask]
        if Xtr.empty:
            raise ValueError("No valid training rows after NaN drop")

        self.feature_names_ = list(Xtr.columns)
        train_ds = lgb.Dataset(Xtr, label=ytr.values)

        params = {
            "objective": self.config.objective,
            "num_class": self.config.num_class,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "metric": "multi_logloss",
            "verbose": self.config.verbose,
            "seed": self.config.seed,
            "n_jobs": -1,
        }

        valid_sets = [train_ds]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]

        if X_val is not None and y_val is not None and regime_val is not None:
            Xv = self._make_X(X_val, regime_val)
            yv = y_val.map(self.LABEL_TO_CLASS).astype(int)
            mv = Xv.notna().all(axis=1) & yv.notna()
            Xv = Xv[mv]
            yv = yv[mv]
            if not Xv.empty:
                val_ds = lgb.Dataset(Xv, label=yv.values, reference=train_ds)
                valid_sets.append(val_ds)
                valid_names.append("val")
                callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                params,
                train_ds,
                num_boost_round=self.config.num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
        return self

    def predict_proba(self, X_test: pd.DataFrame, regime_test: pd.Series) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit yet")
        Xt = self._make_X(X_test, regime_test)
        # Reindex columns dans le meme ordre que train
        Xt = Xt.reindex(columns=self.feature_names_)
        # LightGBM tolere NaN. On predict directement.
        proba = self.model.predict(Xt)  # shape (N, 3)
        return proba

    def predict(self, X_test: pd.DataFrame, regime_test: pd.Series) -> np.ndarray:
        """Returns labels in {-1, 0, +1}."""
        proba = self.predict_proba(X_test, regime_test)
        cls = np.argmax(proba, axis=1)
        return np.array([self.CLASS_TO_LABEL[c] for c in cls], dtype=int)

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not fit yet")
        imps = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names_, imps.astype(float)))


__all__ = [
    "fit_hmm_regimes",
    "predict_hmm_regimes",
    "build_features",
    "label_directional",
    "RegimeLGBMConfig",
    "RegimeLightGBM",
]
