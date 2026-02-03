#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.1 & P3.2 - HMM Feature Engineering and Model Selection

Features for Hidden Markov Model:
- Spectral features (from fourier_features)
- Price/momentum features
- Volatility features
- Ichimoku-derived features

Model selection:
- AIC/BIC for optimal K
- Cross-validation scoring

Usage:
    from src.spectral.hmm_features import HMMFeatureBuilder, select_optimal_k

    builder = HMMFeatureBuilder()
    features_df = builder.build(ohlcv_df)

    best_k, scores = select_optimal_k(features_df, k_range=[3, 4, 5, 6])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .fourier_features import compute_spectral_features, SpectralFeatures


@dataclass
class HMMFeatureSet:
    """Complete feature set for HMM."""
    df: pd.DataFrame
    feature_names: List[str]
    spectral_names: List[str]
    price_names: List[str]
    volatility_names: List[str]


class HMMFeatureBuilder:
    """Builds feature matrix for HMM regime detection.

    Features include:
    - Spectral: LFP, flatness, centroid, entropy, band powers
    - Price: log returns, momentum, distance from MA
    - Volatility: ATR, realized vol, skew, kurtosis
    - Ichimoku: cloud distance, TK cross, Chikou position
    """

    def __init__(
        self,
        window_spectral: int = 256,
        window_vol: int = 20,
        fs: float = 12.0,
    ):
        """
        Args:
            window_spectral: Window for spectral features (bars)
            window_vol: Window for volatility features (bars)
            fs: Sampling frequency (bars per day)
        """
        self.window_spectral = window_spectral
        self.window_vol = window_vol
        self.fs = fs

    def _add_spectral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling spectral features."""
        close = df["close"].values
        n = len(close)

        # Pre-allocate
        features = {
            "lfp": np.full(n, np.nan),
            "flatness": np.full(n, np.nan),
            "centroid": np.full(n, np.nan),
            "entropy": np.full(n, np.nan),
            "band_low": np.full(n, np.nan),
            "band_high": np.full(n, np.nan),
            "slope": np.full(n, np.nan),
        }

        # Rolling computation
        for i in range(self.window_spectral, n):
            chunk = close[i - self.window_spectral:i]
            spec = compute_spectral_features(chunk, fs=self.fs)

            features["lfp"][i] = spec.lfp
            features["flatness"][i] = spec.flatness
            features["centroid"][i] = spec.centroid
            features["entropy"][i] = spec.entropy
            features["band_low"][i] = spec.band_low
            features["band_high"][i] = spec.band_high
            features["slope"][i] = spec.slope

        for name, values in features.items():
            df[f"spec_{name}"] = values

        # Deltas
        df["spec_lfp_delta"] = df["spec_lfp"].diff()
        df["spec_centroid_delta"] = df["spec_centroid"].diff()

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Momentum at various horizons
        for h in [6, 12, 24, 48]:  # 12h, 24h, 48h, 96h for H2
            df[f"momentum_{h}"] = df["close"].pct_change(h)

        # Distance from moving averages
        for w in [20, 50, 100]:
            ma = df["close"].rolling(w).mean()
            df[f"dist_ma{w}"] = (df["close"] - ma) / ma

        # RSI proxy (simple)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # True Range
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR
        df["atr"] = df["tr"].rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]

        # Realized volatility
        df["realized_vol"] = df["log_return"].rolling(self.window_vol).std() * np.sqrt(252 * 12)

        # Volatility ratio (current vs long-term)
        vol_short = df["log_return"].rolling(10).std()
        vol_long = df["log_return"].rolling(50).std()
        df["vol_ratio"] = vol_short / (vol_long + 1e-10)

        # Skewness and Kurtosis
        df["skew"] = df["log_return"].rolling(self.window_vol).skew()
        df["kurtosis"] = df["log_return"].rolling(self.window_vol).kurt()

        return df

    def _add_ichimoku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku-derived features."""
        # Tenkan-sen (9 periods)
        tenkan_high = df["high"].rolling(9).max()
        tenkan_low = df["low"].rolling(9).min()
        df["tenkan"] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (26 periods)
        kijun_high = df["high"].rolling(26).max()
        kijun_low = df["low"].rolling(26).min()
        df["kijun"] = (kijun_high + kijun_low) / 2

        # Senkou Span A
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2)

        # Senkou Span B (52 periods)
        senkou_high = df["high"].rolling(52).max()
        senkou_low = df["low"].rolling(52).min()
        df["senkou_b"] = (senkou_high + senkou_low) / 2

        # Cloud metrics
        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)
        df["cloud_thickness"] = (df["cloud_top"] - df["cloud_bottom"]) / df["close"]

        # Price vs cloud
        df["price_vs_cloud"] = (df["close"] - df["cloud_top"]) / df["close"]
        df["in_cloud"] = ((df["close"] >= df["cloud_bottom"]) & (df["close"] <= df["cloud_top"])).astype(float)

        # TK cross signal
        df["tk_diff"] = (df["tenkan"] - df["kijun"]) / df["close"]

        return df

    def build(
        self,
        df: pd.DataFrame,
        include_spectral: bool = True,
        include_price: bool = True,
        include_volatility: bool = True,
        include_ichimoku: bool = True,
    ) -> HMMFeatureSet:
        """Build complete feature set.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            include_*: Which feature groups to include

        Returns:
            HMMFeatureSet with features DataFrame and metadata
        """
        df = df.copy()

        spectral_names = []
        price_names = []
        volatility_names = []

        if include_spectral:
            df = self._add_spectral_features(df)
            spectral_names = [c for c in df.columns if c.startswith("spec_")]

        if include_price:
            df = self._add_price_features(df)
            price_names = ["log_return", "rsi"] + \
                         [c for c in df.columns if c.startswith("momentum_") or c.startswith("dist_ma")]

        if include_volatility:
            df = self._add_volatility_features(df)
            volatility_names = ["atr_pct", "realized_vol", "vol_ratio", "skew", "kurtosis"]

        if include_ichimoku:
            df = self._add_ichimoku_features(df)
            # Add to price names
            price_names.extend(["cloud_thickness", "price_vs_cloud", "in_cloud", "tk_diff"])

        feature_names = spectral_names + price_names + volatility_names

        return HMMFeatureSet(
            df=df,
            feature_names=feature_names,
            spectral_names=spectral_names,
            price_names=price_names,
            volatility_names=volatility_names,
        )


def fit_hmm(
    features: np.ndarray,
    n_states: int,
    random_state: int = 42,
) -> Tuple[Any, float, float]:
    """Fit a Gaussian HMM and return model with scores.

    Returns:
        (model, log_likelihood, bic)
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise ImportError("hmmlearn required. Install with: pip install hmmlearn")

    # Remove NaN rows
    mask = ~np.any(np.isnan(features), axis=1)
    clean_features = features[mask]

    if len(clean_features) < 100:
        raise ValueError("Insufficient data for HMM fitting")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=random_state,
    )

    model.fit(clean_features)

    # Log likelihood
    log_likelihood = model.score(clean_features)

    # BIC = -2 * log_likelihood + k * log(n)
    # k = number of parameters
    n_features = clean_features.shape[1]
    n_params = n_states * n_features + n_states * n_features + n_states * n_states + n_states - 1
    n_samples = len(clean_features)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return model, log_likelihood, bic


def select_optimal_k(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    k_range: List[int] = [3, 4, 5, 6],
    random_state: int = 42,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """Select optimal number of HMM states using BIC.

    Args:
        features_df: DataFrame with features
        feature_cols: Columns to use as features
        k_range: List of K values to test
        random_state: Random seed

    Returns:
        (best_k, scores_dict)
    """
    features = features_df[feature_cols].values

    # Standardize
    mask = ~np.any(np.isnan(features), axis=1)
    clean = features[mask]
    mean = np.nanmean(clean, axis=0)
    std = np.nanstd(clean, axis=0) + 1e-8
    features_norm = (features - mean) / std

    scores = {}
    for k in k_range:
        try:
            model, ll, bic = fit_hmm(features_norm, k, random_state)
            aic = -2 * ll + 2 * model.n_features_in_ * k

            scores[k] = {
                "log_likelihood": float(ll),
                "bic": float(bic),
                "aic": float(aic),
                "converged": model.monitor_.converged,
            }
        except Exception as e:
            scores[k] = {
                "error": str(e),
                "bic": float("inf"),
            }

    # Best K by BIC (lower is better)
    valid_k = [k for k in scores if "bic" in scores[k] and np.isfinite(scores[k]["bic"])]
    if not valid_k:
        return k_range[0], scores

    best_k = min(valid_k, key=lambda k: scores[k]["bic"])
    return best_k, scores


def predict_states(
    model: Any,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Predict HMM states for new data.

    Args:
        model: Fitted HMM model
        features: Feature array
        mean: Training mean for normalization
        std: Training std for normalization

    Returns:
        Array of state predictions
    """
    features_norm = (features - mean) / std

    # Handle NaN
    states = np.full(len(features), -1)
    mask = ~np.any(np.isnan(features_norm), axis=1)
    if mask.sum() > 0:
        states[mask] = model.predict(features_norm[mask])

    return states


def interpret_states(
    df: pd.DataFrame,
    states: np.ndarray,
    n_states: int,
) -> Dict[int, Dict[str, float]]:
    """Interpret HMM states based on feature statistics.

    Returns dict mapping state -> characteristics
    """
    df = df.copy()
    df["state"] = states

    interpretations = {}
    for state in range(n_states):
        mask = df["state"] == state
        if mask.sum() == 0:
            continue

        subset = df[mask]

        interp = {
            "count": int(mask.sum()),
            "pct": float(mask.mean()),
        }

        # Add feature means if available
        for col in ["log_return", "realized_vol", "spec_lfp", "spec_flatness", "atr_pct"]:
            if col in df.columns:
                val = subset[col].mean()
                if np.isfinite(val):
                    interp[f"mean_{col}"] = float(val)

        # Classify state
        if "mean_log_return" in interp:
            ret = interp["mean_log_return"]
            if ret > 0.001:
                interp["label"] = "bullish"
            elif ret < -0.001:
                interp["label"] = "bearish"
            else:
                interp["label"] = "neutral"

        if "mean_realized_vol" in interp:
            vol = interp["mean_realized_vol"]
            if vol > 0.5:
                interp["volatility"] = "high"
            elif vol < 0.2:
                interp["volatility"] = "low"
            else:
                interp["volatility"] = "medium"

        interpretations[state] = interp

    return interpretations
