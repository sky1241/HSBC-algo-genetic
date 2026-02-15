#!/usr/bin/env python3
"""
NHHM v2 - Non-Homogeneous HMM using hmmlearn with rolling window.

Key differences from regime_hmm.py:
- Rolling window (10k points) instead of fitting on ALL data → NO LOOK-AHEAD
- Designed for DIRECTION detection (bull/bear), not just regime identification
- Periodic refitting to capture changing market dynamics

Key differences from regime_nhhm.py (v1):
- Uses hmmlearn (stable) instead of statsmodels MarkovRegression (crashes on 60k+)
- Rolling window approach instead of single fit
- Simpler, more robust

Usage:
    from src.regime_nhhm_v2 import NHHMv2, NHHMv2Config, generate_nhhm_labels

    model = NHHMv2()
    results = model.fit_predict_rolling(df)
    # results has columns: p_bull, p_bear, label, regime

    # Or use convenience function:
    generate_nhhm_labels()  # reads BTC_FUSED_2h.csv, saves NHHM_labels.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# ============================================================
# HALVING DATES (fixed, known in advance - no look-ahead)
# ============================================================
HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

CYCLE_LENGTH_DAYS = 1460  # ~4 years


@dataclass
class NHHMv2Config:
    """Configuration for NHHM v2."""
    n_states: int = 2                    # 2 = bull/bear
    window_size: int = 10000             # Rolling window size (points)
    refit_every: int = 500               # Refit every N points (~42 days H2)
    min_window: int = 2000               # Minimum points before first fit
    n_iter: int = 100                    # HMM EM iterations
    covariance_type: str = "full"        # Covariance type
    n_random_starts: int = 5             # Multiple random starts for stability
    bull_threshold: float = 0.5          # P(bull) > threshold → label=1
    feature_cols: Optional[list] = None  # Feature columns (auto if None)


def _halving_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute halving cycle features. Fixed dates = no look-ahead."""
    n = len(timestamps)
    progress = np.full(n, np.nan)
    direction = np.full(n, 0.0)

    for i, ts in enumerate(timestamps):
        ts = pd.Timestamp(ts)
        last_h = None
        for h in HALVING_DATES:
            if h <= ts:
                last_h = h
        if last_h is None:
            continue
        d = (ts - last_h).days
        progress[i] = min(d / CYCLE_LENGTH_DAYS, 1.0)
        # Bull phases: 0-540 days post-halving, Bear: 540-1095, Late: 1095+
        if d < 540:
            direction[i] = 1.0
        elif d < 1095:
            direction[i] = -1.0
        else:
            direction[i] = 0.0

    return pd.DataFrame({
        'halving_progress': progress,
        'halving_direction': direction,
    }, index=timestamps)


def build_features(
    df: pd.DataFrame,
    include_fourier: bool = True,
    include_halving: bool = False,
) -> pd.DataFrame:
    """
    Build features for NHHM v2. All features are CAUSAL (no look-ahead).

    Returns DataFrame with market-based features:
    - momentum_6/12/24: price momentum at different horizons
    - vol_ratio: short/long volatility ratio
    - rsi_centered: RSI - 50 (mean-reversion signal)
    - dist_ma20: distance from 20-period MA
    - return_1: 1-bar return (direction signal)
    - halving_progress/direction: (optional, off by default)
    - P1_period, LFP_ratio: Fourier features (if in df)
    """
    out = pd.DataFrame(index=df.index)
    close = df['close']

    # Short-term return (causal, direct direction signal)
    out['return_1'] = close.pct_change(1)

    # Momentum at multiple horizons (causal)
    out['momentum_6'] = close.pct_change(6)
    out['momentum_12'] = close.pct_change(12)
    out['momentum_24'] = close.pct_change(24)

    # Volatility ratio (causal)
    log_ret = np.log(close / close.shift(1))
    vol_short = log_ret.rolling(10).std()
    vol_long = log_ret.rolling(50).std()
    out['vol_ratio'] = vol_short / (vol_long + 1e-10)

    # RSI (causal, mean-reversion signal)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    out['rsi_centered'] = rsi - 50

    # Distance from MA (causal)
    ma20 = close.rolling(20).mean()
    out['dist_ma20'] = (close - ma20) / ma20

    # Halving features (optional, off by default to avoid domination)
    if include_halving:
        halving = _halving_features(df.index)
        out['halving_progress'] = halving['halving_progress']
        out['halving_direction'] = halving['halving_direction']

    # Fourier features if already computed in df
    if include_fourier:
        for col in ['P1_period', 'LFP_ratio']:
            if col in df.columns:
                out[col] = df[col]

    return out


class NHHMv2:
    """
    Non-Homogeneous HMM v2 using hmmlearn + rolling window.

    The "non-homogeneous" aspect: periodic refitting on a sliding window
    means transition probabilities effectively change over time.

    CLEAN: at time t, only uses data from [t - window_size, t].
    """

    def __init__(self, config: Optional[NHHMv2Config] = None):
        self.config = config or NHHMv2Config()

    def _fit_single(self, X: np.ndarray, random_state: int) -> tuple:
        """Fit one HMM, return (model, log-likelihood)."""
        model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        return model, model.score(X)

    def _fit_best(self, X: np.ndarray, base_seed: int = 42) -> GaussianHMM:
        """Fit HMM with multiple random starts, keep best log-likelihood."""
        best_model = None
        best_score = -np.inf

        for i in range(self.config.n_random_starts):
            try:
                model, score = self._fit_single(X, random_state=base_seed + i)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("All HMM random starts failed")
        return best_model

    def _identify_bull_state(self, model: GaussianHMM, X: np.ndarray,
                              return_col_idx: int = 0) -> int:
        """
        Identify which HMM state is 'bull'.

        Uses mean of the return/momentum feature per state.
        State with higher mean return = bull.
        Solves the HMM label-switching problem.
        """
        states = model.predict(X)
        state_return = {}
        for s in range(self.config.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_return[s] = float(X[mask, return_col_idx].mean())
            else:
                state_return[s] = 0.0
        return max(state_return, key=state_return.get)

    def fit_predict_rolling(
        self,
        df: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fit HMM on rolling window, predict P(bull) at each point.

        CLEAN: at each refit point t, model is fit on data BEFORE t,
        then predicts FORWARD from t.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            features: Pre-computed features (builds them if None)
            verbose: Print progress

        Returns:
            DataFrame with: timestamp, p_bull, p_bear, label, regime
        """
        if features is None:
            if verbose:
                print("Building features...")
            features = build_features(df)

        # Select feature columns
        if self.config.feature_cols:
            cols = [c for c in self.config.feature_cols if c in features.columns]
        else:
            cols = list(features.columns)

        if verbose:
            print(f"Features: {cols}")

        # Drop NaN rows
        feat_clean = features[cols].dropna()
        X_all = feat_clean.values.astype(np.float64)
        valid_idx = feat_clean.index
        n = len(X_all)

        if verbose:
            print(f"Valid points: {n}")
            print(f"Window: {self.config.window_size}, refit every: {self.config.refit_every}")

        # Output arrays
        p_bull = np.full(n, np.nan)
        p_bear = np.full(n, np.nan)

        # Return column index for bull/bear identification
        # Prefer return_1 (direct direction signal), fallback to momentum_12
        if 'return_1' in cols:
            return_idx = cols.index('return_1')
        elif 'momentum_12' in cols:
            return_idx = cols.index('momentum_12')
        else:
            return_idx = 0

        # Compute refit schedule
        refit_points = list(range(
            self.config.min_window,
            n,
            self.config.refit_every,
        ))

        if verbose:
            print(f"Planned refits: {len(refit_points)}")

        current_model = None
        current_bull_state = 0
        current_mean = None
        current_std = None
        n_fits = 0

        for fit_idx in refit_points:
            # Training window: [start, fit_idx) — only PAST data
            start = max(0, fit_idx - self.config.window_size)
            X_window = X_all[start:fit_idx].copy()

            if len(X_window) < self.config.min_window:
                continue

            # Normalize (per-window z-score)
            w_mean = X_window.mean(axis=0)
            w_std = X_window.std(axis=0)
            w_std[w_std == 0] = 1.0
            X_norm = (X_window - w_mean) / w_std

            # Fit
            try:
                model = self._fit_best(X_norm, base_seed=42 + n_fits)
                bull_state = self._identify_bull_state(model, X_norm, return_idx)

                current_model = model
                current_bull_state = bull_state
                current_mean = w_mean
                current_std = w_std
                n_fits += 1

                if verbose and (n_fits <= 3 or n_fits % 20 == 0):
                    print(f"  Fit #{n_fits}: idx={fit_idx}/{n}, "
                          f"bull_state={bull_state}, "
                          f"date~{valid_idx[fit_idx]}")

            except Exception as e:
                if verbose:
                    print(f"  Fit FAILED at idx={fit_idx}: {e}")
                continue

            # Predict on NEXT chunk: [fit_idx, fit_idx + refit_every)
            if current_model is not None:
                end = min(fit_idx + self.config.refit_every, n)
                X_chunk = X_all[fit_idx:end].copy()
                X_chunk_norm = (X_chunk - current_mean) / current_std

                try:
                    probs = current_model.predict_proba(X_chunk_norm)
                    bear_state = 1 - current_bull_state
                    p_bull[fit_idx:end] = probs[:, current_bull_state]
                    p_bear[fit_idx:end] = probs[:, bear_state]
                except Exception:
                    pass

        if verbose:
            valid_count = np.sum(~np.isnan(p_bull))
            print(f"\nDone: {n_fits} fits, {valid_count}/{n} predictions")

        # Build result
        result = pd.DataFrame(index=valid_idx)
        result['p_bull'] = p_bull
        result['p_bear'] = p_bear
        result['label'] = np.where(
            np.isnan(p_bull), np.nan,
            (p_bull > self.config.bull_threshold).astype(float)
        )
        result['regime'] = result['label'].map({1.0: 'bull', 0.0: 'bear'})
        result['timestamp'] = result.index

        return result


def generate_nhhm_labels(
    csv_path: str = 'data/BTC_FUSED_2h.csv',
    output_path: str = 'data/NHHM_labels.csv',
    config: Optional[NHHMv2Config] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate NHHM labels from BTC data. Convenience function.

    Loads data → builds features → rolling HMM → saves labels CSV.
    """
    if verbose:
        print("=" * 70)
        print("NHHM v2 - Rolling HMM Label Generation")
        print("=" * 70)

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    if verbose:
        print(f"Loaded {len(df)} rows from {csv_path}")

    features = build_features(df)
    if verbose:
        print(f"Features: {list(features.columns)}")

    model = NHHMv2(config or NHHMv2Config())
    result = model.fit_predict_rolling(df, features, verbose=verbose)

    # Stats
    if verbose:
        valid = result['label'].notna()
        n_valid = int(valid.sum())
        if n_valid > 0:
            n_bull = int((result.loc[valid, 'label'] == 1).sum())
            n_bear = int((result.loc[valid, 'label'] == 0).sum())
            print(f"\n{'=' * 70}")
            print(f"LABELS: {n_bull} bull ({100*n_bull/n_valid:.1f}%), "
                  f"{n_bear} bear ({100*n_bear/n_valid:.1f}%)")
            print(f"{'=' * 70}")

    output = result[['timestamp', 'label', 'regime', 'p_bull', 'p_bear']].dropna(subset=['label'])
    output['label'] = output['label'].astype(int)
    output.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(output)} rows to {output_path}")

    return result


__all__ = ['NHHMv2', 'NHHMv2Config', 'build_features', 'generate_nhhm_labels']
