#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test of NHHM (Non-Homogeneous HMM) implementation.

This script validates that the NHHM model:
1. Fits successfully on BTC data
2. Produces directional probabilities (P(bull), P(bear))
3. Generates trading signals
4. Computes meaningful metrics (hit rate, Sharpe)

Usage:
    python scripts/test_nhhm_quick.py

Expected output:
    - Fit summary
    - P(bull) and P(bear) statistics
    - Signal distribution
    - Basic performance metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.regime_nhhm import NHHM, NHHMConfig, build_nhhm_features


def main() -> int:
    print("=" * 60)
    print("NHHM Quick Test - Non-Homogeneous Hidden Markov Model")
    print("=" * 60)

    # Load data
    csv_path = _ROOT / "data" / "BTC_FUSED_2h.csv"
    if not csv_path.exists():
        print(f"ERROR: Data file not found: {csv_path}")
        return 1

    print(f"\n1. Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"   Loaded {len(df)} rows, from {df.index[0]} to {df.index[-1]}")

    # Build features
    print("\n2. Building NHHM features...")
    df = build_nhhm_features(df)

    # Check for NaN
    initial_len = len(df)
    df = df.dropna()
    print(f"   After dropna: {len(df)} rows ({initial_len - len(df)} dropped)")

    # Show feature stats
    feature_cols = ['momentum_12', 'vol_ratio', 'rsi_centered', 'dist_ma20']
    print(f"\n   Feature statistics:")
    for col in feature_cols:
        print(f"   - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    # Split train/test
    train_ratio = 0.7
    n = len(df)
    split_idx = int(n * train_ratio)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    print(f"\n3. Train/Test split: {len(df_train)} / {len(df_test)} rows")

    # Fit NHHM
    print("\n4. Fitting NHHM model...")
    config = NHHMConfig(
        n_regimes=2,  # bull vs bear
        forward_horizon=12,  # 24h ahead (12 bars * 2h)
        switching_variance=True
    )
    nhhm = NHHM(config)

    try:
        nhhm.fit(
            df_train,
            exog_tvtp_cols=feature_cols,
            close_col='close',
            verbose=True
        )
    except Exception as e:
        print(f"ERROR during fit: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Get predictions
    print("\n5. Generating predictions...")
    try:
        result = nhhm.predict(df_train, exog_tvtp_cols=feature_cols)
    except Exception as e:
        print(f"ERROR during predict: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show probability stats
    print(f"\n   P(bull) statistics:")
    print(f"   - mean: {result.p_bull.mean():.4f}")
    print(f"   - std:  {result.p_bull.std():.4f}")
    print(f"   - min:  {result.p_bull.min():.4f}")
    print(f"   - max:  {result.p_bull.max():.4f}")

    print(f"\n   P(bear) statistics:")
    print(f"   - mean: {result.p_bear.mean():.4f}")
    print(f"   - std:  {result.p_bear.std():.4f}")

    print(f"\n   Expected return statistics:")
    print(f"   - mean: {result.expected_return.mean():.6f}")
    print(f"   - std:  {result.expected_return.std():.6f}")

    # Signal distribution
    print(f"\n   Signal distribution:")
    print(f"   - Long (1):    {(result.signal == 1).sum()} ({(result.signal == 1).mean()*100:.1f}%)")
    print(f"   - Short (-1):  {(result.signal == -1).sum()} ({(result.signal == -1).mean()*100:.1f}%)")
    print(f"   - Neutral (0): {(result.signal == 0).sum()} ({(result.signal == 0).mean()*100:.1f}%)")

    # Compute performance metrics
    print("\n6. Computing performance metrics...")
    forward_horizon = config.forward_horizon
    fwd_ret = df_train['close'].shift(-forward_horizon) / df_train['close'] - 1

    # Align signals with forward returns
    # Result signals may be shorter due to NaN handling in model
    n_signals = len(result.signal)
    fwd_ret_aligned = fwd_ret.iloc[:n_signals]
    valid_mask = ~fwd_ret_aligned.isna()
    signal_valid = result.signal[valid_mask.values]
    fwd_ret_valid = fwd_ret_aligned[valid_mask].values

    # Strategy returns
    strategy_ret = signal_valid * fwd_ret_valid

    # Only count non-zero signals
    active_mask = signal_valid != 0
    if active_mask.sum() > 0:
        active_ret = strategy_ret[active_mask]

        hit_rate = (active_ret > 0).mean()
        mean_ret = active_ret.mean()
        std_ret = active_ret.std()
        sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(252 * 12 / forward_horizon)

        print(f"\n   Active trades: {active_mask.sum()}")
        print(f"   Hit rate:      {hit_rate*100:.1f}%")
        print(f"   Mean return:   {mean_ret*100:.4f}%")
        print(f"   Std return:    {std_ret*100:.4f}%")
        print(f"   Sharpe ratio:  {sharpe:.2f}")

        # Regime means
        print(f"\n   Regime means (drift):")
        for regime, mean in result.regime_means.items():
            label = "BULL" if regime == nhhm.bull_regime_idx else "BEAR"
            print(f"   - Regime {regime} ({label}): {mean*100:.4f}%")
    else:
        print("   WARNING: No active signals generated!")

    # Show transition effects
    print("\n7. Transition probability effects:")
    try:
        effects = nhhm.get_transition_effects()
        print(effects.to_string(index=False))
    except Exception as e:
        print(f"   Could not extract transition effects: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("NHHM Quick Test Complete!")
    print("=" * 60)

    if active_mask.sum() > 0 and hit_rate > 0.5:
        print("\n[OK] Model shows positive directional prediction")
        print("     Next step: Integrate with WFA pipeline for full backtest")
    else:
        print("\n[WARN] Model needs tuning - hit rate below 50%")
        print("       Consider: different features, more regimes, or longer horizon")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
