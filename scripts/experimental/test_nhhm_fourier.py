#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test NHHM avec features Fourier integrees.

Compare:
1. NHHM baseline (momentum, vol_ratio, rsi)
2. NHHM + Fourier (LFP_signal, P1_norm)
3. NHHM + Fourier + Funding (si 2019+)

Usage:
    py -3 scripts/test_nhhm_fourier.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# Import NHHM
from src.regime_nhhm import (
    NHHM, NHHMConfig, build_nhhm_features, get_recommended_tvtp_cols
)


def load_btc_data(csv_path: str = 'data/BTC_FUSED_2h.csv') -> pd.DataFrame:
    """Load BTC data."""
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df


def evaluate_nhhm(
    df: pd.DataFrame,
    tvtp_cols: list,
    config: NHHMConfig,
    train_ratio: float = 0.7,
    name: str = "Model"
) -> dict:
    """Evaluate NHHM with given features."""
    # Split
    n = len(df)
    split_idx = int(n * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    # Check columns exist
    missing = [c for c in tvtp_cols if c not in df_train.columns]
    if missing:
        return {'error': f"Missing columns: {missing}"}

    # Check for NaN
    valid_mask = df_train[tvtp_cols].notna().all(axis=1)
    df_train_clean = df_train[valid_mask]

    if len(df_train_clean) < config.min_observations:
        return {'error': f"Not enough data: {len(df_train_clean)} < {config.min_observations}"}

    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"Features: {tvtp_cols}")
    print(f"Train size: {len(df_train_clean):,}")
    print(f"{'='*50}")

    try:
        # Fit
        nhhm = NHHM(config)
        nhhm.fit(df_train_clean, exog_tvtp_cols=tvtp_cols, verbose=True)

        # Predict
        result = nhhm.predict(df_train_clean, exog_tvtp_cols=tvtp_cols)

        # Compute metrics
        # Note: result arrays may be shorter due to outlier removal in fit
        n_result = len(result.signal)
        fwd_ret = df_train_clean['close'].shift(-config.forward_horizon) / df_train_clean['close'] - 1

        # Align sizes - use only the last n_result rows
        fwd_ret_aligned = fwd_ret.iloc[-n_result:].values
        valid_mask = ~np.isnan(fwd_ret_aligned) & (result.signal != 0)

        if valid_mask.sum() < 10:
            return {'error': 'Too few signals'}

        strategy_ret = result.signal[valid_mask] * fwd_ret_aligned[valid_mask]

        metrics = {
            'name': name,
            'n_train': len(df_train_clean),
            'n_signals': int(valid_mask.sum()),
            'hit_rate': float((strategy_ret > 0).mean()),
            'mean_return': float(strategy_ret.mean() * 100),  # in %
            'sharpe': float(strategy_ret.mean() / (strategy_ret.std() + 1e-10) * np.sqrt(252 * 12 / config.forward_horizon)),
            'bull_pct': float((result.signal == 1).mean()),
            'bear_pct': float((result.signal == -1).mean()),
        }

        print(f"\nResults:")
        print(f"  Hit rate: {metrics['hit_rate']:.2%}")
        print(f"  Sharpe:   {metrics['sharpe']:.2f}")
        print(f"  Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull, {metrics['bear_pct']:.1%} bear)")

        return metrics

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    print("="*60)
    print("NHHM Fourier Integration Test")
    print("="*60)

    # Load data
    print("\nLoading BTC data...")
    df = load_btc_data()
    print(f"Loaded {len(df):,} rows ({df.index[0]} to {df.index[-1]})")

    # Config
    config = NHHMConfig(
        n_regimes=2,
        forward_horizon=12,  # 24h
        min_observations=500
    )

    results = []

    # ========================================
    # Test 1: With Halving Cycle (BEST FEATURE)
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: With Halving Cycle (NEW - Best Feature)")
    print("="*60)

    df_halving = build_nhhm_features(df, include_fourier=False, include_funding=False, include_halving=True)
    halving_cols = get_recommended_tvtp_cols(include_fourier=False, include_funding=False, include_halving=True)
    df_halving = df_halving.dropna(subset=halving_cols)

    result1 = evaluate_nhhm(df_halving, halving_cols, config, name="Halving Cycle")
    results.append(result1)

    # ========================================
    # Test 2: Halving + Fourier
    # ========================================
    print("\n" + "="*60)
    print("TEST 2: Halving + Fourier")
    print("="*60)

    df_halv_four = build_nhhm_features(df, include_fourier=True, include_funding=False, include_halving=True)

    if 'LFP_signal' in df_halv_four.columns:
        halv_four_cols = ['halving_direction', 'LFP_signal', 'momentum_12', 'vol_ratio']
        df_halv_four = df_halv_four.dropna(subset=halv_four_cols)
        result2 = evaluate_nhhm(df_halv_four, halv_four_cols, config, name="Halving + Fourier")
        results.append(result2)
    else:
        print("WARNING: Fourier features not computed.")
        results.append({'error': 'Fourier features missing'})

    # ========================================
    # Test 3: Fourier + Funding (2019+ only)
    # ========================================
    print("\n" + "="*60)
    print("TEST 3: Fourier + Funding (2019+ only)")
    print("="*60)

    # Pre-load funding data
    try:
        from src.funding_rate import load_or_fetch_funding_rate
        funding_df = load_or_fetch_funding_rate(verbose=True)
        print(f"Loaded {len(funding_df)} funding records")
    except Exception as e:
        print(f"Could not load funding: {e}")
        funding_df = None

    if funding_df is not None and len(funding_df) > 0:
        df_funding = build_nhhm_features(df, include_fourier=True, include_funding=True, funding_df=funding_df)

        # Filter to 2019-10+ (funding started Sept 2019)
        df_funding_2019 = df_funding[df_funding.index >= '2019-10-01']

        # Check for funding columns
        funding_col = 'funding_zscore'
        if funding_col in df_funding_2019.columns:
            funding_cols = get_recommended_tvtp_cols(include_fourier=True, include_funding=True)
            print(f"Using columns: {funding_cols}")

            # Check for NaN
            df_funding_2019 = df_funding_2019.dropna(subset=funding_cols)
            print(f"Data after dropna: {len(df_funding_2019)}")

            if len(df_funding_2019) > config.min_observations:
                result3 = evaluate_nhhm(df_funding_2019, funding_cols, config, name="Fourier + Funding (2019+)")
                results.append(result3)
            else:
                print(f"Not enough data after filter: {len(df_funding_2019)}")
                results.append({'error': 'Not enough 2019+ data', 'name': 'Fourier + Funding'})
        else:
            print(f"Funding column '{funding_col}' not found")
            print(f"Available columns: {[c for c in df_funding_2019.columns if 'fund' in c.lower()]}")
            results.append({'error': 'No funding features', 'name': 'Fourier + Funding'})
    else:
        print("Funding data not available")
        results.append({'error': 'No funding data', 'name': 'Fourier + Funding'})

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Model':<30} {'Hit Rate':>10} {'Sharpe':>10} {'Signals':>10}")
    print("-"*60)

    for r in results:
        if 'error' in r:
            print(f"{r.get('name', 'Error'):<30} {'ERROR':>10} {r['error'][:20]:>10}")
        else:
            print(f"{r['name']:<30} {r['hit_rate']:>10.2%} {r['sharpe']:>10.2f} {r['n_signals']:>10,}")

    # Compare
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) >= 2:
        baseline = next((r for r in valid_results if 'Baseline' in r['name']), None)
        fourier = next((r for r in valid_results if 'Fourier' in r['name'] and 'Funding' not in r['name']), None)

        if baseline and fourier:
            hr_diff = fourier['hit_rate'] - baseline['hit_rate']
            sh_diff = fourier['sharpe'] - baseline['sharpe']
            print(f"\nFourier improvement:")
            print(f"  Hit rate: {hr_diff:+.2%}")
            print(f"  Sharpe:   {sh_diff:+.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
