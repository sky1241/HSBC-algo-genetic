#!/usr/bin/env python3
"""
Validation du NHHM v2 sur des bear markets connus.

Tests:
  1. Mars 2020 (crash COVID)  → NHHM detecte bear?
  2. Mai 2021 (top BTC ~64k)  → NHHM detecte distribution/bear?
  3. Nov 2022 (crash FTX)     → NHHM detecte bear?
  4. Jan 2023 (recovery)      → NHHM detecte bull?

Criteres:
  - Hit rate > 53% sur forward returns
  - Bear markets correctement identifies (label=0 pendant crashes)
  - Bull markets correctement identifies (label=1 pendant recoveries)

Usage:
  py -3 scripts/experimental/test_nhhm_v2_validation.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from src.regime_nhhm_v2 import NHHMv2, NHHMv2Config, build_features

# Bear market test periods
TEST_PERIODS = [
    {
        'name': 'COVID Crash (Mars 2020)',
        'start': '2020-02-15',
        'end': '2020-04-15',
        'expected': 'bear',
        'description': 'BTC passe de ~10k a ~4k en quelques jours',
    },
    {
        'name': 'Top BTC 64k (Mai 2021)',
        'start': '2021-04-15',
        'end': '2021-07-15',
        'expected': 'bear',
        'description': 'BTC passe de ~64k a ~30k, chute de 50%+',
    },
    {
        'name': 'FTX Crash (Nov 2022)',
        'start': '2022-10-15',
        'end': '2022-12-31',
        'expected': 'bear',
        'description': 'BTC passe de ~21k a ~16k apres faillite FTX',
    },
    {
        'name': 'Recovery 2023',
        'start': '2023-01-01',
        'end': '2023-03-31',
        'expected': 'bull',
        'description': 'BTC remonte de ~16k a ~28k',
    },
    {
        'name': 'Bull Post-Halving 2024',
        'start': '2024-10-01',
        'end': '2024-12-31',
        'expected': 'bull',
        'description': 'BTC monte vers nouveaux ATH',
    },
]


def main():
    print("=" * 70)
    print("NHHM v2 VALIDATION - Bear Markets Connus")
    print("=" * 70)

    # Load data
    csv_path = 'data/BTC_FUSED_2h.csv'
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"  {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # Build features
    print("\nBuilding features...")
    features = build_features(df)
    print(f"  Features: {list(features.columns)}")
    print(f"  NaN counts:\n{features.isna().sum()}")

    # Fit rolling NHHM
    print("\n" + "=" * 70)
    print("FITTING ROLLING HMM (window=10000, refit=500)")
    print("=" * 70)

    # Use market-based features (no halving to avoid domination)
    feature_cols = ['return_1', 'momentum_6', 'momentum_12', 'vol_ratio',
                    'rsi_centered', 'dist_ma20']

    config = NHHMv2Config(
        n_states=2,
        window_size=10000,
        refit_every=500,
        min_window=2000,
        n_iter=100,
        n_random_starts=5,
        feature_cols=feature_cols,
    )
    model = NHHMv2(config)
    result = model.fit_predict_rolling(df, features, verbose=True)

    # Global stats
    valid = result['label'].notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        print("\nERROR: No valid predictions!")
        return

    n_bull = int((result.loc[valid, 'label'] == 1).sum())
    n_bear = int((result.loc[valid, 'label'] == 0).sum())
    print(f"\n{'=' * 70}")
    print(f"GLOBAL STATS")
    print(f"{'=' * 70}")
    print(f"  Valid predictions: {n_valid}/{len(result)}")
    print(f"  Bull: {n_bull} ({100*n_bull/n_valid:.1f}%)")
    print(f"  Bear: {n_bear} ({100*n_bear/n_valid:.1f}%)")

    # Hit rate on forward returns
    print(f"\n{'=' * 70}")
    print(f"HIT RATE (direction vs forward 24h return)")
    print(f"{'=' * 70}")

    close = df['close']
    fwd_ret = close.shift(-12) / close - 1  # 12 bars = 24h forward return

    # Align
    common_idx = result.index.intersection(fwd_ret.dropna().index)
    labels = result.loc[common_idx, 'label'].dropna()
    returns = fwd_ret.loc[labels.index]

    # Bull predictions: label=1, did price go up?
    bull_mask = labels == 1
    if bull_mask.sum() > 0:
        bull_correct = (returns[bull_mask] > 0).mean()
        print(f"  Bull predictions: {bull_mask.sum()}, correct: {100*bull_correct:.1f}%")

    # Bear predictions: label=0, did price go down?
    bear_mask = labels == 0
    if bear_mask.sum() > 0:
        bear_correct = (returns[bear_mask] < 0).mean()
        print(f"  Bear predictions: {bear_mask.sum()}, correct: {100*bear_correct:.1f}%")

    # Overall
    signal = labels.map({1: 1, 0: -1})
    strat_ret = signal * returns
    overall_hit = (strat_ret > 0).mean()
    print(f"  Overall hit rate: {100*overall_hit:.1f}%")

    # Sharpe
    if strat_ret.std() > 0:
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 12)
        print(f"  Strategy Sharpe: {sharpe:.2f}")

    # Test each period
    print(f"\n{'=' * 70}")
    print(f"BEAR MARKET TESTS")
    print(f"{'=' * 70}")

    results_summary = []

    for test in TEST_PERIODS:
        print(f"\n--- {test['name']} ---")
        print(f"  {test['description']}")

        start = pd.Timestamp(test['start'])
        end = pd.Timestamp(test['end'])
        expected = test['expected']

        # Filter result for this period
        mask = (result.index >= start) & (result.index <= end)
        period_data = result.loc[mask]
        period_valid = period_data['label'].notna()

        if period_valid.sum() == 0:
            print(f"  [WARN] No predictions for this period!")
            results_summary.append({
                'period': test['name'],
                'expected': expected,
                'detected': 'N/A',
                'match': False,
                'pct_bull': np.nan,
                'pct_bear': np.nan,
                'mean_p_bull': np.nan,
            })
            continue

        n_period = int(period_valid.sum())
        n_bull_p = int((period_data.loc[period_valid, 'label'] == 1).sum())
        n_bear_p = int((period_data.loc[period_valid, 'label'] == 0).sum())
        pct_bull = 100 * n_bull_p / n_period
        pct_bear = 100 * n_bear_p / n_period
        mean_p_bull = period_data.loc[period_valid, 'p_bull'].mean()

        # Majority vote
        detected = 'bull' if n_bull_p > n_bear_p else 'bear'
        match = detected == expected

        icon = "OK" if match else "FAIL"
        print(f"  Attendu: {expected.upper()}")
        print(f"  Detecte: {detected.upper()} [{icon}]")
        print(f"  Bull: {n_bull_p} ({pct_bull:.1f}%), Bear: {n_bear_p} ({pct_bear:.1f}%)")
        print(f"  Mean P(bull): {mean_p_bull:.3f}")

        # Price change during period
        price_mask = (close.index >= start) & (close.index <= end)
        period_close = close.loc[price_mask]
        if len(period_close) > 0:
            price_change = (period_close.iloc[-1] / period_close.iloc[0] - 1) * 100
            print(f"  Price change: {price_change:+.1f}%")

        results_summary.append({
            'period': test['name'],
            'expected': expected,
            'detected': detected,
            'match': match,
            'pct_bull': pct_bull,
            'pct_bear': pct_bear,
            'mean_p_bull': mean_p_bull,
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")

    summary_df = pd.DataFrame(results_summary)
    n_tests = len(summary_df)
    n_correct = int(summary_df['match'].sum())

    print(f"\n{'Period':<35} {'Expected':>8} {'Detected':>8} {'Match':>6} {'P(bull)':>8}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        icon = "OK" if row['match'] else "FAIL"
        p_bull_str = f"{row['mean_p_bull']:.3f}" if not np.isnan(row['mean_p_bull']) else "N/A"
        print(f"{row['period']:<35} {row['expected']:>8} {row['detected']:>8} {icon:>6} {p_bull_str:>8}")

    print(f"\nScore: {n_correct}/{n_tests} ({100*n_correct/n_tests:.0f}%)")

    if n_correct >= 3:
        print(f"\n[PASS] NHHM v2 passe la validation! Pret pour generer les labels.")
    elif n_correct >= 2:
        print(f"\n[WARN] Resultats mitiges. A analyser avant de continuer.")
    else:
        print(f"\n[FAIL] NHHM v2 echoue la validation. Revoir les features/parametres.")

    # Save detailed results
    output_path = 'outputs/nhhm_v2_validation.csv'
    os.makedirs('outputs', exist_ok=True)
    result_export = result[['timestamp', 'p_bull', 'p_bear', 'label', 'regime']].dropna(subset=['label'])
    result_export['label'] = result_export['label'].astype(int)
    result_export.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to {output_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
