#!/usr/bin/env python3
"""
Test des variantes du cycle trading pour optimiser les phases SHORT.

Variantes:
1. Original: SHORT pendant distribution/bear
2. Cash: CASH pendant distribution/bear (pas de short)
3. Selective Short: SHORT seulement si momentum negatif
4. Inverse Bear: LONG meme en bear (BTC monte toujours long terme)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

# Phases avec directions
PHASES = [
    (0, 180, 'accumulation'),
    (180, 365, 'early_bull'),
    (365, 540, 'parabolic'),
    (540, 730, 'distribution'),
    (730, 1095, 'bear'),
    (1095, 1460, 'late_bear'),
]


def get_phase(timestamp):
    """Get phase name for timestamp."""
    ts = pd.Timestamp(timestamp)
    last_h = None
    for h in HALVINGS:
        if h <= ts:
            last_h = h
    if last_h is None:
        return 'pre-halving'
    days = (ts - last_h).days
    for start, end, name in PHASES:
        if start <= days < end:
            return name
    return 'unknown'


def add_features(df):
    """Add features."""
    out = df.copy()
    out['phase'] = [get_phase(ts) for ts in out.index]
    out['momentum_12'] = out['close'].pct_change(12)
    out['ath'] = out['close'].expanding().max()
    out['drawdown'] = (out['close'] - out['ath']) / out['ath']
    return out


def evaluate(df, signals, forward_horizon=12, name=""):
    """Evaluate."""
    fwd_ret = df['close'].shift(-forward_horizon) / df['close'] - 1
    valid = ~fwd_ret.isna() & (signals != 0)
    if valid.sum() < 10:
        return None
    strat_ret = signals[valid] * fwd_ret.values[valid]
    long_mask = signals[valid.values] == 1
    short_mask = signals[valid.values] == -1
    return {
        'name': name,
        'n': int(valid.sum()),
        'hr': float((strat_ret > 0).mean()),
        'sharpe': float(strat_ret.mean() / (strat_ret.std() + 1e-10) * np.sqrt(252 * 12 / forward_horizon)),
        'long_pct': float(long_mask.mean()),
        'long_hr': float((strat_ret[long_mask] > 0).mean()) if long_mask.sum() > 0 else 0,
        'short_hr': float((strat_ret[short_mask] > 0).mean()) if short_mask.sum() > 0 else 0,
    }


def main():
    print("="*70)
    print("CYCLE TRADING - VARIANTES")
    print("="*70)

    # Load
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = add_features(df)
    df = df.dropna(subset=['momentum_12'])
    print(f"Loaded {len(df)} rows\n")

    results = []

    # ========================================
    # Variante 1: Original (SHORT en bear)
    # ========================================
    print("="*70)
    print("V1: ORIGINAL - SHORT pendant distribution/bear")
    print("="*70)

    ORIGINAL_DIR = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': -1, 'bear': -1, 'late_bear': 1,
        'pre-halving': 0, 'unknown': 0
    }
    signals = df['phase'].map(ORIGINAL_DIR).values.astype(float)
    m = evaluate(df, signals, name="V1: Original")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        print(f"  Long: {m['long_pct']:.1%} (HR {m['long_hr']:.1%}), Short: {1-m['long_pct']:.1%} (HR {m['short_hr']:.1%})")
        results.append(m)

    # ========================================
    # Variante 2: CASH pendant distribution/bear
    # ========================================
    print("\n" + "="*70)
    print("V2: CASH - Pas de trading pendant distribution/bear")
    print("="*70)

    CASH_DIR = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': 0, 'bear': 0, 'late_bear': 1,  # CASH instead of SHORT
        'pre-halving': 0, 'unknown': 0
    }
    signals = df['phase'].map(CASH_DIR).values.astype(float)
    m = evaluate(df, signals, name="V2: Cash in bear")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        print(f"  (Trading only during bullish phases)")
        results.append(m)

    # ========================================
    # Variante 3: SHORT seulement si momentum negatif
    # ========================================
    print("\n" + "="*70)
    print("V3: SELECTIVE SHORT - Short seulement si momentum < 0")
    print("="*70)

    def selective_short(row):
        phase = row['phase']
        mom = row['momentum_12']

        if phase in ['accumulation', 'early_bull', 'parabolic', 'late_bear']:
            return 1  # LONG

        if phase in ['distribution', 'bear']:
            # SHORT seulement si momentum negatif
            if mom < 0:
                return -1
            else:
                return 0  # Cash si momentum positif

        return 0

    signals = df.apply(selective_short, axis=1).values
    m = evaluate(df, signals, name="V3: Selective short")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        print(f"  Long: {m['long_pct']:.1%} (HR {m['long_hr']:.1%}), Short: {1-m['long_pct']:.1%} (HR {m['short_hr']:.1%})")
        results.append(m)

    # ========================================
    # Variante 4: LONG tout le temps (BTC bias)
    # ========================================
    print("\n" + "="*70)
    print("V4: ALWAYS LONG - BTC monte toujours long terme")
    print("="*70)

    ALWAYS_LONG = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': 1, 'bear': 1, 'late_bear': 1,  # LONG meme en bear
        'pre-halving': 1, 'unknown': 0
    }
    signals = df['phase'].map(ALWAYS_LONG).values.astype(float)
    m = evaluate(df, signals, name="V4: Always long")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(m)

    # ========================================
    # Variante 5: LONG sauf deep bear (>12 mois post-peak)
    # ========================================
    print("\n" + "="*70)
    print("V5: LONG sauf deep bear - Cash seulement en bear profond")
    print("="*70)

    LONG_EXCEPT_DEEP = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': 1, 'bear': 0, 'late_bear': 1,  # Cash seulement en bear
        'pre-halving': 0, 'unknown': 0
    }
    signals = df['phase'].map(LONG_EXCEPT_DEEP).values.astype(float)
    m = evaluate(df, signals, name="V5: Long except bear")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(m)

    # ========================================
    # Variante 6: Drawdown-based refinement
    # ========================================
    print("\n" + "="*70)
    print("V6: DRAWDOWN REFINEMENT - Short seulement si DD > 20% en bear")
    print("="*70)

    def drawdown_refined(row):
        phase = row['phase']
        dd = row['drawdown']

        if phase in ['accumulation', 'early_bull', 'parabolic', 'late_bear']:
            return 1

        if phase in ['distribution', 'bear']:
            # SHORT seulement si drawdown significatif
            if dd < -0.20:
                return -1  # Short en correction
            elif dd < -0.10:
                return 0  # Cash
            else:
                return 1  # Long si encore pres ATH

        return 0

    signals = df.apply(drawdown_refined, axis=1).values
    m = evaluate(df, signals, name="V6: DD refined")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        print(f"  Long: {m['long_pct']:.1%} (HR {m['long_hr']:.1%}), Short: {1-m['long_pct']:.1%} (HR {m['short_hr']:.1%})")
        results.append(m)

    # ========================================
    # Variante 7: OPTIMAL - Combine best elements
    # ========================================
    print("\n" + "="*70)
    print("V7: OPTIMAL - Long phases + selective short avec DD")
    print("="*70)

    def optimal_signal(row):
        phase = row['phase']
        dd = row['drawdown']
        mom = row['momentum_12']

        # Phases bullish = LONG
        if phase in ['accumulation', 'early_bull', 'late_bear']:
            return 1

        # Parabolic = LONG mais attention pres du top
        if phase == 'parabolic':
            if dd > -0.05:  # Tres pres ATH
                return 0.8 if mom > 0 else 0.5
            return 1

        # Distribution/Bear = decision basee sur DD + momentum
        if phase in ['distribution', 'bear']:
            if dd < -0.30 and mom < 0:
                return -1  # Short fort
            elif dd < -0.20:
                return -0.5 if mom < 0 else 0
            elif dd > -0.10:
                return 0.5  # Encore bullish
            return 0

        return 0

    signals = df.apply(lambda r: np.sign(optimal_signal(r)) if abs(optimal_signal(r)) > 0.3 else 0, axis=1).values
    m = evaluate(df, signals, name="V7: Optimal")
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        print(f"  Long: {m['long_pct']:.1%} (HR {m['long_hr']:.1%}), Short: {1-m['long_pct']:.1%} (HR {m['short_hr']:.1%})")
        results.append(m)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY - SORTED BY SHARPE")
    print("="*70)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n{'Strategy':<25} {'Hit Rate':>10} {'Sharpe':>10} {'Signals':>10}")
    print("-"*60)
    for m in results:
        print(f"{m['name']:<25} {m['hr']:>10.2%} {m['sharpe']:>10.2f} {m['n']:>10,}")

    best = results[0]
    print(f"\n*** BEST: {best['name']} ***")
    print(f"    Hit Rate: {best['hr']:.2%}")
    print(f"    Sharpe: {best['sharpe']:.2f}")
    print(f"    Trades: {best['n']:,}")


if __name__ == '__main__':
    main()
