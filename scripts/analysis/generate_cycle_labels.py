#!/usr/bin/env python3
"""
Generate labels based on halving cycle for WFA integration.

Variantes disponibles:
- V1: Original (SHORT en bear)
- V2: Cash en bear (meilleur Sharpe)
- V3: Selective short (compromis)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

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


def generate_v1_labels(df):
    """V1: Original - SHORT en distribution/bear."""
    DIR_MAP = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': -1, 'bear': -1, 'late_bear': 1,
        'pre-halving': 0, 'unknown': 0
    }
    return df['phase'].map(DIR_MAP)


def generate_v2_labels(df):
    """V2: Cash en bear - Meilleur Sharpe."""
    DIR_MAP = {
        'accumulation': 1, 'early_bull': 1, 'parabolic': 1,
        'distribution': 0, 'bear': 0, 'late_bear': 1,
        'pre-halving': 0, 'unknown': 0
    }
    return df['phase'].map(DIR_MAP)


def generate_v3_labels(df):
    """V3: Selective short - Short seulement si momentum negatif."""
    def signal(row):
        phase = row['phase']
        mom = row['momentum_12']

        if phase in ['accumulation', 'early_bull', 'parabolic', 'late_bear']:
            return 1
        if phase in ['distribution', 'bear']:
            if pd.notna(mom) and mom < 0:
                return -1
            return 0
        return 0

    return df.apply(signal, axis=1)


def main():
    print("="*70)
    print("GENERATE CYCLE LABELS FOR WFA")
    print("="*70)

    # Load data
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df['phase'] = [get_phase(ts) for ts in df.index]
    df['momentum_12'] = df['close'].pct_change(12)
    print(f"Loaded {len(df)} rows")

    # Generate all variants
    df['label_v1'] = generate_v1_labels(df)
    df['label_v2'] = generate_v2_labels(df)
    df['label_v3'] = generate_v3_labels(df)

    # Stats
    print("\n" + "="*70)
    print("LABEL STATISTICS")
    print("="*70)

    for v in ['v1', 'v2', 'v3']:
        col = f'label_{v}'
        total = len(df)
        long_pct = (df[col] == 1).sum() / total * 100
        short_pct = (df[col] == -1).sum() / total * 100
        cash_pct = (df[col] == 0).sum() / total * 100
        print(f"\n{v.upper()}:")
        print(f"  LONG:  {long_pct:>5.1f}%")
        print(f"  SHORT: {short_pct:>5.1f}%")
        print(f"  CASH:  {cash_pct:>5.1f}%")

    # Save labels
    print("\n" + "="*70)
    print("SAVING LABELS")
    print("="*70)

    # Format for WFA compatibility (like K5 labels)
    for variant, name, sharpe in [
        ('label_v1', 'CYCLE_original', 1.10),
        ('label_v2', 'CYCLE_cash_bear', 1.61),
        ('label_v3', 'CYCLE_selective', 1.28),
    ]:
        output = df[[variant]].copy()
        output.columns = ['label']
        output['timestamp'] = output.index

        # Map to regime names for compatibility
        regime_map = {1: 'bull', -1: 'bear', 0: 'neutral'}
        output['regime'] = output['label'].map(regime_map)

        # Save
        filename = f'data/{name}.csv'
        output[['timestamp', 'label', 'regime']].to_csv(filename, index=False)
        print(f"Saved {filename} (Sharpe ~{sharpe:.2f})")

    # Also save a combined file with all info
    combined = df[['close', 'phase', 'momentum_12', 'label_v1', 'label_v2', 'label_v3']].copy()
    combined.to_csv('data/CYCLE_labels_all.csv')
    print(f"\nSaved data/CYCLE_labels_all.csv (combined)")

    # Show recent labels
    print("\n" + "="*70)
    print("RECENT LABELS (last 10 days)")
    print("="*70)

    recent = df.tail(120)  # ~10 days in 2H bars
    print(f"\n{'Date':<20} {'Phase':<15} {'V1':>4} {'V2':>4} {'V3':>4}")
    print("-"*50)
    for i in range(0, len(recent), 12):  # Sample every day
        row = recent.iloc[i]
        ts = recent.index[i]
        print(f"{str(ts)[:19]:<20} {row['phase']:<15} {int(row['label_v1']):>4} {int(row['label_v2']):>4} {int(row['label_v3']):>4}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nRecommendation:")
    print("  - V2 (Cash in bear) = Best Sharpe (1.61) but only 58% time")
    print("  - V3 (Selective)    = Good compromise (1.28 Sharpe, 75% time)")
    print("  - V1 (Original)     = Trade most (91%) but lower Sharpe (1.10)")


if __name__ == '__main__':
    main()
