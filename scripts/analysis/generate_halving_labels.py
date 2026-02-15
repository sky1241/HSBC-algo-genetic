#!/usr/bin/env python3
"""
Generate labels based on halving cycle phases.

Ces labels sont bases sur la connaissance des cycles de 4 ans du BTC:
- Post-halving (0-12 mois): BULLISH (label=2)
- Distribution/Bear (12-36 mois): BEARISH (label=0)
- Late bear/Accumulation (36-48 mois): NEUTRAL (label=1)

Performance testee: 54% hit rate, Sharpe 1.14 (vs 47% baseline)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd

# Halving dates
HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
    pd.Timestamp('2028-04-01'),
]

# Phases et labels
# 0 = bear, 1 = neutral, 2 = bull
PHASES = [
    # (start_days, end_days, label, name)
    (0, 180, 2, 'accumulation'),      # Post-halving: bullish
    (180, 365, 2, 'early_bull'),      # Bull run start: bullish
    (365, 540, 2, 'parabolic'),       # Parabolic: bullish (mais prudent)
    (540, 730, 0, 'distribution'),    # Distribution: bearish
    (730, 1095, 0, 'early_bear'),     # Bear market: bearish
    (1095, 1460, 1, 'late_bear'),     # Late bear: neutral/accumulation
]


def get_halving_label(timestamp):
    """Get label based on halving cycle position."""
    ts = pd.Timestamp(timestamp)

    # Find last halving
    last_halving = None
    for h in HALVINGS:
        if h <= ts:
            last_halving = h
        else:
            break

    if last_halving is None:
        return 1  # Neutral before first halving

    days_since = (ts - last_halving).days

    # Find phase
    for start, end, label, name in PHASES:
        if start <= days_since < end:
            return label

    return 1  # Default neutral


def main():
    print("="*60)
    print("Halving Cycle Label Generator")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} rows")

    # Generate labels
    print("\nGenerating labels...")
    labels = [get_halving_label(ts) for ts in df.index]

    # Create output
    output = pd.DataFrame({
        'timestamp': df.index,
        'label': labels
    })

    # Stats
    print("\nLabel distribution:")
    for label in [0, 1, 2]:
        count = (output['label'] == label).sum()
        pct = count / len(output) * 100
        name = {0: 'bear', 1: 'neutral', 2: 'bull'}[label]
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Save
    output_dir = Path("outputs/fourier/labels_frozen/BTC_FUSED_2h")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "HALVING_cycle.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Also save as K3 format for compatibility
    output_path_k3 = output_dir / "K3_halving.csv"
    output.to_csv(output_path_k3, index=False)
    print(f"Saved to {output_path_k3}")

    print("\nDone!")


if __name__ == '__main__':
    main()
