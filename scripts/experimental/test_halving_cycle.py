#!/usr/bin/env python3
"""
Test de la strategie basee sur les cycles de halving BTC.

Cycles connus:
- Halving tous les ~4 ans (210,000 blocs)
- Post-halving: stagnation (3-6 mois) → explosion (6-12 mois) → pic → correction
- Le pic arrive generalement 12-18 mois apres le halving
- Rendements decroissants a chaque cycle

Halvings historiques:
- 28 Nov 2012 (bloc 210,000)
- 9 Jul 2016 (bloc 420,000)
- 11 May 2020 (bloc 630,000)
- 20 Apr 2024 (bloc 840,000)
- ~2028 (bloc 1,050,000)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# Dates des halvings
HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
    pd.Timestamp('2028-04-01'),  # Estimation
]

# Phases du cycle (en jours apres halving)
CYCLE_PHASES = {
    'accumulation': (0, 180),      # 0-6 mois: stagnation/accumulation
    'early_bull': (180, 365),      # 6-12 mois: debut du bull run
    'parabolic': (365, 540),       # 12-18 mois: phase parabolique
    'distribution': (540, 730),    # 18-24 mois: distribution/pic
    'early_bear': (730, 1095),     # 24-36 mois: bear market
    'late_bear': (1095, 1460),     # 36-48 mois: fin du bear, accumulation
}

def get_halving_features(timestamp):
    """
    Calcule les features liees au cycle de halving.

    Returns:
        - days_since_halving: jours depuis le dernier halving
        - cycle_phase: phase du cycle (0-5)
        - cycle_progress: progression dans le cycle de 4 ans (0-1)
        - expected_direction: direction attendue basee sur la phase
    """
    ts = pd.Timestamp(timestamp)

    # Trouver le dernier halving
    last_halving = None
    next_halving = None
    for i, h in enumerate(HALVINGS):
        if h <= ts:
            last_halving = h
            if i + 1 < len(HALVINGS):
                next_halving = HALVINGS[i + 1]
        else:
            if last_halving is None:
                # Avant le premier halving
                return None, None, None, None
            break

    if last_halving is None:
        return None, None, None, None

    days_since = (ts - last_halving).days

    # Cycle progress (0-1)
    if next_halving:
        cycle_length = (next_halving - last_halving).days
        cycle_progress = min(days_since / cycle_length, 1.0)
    else:
        cycle_progress = min(days_since / 1460, 1.0)  # ~4 ans

    # Phase du cycle
    phase = 'unknown'
    phase_idx = -1
    expected_dir = 0

    for i, (phase_name, (start, end)) in enumerate(CYCLE_PHASES.items()):
        if start <= days_since < end:
            phase = phase_name
            phase_idx = i
            break

    # Direction attendue par phase
    phase_directions = {
        'accumulation': 0.3,    # Légèrement bullish (accumulation)
        'early_bull': 0.7,      # Bullish
        'parabolic': 1.0,       # Très bullish
        'distribution': -0.3,   # Début correction
        'early_bear': -0.7,     # Bearish
        'late_bear': 0.0,       # Neutre/accumulation
    }
    expected_dir = phase_directions.get(phase, 0)

    return days_since, phase_idx, cycle_progress, expected_dir


def add_halving_features(df):
    """Ajoute les features de cycle halving au DataFrame."""
    out = df.copy()

    days_since = []
    phase_idx = []
    cycle_progress = []
    expected_dir = []

    for ts in out.index:
        d, p, c, e = get_halving_features(ts)
        days_since.append(d)
        phase_idx.append(p)
        cycle_progress.append(c)
        expected_dir.append(e)

    out['halving_days'] = days_since
    out['halving_phase'] = phase_idx
    out['halving_progress'] = cycle_progress
    out['halving_direction'] = expected_dir

    return out


def test_halving_strategy(df, use_momentum_filter=True):
    """
    Strategie basee sur les cycles de halving.

    Signal = halving_direction * (1 + momentum_confirmation)
    """
    signals = np.zeros(len(df))

    halving_dir = df['halving_direction'].values
    momentum = df['close'].pct_change(12).values  # 24h momentum

    for i in range(len(df)):
        if pd.isna(halving_dir[i]) or halving_dir[i] == 0:
            continue

        base_signal = halving_dir[i]

        if use_momentum_filter and not np.isnan(momentum[i]):
            # Boost signal if momentum confirms
            if np.sign(momentum[i]) == np.sign(base_signal):
                base_signal *= 1.3
            else:
                base_signal *= 0.7

        signals[i] = np.sign(base_signal) if abs(base_signal) > 0.2 else 0

    return signals


def evaluate(df, signals, forward_horizon=12):
    """Evaluate performance."""
    fwd_ret = df['close'].shift(-forward_horizon) / df['close'] - 1

    valid = ~fwd_ret.isna() & (signals != 0)
    if valid.sum() < 10:
        return None

    strat_ret = signals[valid] * fwd_ret.values[valid]

    return {
        'n_signals': int(valid.sum()),
        'hit_rate': float((strat_ret > 0).mean()),
        'sharpe': float(strat_ret.mean() / (strat_ret.std() + 1e-10) * np.sqrt(252 * 12 / forward_horizon)),
        'mean_ret': float(strat_ret.mean() * 100),
        'bull_pct': float((signals[valid.values] == 1).mean()),
    }


def main():
    print("="*60)
    print("HALVING CYCLE STRATEGY TEST")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} rows ({df.index[0]} to {df.index[-1]})")

    # Add halving features
    print("\nAdding halving cycle features...")
    df = add_halving_features(df)

    # Show halving stats
    print("\nHalving phase distribution:")
    phase_names = list(CYCLE_PHASES.keys())
    for i, name in enumerate(phase_names):
        count = (df['halving_phase'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {name}: {count:,} bars ({pct:.1f}%)")

    # ========================================
    # Test 1: Halving direction only
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: Halving Direction Only")
    print("="*60)

    signals = test_halving_strategy(df, use_momentum_filter=False)
    metrics = evaluate(df, signals)

    if metrics:
        print(f"Hit rate: {metrics['hit_rate']:.2%}")
        print(f"Sharpe:   {metrics['sharpe']:.2f}")
        print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull)")

    # ========================================
    # Test 2: Halving + Momentum filter
    # ========================================
    print("\n" + "="*60)
    print("TEST 2: Halving + Momentum Confirmation")
    print("="*60)

    signals = test_halving_strategy(df, use_momentum_filter=True)
    metrics = evaluate(df, signals)

    if metrics:
        print(f"Hit rate: {metrics['hit_rate']:.2%}")
        print(f"Sharpe:   {metrics['sharpe']:.2f}")
        print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull)")

    # ========================================
    # Test 3: Performance by phase
    # ========================================
    print("\n" + "="*60)
    print("TEST 3: Performance by Halving Phase")
    print("="*60)

    for i, name in enumerate(phase_names):
        df_phase = df[df['halving_phase'] == i]
        if len(df_phase) < 100:
            continue

        signals = test_halving_strategy(df_phase, use_momentum_filter=True)
        metrics = evaluate(df_phase, signals)

        if metrics:
            print(f"{name:15}: HR={metrics['hit_rate']:.1%}, Sharpe={metrics['sharpe']:+.2f}, n={metrics['n_signals']:,}")

    # ========================================
    # Test 4: Compare with simple momentum
    # ========================================
    print("\n" + "="*60)
    print("TEST 4: Comparison with Simple Momentum")
    print("="*60)

    # Simple momentum strategy
    momentum = df['close'].pct_change(12)
    signals_mom = np.sign(momentum.values)
    signals_mom[np.isnan(signals_mom)] = 0

    metrics_mom = evaluate(df, signals_mom)
    print(f"Momentum only:     HR={metrics_mom['hit_rate']:.2%}, Sharpe={metrics_mom['sharpe']:.2f}")

    # Halving + momentum
    signals_halv = test_halving_strategy(df, use_momentum_filter=True)
    metrics_halv = evaluate(df, signals_halv)
    print(f"Halving + Mom:     HR={metrics_halv['hit_rate']:.2%}, Sharpe={metrics_halv['sharpe']:.2f}")

    # Difference
    print(f"\nImprovement: HR {metrics_halv['hit_rate'] - metrics_mom['hit_rate']:+.2%}, "
          f"Sharpe {metrics_halv['sharpe'] - metrics_mom['sharpe']:+.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
