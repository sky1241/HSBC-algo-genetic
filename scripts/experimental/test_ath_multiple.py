#!/usr/bin/env python3
"""
Strategie basee sur le multiple de l'ATH precedent.

Pattern observe:
- Apres chaque halving, le prix monte jusqu'a x2-x3 de l'ATH precedent
- Puis CRASH
- Le multiple diminue a chaque cycle (36x -> 18x -> 3.5x -> 1.5x)

Strategie:
- LONG: tant qu'on n'a pas atteint le "target multiple" de l'ATH precedent
- NEUTRAL/SHORT: quand on approche ou depasse le target
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ATH de chaque cycle - utiliser le VRAI peak pour calibrer
# Le pattern: chaque cycle peak = previous_peak * diminishing_multiple
CYCLE_DATA = [
    # (cycle_start, previous_peak, expected_peak_multiple)
    # On utilise le peak REEL comme reference pour le short signal
    ('2012-11-28', 30, 30),        # 2013: peak ~$1100, x36 du bottom
    ('2016-07-09', 1100, 15),      # 2017: peak ~$20000, x18 du previous ATH
    ('2020-05-11', 20000, 3.5),    # 2021: peak ~$69000, x3.5 du previous ATH
    ('2024-04-20', 69000, 2.0),    # 2024: target ~$138000, x2 du previous ATH (diminishing)
]


def get_cycle_info(timestamp, price):
    """
    Get cycle info for a given timestamp and price.

    Returns:
        - cycle_idx: which cycle we're in
        - prev_ath: ATH of previous cycle
        - target_multiple: expected peak multiple
        - current_multiple: current price / prev_ath
        - pct_to_target: how close we are to target (0-1, >1 = exceeded)
    """
    ts = pd.Timestamp(timestamp)

    cycle_idx = -1
    prev_ath = None
    target_mult = None

    for i, (start, ath, target, peak) in enumerate(CYCLE_DATA):
        if ts >= pd.Timestamp(start):
            cycle_idx = i
            prev_ath = ath
            target_mult = target

    if prev_ath is None or prev_ath == 0:
        return None

    current_mult = price / prev_ath
    pct_to_target = current_mult / target_mult if target_mult > 0 else 0

    return {
        'cycle_idx': cycle_idx,
        'prev_ath': prev_ath,
        'target_multiple': target_mult,
        'current_multiple': current_mult,
        'pct_to_target': pct_to_target,
    }


def add_ath_features(df):
    """Add ATH-based features."""
    out = df.copy()

    current_mult = []
    pct_to_target = []
    signal = []

    for ts, row in out.iterrows():
        info = get_cycle_info(ts, row['close'])
        if info is None:
            current_mult.append(np.nan)
            pct_to_target.append(np.nan)
            signal.append(0)
            continue

        current_mult.append(info['current_multiple'])
        pct_to_target.append(info['pct_to_target'])

        # Signal logic:
        # - pct < 0.5: early bull, STRONG LONG
        # - pct 0.5-0.8: mid bull, LONG
        # - pct 0.8-1.0: late bull, CAUTIOUS LONG
        # - pct > 1.0: exceeded target, SHORT/NEUTRAL
        pct = info['pct_to_target']
        if pct < 0.5:
            sig = 1.0  # Strong long
        elif pct < 0.8:
            sig = 0.7  # Long
        elif pct < 1.0:
            sig = 0.3  # Cautious
        elif pct < 1.5:
            sig = -0.5  # Short
        else:
            sig = -1.0  # Strong short

        signal.append(sig)

    out['ath_multiple'] = current_mult
    out['pct_to_target'] = pct_to_target
    out['ath_signal'] = signal

    return out


def generate_signal(row, use_momentum=True):
    """Generate combined signal."""
    base = row['ath_signal']

    if base == 0 or np.isnan(base):
        return 0

    signal = base

    # Momentum confirmation
    if use_momentum and not np.isnan(row.get('momentum_12', np.nan)):
        mom = row['momentum_12']
        if np.sign(mom) == np.sign(base):
            signal *= 1.2
        else:
            signal *= 0.7

    return np.sign(signal) if abs(signal) > 0.2 else 0


def evaluate(df, signals, forward_horizon=12):
    """Evaluate."""
    fwd_ret = df['close'].shift(-forward_horizon) / df['close'] - 1
    valid = ~fwd_ret.isna() & (signals != 0)
    if valid.sum() < 10:
        return None
    strat_ret = signals[valid] * fwd_ret.values[valid]
    return {
        'n': int(valid.sum()),
        'hr': float((strat_ret > 0).mean()),
        'sharpe': float(strat_ret.mean() / (strat_ret.std() + 1e-10) * np.sqrt(252 * 12 / forward_horizon)),
        'bull_pct': float((signals[valid.values] == 1).mean()),
    }


def main():
    print("="*70)
    print("ATH MULTIPLE STRATEGY TEST")
    print("="*70)

    # Load
    print("\nLoading data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df)} rows")

    # Add features
    print("\nAdding features...")
    df['momentum_12'] = df['close'].pct_change(12)
    df = add_ath_features(df)

    # Show current cycle info
    last_row = df.iloc[-1]
    print(f"\nCurrent state:")
    print(f"  Price: ${last_row['close']:,.0f}")
    print(f"  ATH multiple: {last_row['ath_multiple']:.2f}x")
    print(f"  % to target: {last_row['pct_to_target']:.1%}")
    print(f"  Signal: {last_row['ath_signal']:.2f}")

    # Show stats by pct_to_target
    print("\nDistribution by % to target:")
    bins = [0, 0.5, 0.8, 1.0, 1.5, 100]
    labels = ['<50%', '50-80%', '80-100%', '100-150%', '>150%']
    df['pct_bin'] = pd.cut(df['pct_to_target'], bins=bins, labels=labels)
    for label in labels:
        count = (df['pct_bin'] == label).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    # ========================================
    # Test 1: ATH signal only
    # ========================================
    print("\n" + "="*70)
    print("TEST 1: ATH Multiple Signal")
    print("="*70)

    df_clean = df.dropna(subset=['ath_signal', 'momentum_12'])
    signals = df_clean.apply(lambda r: generate_signal(r, use_momentum=False), axis=1).values
    m = evaluate(df_clean, signals)
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")

    # ========================================
    # Test 2: ATH + Momentum
    # ========================================
    print("\n" + "="*70)
    print("TEST 2: ATH + Momentum")
    print("="*70)

    signals = df_clean.apply(lambda r: generate_signal(r, use_momentum=True), axis=1).values
    m = evaluate(df_clean, signals)
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")

    # ========================================
    # Test 3: Performance by zone
    # ========================================
    print("\n" + "="*70)
    print("TEST 3: Performance by % to Target Zone")
    print("="*70)

    for label in labels:
        df_zone = df_clean[df_clean['pct_bin'] == label]
        if len(df_zone) < 100:
            continue
        signals = df_zone.apply(lambda r: generate_signal(r, use_momentum=True), axis=1).values
        m = evaluate(df_zone, signals)
        if m:
            print(f"{label:>10}: HR={m['hr']:.1%}, Sharpe={m['sharpe']:+.2f}, N={m['n']:,}")

    # ========================================
    # Test 4: Compare with halving strategy
    # ========================================
    print("\n" + "="*70)
    print("TEST 4: Comparison")
    print("="*70)

    # ATH strategy
    signals_ath = df_clean.apply(lambda r: generate_signal(r, use_momentum=True), axis=1).values
    m_ath = evaluate(df_clean, signals_ath)

    # Simple momentum
    signals_mom = np.sign(df_clean['momentum_12'].values)
    m_mom = evaluate(df_clean, signals_mom)

    print(f"Momentum only:  HR={m_mom['hr']:.2%}, Sharpe={m_mom['sharpe']:.2f}")
    print(f"ATH Multiple:   HR={m_ath['hr']:.2%}, Sharpe={m_ath['sharpe']:.2f}")
    print(f"\nImprovement: HR {m_ath['hr'] - m_mom['hr']:+.2%}, Sharpe {m_ath['sharpe'] - m_mom['sharpe']:+.2f}")


if __name__ == '__main__':
    main()
