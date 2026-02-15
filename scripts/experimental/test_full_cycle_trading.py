#!/usr/bin/env python3
"""
FULL CYCLE TRADING - Trade TOUT LE TEMPS avec la bonne direction.

Le cycle de 4 ans:
0-6 mois:   LONG  (post-halving accumulation)
6-12 mois:  LONG  (early bull)
12-18 mois: LONG  (parabolic)
18-24 mois: SHORT (distribution/top)
24-36 mois: SHORT (bear market)
36-48 mois: LONG  (late bear accumulation)

On trade TOUJOURS, on change juste la direction.
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

# Direction pour chaque phase (jours depuis halving)
CYCLE_DIRECTION = [
    (0, 180, 1, 'accumulation'),       # 0-6 mois: LONG
    (180, 365, 1, 'early_bull'),       # 6-12 mois: LONG
    (365, 540, 1, 'parabolic'),        # 12-18 mois: LONG
    (540, 730, -1, 'distribution'),    # 18-24 mois: SHORT
    (730, 1095, -1, 'bear'),           # 24-36 mois: SHORT
    (1095, 1460, 1, 'late_bear'),      # 36-48 mois: LONG (accumulation)
]


def get_cycle_direction(timestamp):
    """Get direction based on cycle position."""
    ts = pd.Timestamp(timestamp)

    # Find last halving
    last_h = None
    for h in HALVINGS:
        if h <= ts:
            last_h = h

    if last_h is None:
        return 0, 'pre-halving'

    days = (ts - last_h).days

    for start, end, direction, name in CYCLE_DIRECTION:
        if start <= days < end:
            return direction, name

    return 0, 'unknown'


def add_features(df):
    """Add all features."""
    out = df.copy()

    # Cycle direction
    directions = []
    phases = []
    for ts in out.index:
        d, p = get_cycle_direction(ts)
        directions.append(d)
        phases.append(p)

    out['cycle_direction'] = directions
    out['cycle_phase'] = phases

    # Momentum
    out['momentum_12'] = out['close'].pct_change(12)
    out['momentum_24'] = out['close'].pct_change(24)

    # Drawdown (for refinement)
    out['ath'] = out['close'].expanding().max()
    out['drawdown'] = (out['close'] - out['ath']) / out['ath']

    return out


def evaluate(df, signals, forward_horizon=12, name=""):
    """Evaluate strategy."""
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
    print("FULL CYCLE TRADING - Trade TOUT LE TEMPS")
    print("="*70)

    # Load
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = add_features(df)
    df = df.dropna(subset=['momentum_12'])
    print(f"Loaded {len(df)} rows")

    # Phase distribution
    print("\nPhase distribution:")
    for phase in df['cycle_phase'].unique():
        if phase == 'unknown':
            continue
        count = (df['cycle_phase'] == phase).sum()
        pct = count / len(df) * 100
        direction = df[df['cycle_phase'] == phase]['cycle_direction'].iloc[0]
        dir_str = "LONG" if direction == 1 else "SHORT"
        print(f"  {phase:15}: {count:>6,} ({pct:>5.1f}%) -> {dir_str}")

    results = []

    # ========================================
    # Strategy 1: Momentum only (baseline)
    # ========================================
    print("\n" + "="*70)
    print("BASELINE: Momentum Only")
    print("="*70)

    signals = np.sign(df['momentum_12'].values)
    m = evaluate(df, signals, name="Momentum only")
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    results.append(m)

    # ========================================
    # Strategy 2: Cycle direction only
    # ========================================
    print("\n" + "="*70)
    print("CYCLE DIRECTION: Trade based on halving cycle")
    print("="*70)

    signals = df['cycle_direction'].values.astype(float)
    m = evaluate(df, signals, name="Cycle direction")
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    print(f"  Long trades:  {m['long_pct']:.1%} of total, HR: {m['long_hr']:.1%}")
    print(f"  Short trades: {1-m['long_pct']:.1%} of total, HR: {m['short_hr']:.1%}")
    results.append(m)

    # ========================================
    # Strategy 3: Cycle + Momentum confirmation
    # ========================================
    print("\n" + "="*70)
    print("CYCLE + MOMENTUM: Boost when aligned")
    print("="*70)

    def combined_signal(row):
        cycle = row['cycle_direction']
        mom = np.sign(row['momentum_12']) if not np.isnan(row['momentum_12']) else 0

        if cycle == 0:
            return mom  # No cycle info, use momentum

        # If cycle and momentum agree, stronger signal
        if cycle == mom:
            return cycle * 1.5
        else:
            # Cycle direction but momentum disagrees - still follow cycle but weaker
            return cycle * 0.7

    signals = df.apply(lambda r: np.sign(combined_signal(r)), axis=1).values
    m = evaluate(df, signals, name="Cycle + Momentum")
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    print(f"  Long trades:  {m['long_pct']:.1%}, HR: {m['long_hr']:.1%}")
    print(f"  Short trades: {1-m['long_pct']:.1%}, HR: {m['short_hr']:.1%}")
    results.append(m)

    # ========================================
    # Strategy 4: Cycle + Drawdown refinement
    # ========================================
    print("\n" + "="*70)
    print("CYCLE + DRAWDOWN: Adjust based on distance from ATH")
    print("="*70)

    def refined_signal(row):
        cycle = row['cycle_direction']
        dd = row['drawdown']

        if cycle == 0:
            return 0

        # In LONG phases: stronger signal near ATH
        if cycle == 1:
            if dd > -0.10:  # Near ATH
                return 1.5
            elif dd > -0.30:  # Small correction
                return 1.0
            else:  # Deep drawdown in bull phase = buy the dip
                return 1.2

        # In SHORT phases: stronger signal when far from ATH
        else:
            if dd < -0.30:  # Already crashed, maybe bounce
                return -0.5
            elif dd < -0.15:  # In correction
                return -1.0
            else:  # Near ATH in bear phase = short!
                return -1.5

    signals = df.apply(lambda r: np.sign(refined_signal(r)), axis=1).values
    m = evaluate(df, signals, name="Cycle + Drawdown")
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    print(f"  Long trades:  {m['long_pct']:.1%}, HR: {m['long_hr']:.1%}")
    print(f"  Short trades: {1-m['long_pct']:.1%}, HR: {m['short_hr']:.1%}")
    results.append(m)

    # ========================================
    # Performance by phase
    # ========================================
    print("\n" + "="*70)
    print("PERFORMANCE BY PHASE")
    print("="*70)

    for phase in ['accumulation', 'early_bull', 'parabolic', 'distribution', 'bear', 'late_bear']:
        df_phase = df[df['cycle_phase'] == phase]
        if len(df_phase) < 100:
            continue

        # Use cycle direction for this phase
        signals = df_phase['cycle_direction'].values.astype(float)
        m = evaluate(df_phase, signals, name=phase)
        if m:
            print(f"{phase:15}: HR={m['hr']:.1%}, Sharpe={m['sharpe']:+.2f}, N={m['n']:,}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n{'Strategy':<25} {'Hit Rate':>10} {'Sharpe':>10} {'Signals':>10}")
    print("-"*55)
    for m in results:
        print(f"{m['name']:<25} {m['hr']:>10.2%} {m['sharpe']:>10.2f} {m['n']:>10,}")


if __name__ == '__main__':
    main()
