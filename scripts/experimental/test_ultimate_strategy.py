#!/usr/bin/env python3
"""
ULTIMATE STRATEGY: Combine all best signals.

1. Halving cycle (bullish phases: 0-18 mois post-halving)
2. Drawdown zone (near ATH = strong long)
3. Momentum confirmation

The idea:
- Long when: bullish phase AND near ATH AND momentum positive
- Avoid when: bear phase OR deep drawdown
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

BULLISH_PHASES = [0, 1, 2]  # 0-18 months post-halving


def add_all_features(df):
    """Add all features."""
    out = df.copy()

    # Momentum
    out['momentum_12'] = out['close'].pct_change(12)

    # Drawdown
    out['ath'] = out['close'].expanding().max()
    out['drawdown'] = (out['close'] - out['ath']) / out['ath']

    # Halving phase
    halving_phase = []
    for ts in out.index:
        ts = pd.Timestamp(ts)
        last_h = None
        for h in HALVINGS:
            if h <= ts:
                last_h = h
        if last_h is None:
            halving_phase.append(-1)
            continue
        days = (ts - last_h).days
        if days < 180:
            phase = 0
        elif days < 365:
            phase = 1
        elif days < 540:
            phase = 2
        elif days < 730:
            phase = 3
        elif days < 1095:
            phase = 4
        else:
            phase = 5
        halving_phase.append(phase)

    out['halving_phase'] = halving_phase

    return out


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
    print("ULTIMATE STRATEGY: Halving + Drawdown + Momentum")
    print("="*70)

    # Load
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = add_all_features(df)
    df_clean = df.dropna(subset=['momentum_12'])
    print(f"Loaded {len(df_clean)} rows")

    results = []

    # ========================================
    # Strategy 1: Momentum only (baseline)
    # ========================================
    print("\n" + "="*70)
    print("BASELINE: Momentum Only")
    print("="*70)
    signals = np.sign(df_clean['momentum_12'].values)
    m = evaluate(df_clean, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    results.append(('Momentum only', m))

    # ========================================
    # Strategy 2: Near ATH only (DD < 10%)
    # ========================================
    print("\n" + "="*70)
    print("NEAR ATH: Long when DD < 10%")
    print("="*70)
    df_near = df_clean[df_clean['drawdown'] > -0.10]
    signals = np.sign(df_near['momentum_12'].values)
    m = evaluate(df_near, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    results.append(('Near ATH (<10% DD)', m))

    # ========================================
    # Strategy 3: Bullish phases only
    # ========================================
    print("\n" + "="*70)
    print("BULLISH PHASES: 0-18 months post-halving")
    print("="*70)
    df_bull = df_clean[df_clean['halving_phase'].isin(BULLISH_PHASES)]
    signals = np.sign(df_bull['momentum_12'].values)
    m = evaluate(df_bull, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
    results.append(('Bullish phases', m))

    # ========================================
    # Strategy 4: Bullish phases + Near ATH
    # ========================================
    print("\n" + "="*70)
    print("COMBO 1: Bullish phases + Near ATH")
    print("="*70)
    df_combo1 = df_clean[
        (df_clean['halving_phase'].isin(BULLISH_PHASES)) &
        (df_clean['drawdown'] > -0.10)
    ]
    if len(df_combo1) > 100:
        signals = np.sign(df_combo1['momentum_12'].values)
        m = evaluate(df_combo1, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(('Bullish + Near ATH', m))

    # ========================================
    # Strategy 5: Bullish phases + DD < 20%
    # ========================================
    print("\n" + "="*70)
    print("COMBO 2: Bullish phases + DD < 20%")
    print("="*70)
    df_combo2 = df_clean[
        (df_clean['halving_phase'].isin(BULLISH_PHASES)) &
        (df_clean['drawdown'] > -0.20)
    ]
    if len(df_combo2) > 100:
        signals = np.sign(df_combo2['momentum_12'].values)
        m = evaluate(df_combo2, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(('Bullish + DD<20%', m))

    # ========================================
    # Strategy 6: Any phase + Near ATH + positive momentum
    # ========================================
    print("\n" + "="*70)
    print("COMBO 3: Near ATH + Positive Momentum (any phase)")
    print("="*70)
    df_combo3 = df_clean[
        (df_clean['drawdown'] > -0.10) &
        (df_clean['momentum_12'] > 0)
    ]
    if len(df_combo3) > 100:
        signals = np.ones(len(df_combo3))  # Always long
        m = evaluate(df_combo3, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(('Near ATH + Mom+', m))

    # ========================================
    # Strategy 7: ULTIMATE - Bullish + Near ATH + Momentum+
    # ========================================
    print("\n" + "="*70)
    print("ULTIMATE: Bullish phase + Near ATH + Positive Momentum")
    print("="*70)
    df_ultimate = df_clean[
        (df_clean['halving_phase'].isin(BULLISH_PHASES)) &
        (df_clean['drawdown'] > -0.10) &
        (df_clean['momentum_12'] > 0)
    ]
    if len(df_ultimate) > 100:
        signals = np.ones(len(df_ultimate))  # Always long
        m = evaluate(df_ultimate, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")
        results.append(('ULTIMATE', m))
    else:
        print(f"Not enough data: {len(df_ultimate)}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY - SORTED BY SHARPE")
    print("="*70)

    results.sort(key=lambda x: x[1]['sharpe'], reverse=True)
    print(f"\n{'Strategy':<25} {'Hit Rate':>10} {'Sharpe':>10} {'Signals':>10}")
    print("-"*55)
    for name, m in results:
        print(f"{name:<25} {m['hr']:>10.2%} {m['sharpe']:>10.2f} {m['n']:>10,}")

    # Best vs baseline
    baseline = next(m for n, m in results if n == 'Momentum only')
    best = results[0]
    print(f"\nBest strategy: {best[0]}")
    print(f"  HR improvement:     {best[1]['hr'] - baseline['hr']:+.2%}")
    print(f"  Sharpe improvement: {best[1]['sharpe'] - baseline['sharpe']:+.2f}")


if __name__ == '__main__':
    main()
