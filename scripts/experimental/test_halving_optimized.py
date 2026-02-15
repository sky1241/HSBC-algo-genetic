#!/usr/bin/env python3
"""
Test des combinaisons optimisees avec le cycle halving.

On teste:
1. Halving seul (baseline: 54% HR, 1.14 Sharpe)
2. Halving + Funding (2019+)
3. Halving filtre par phase (only bullish phases)
4. Halving + LFP filter (only trending)
5. Combinaison optimale
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Import halving features
HALVINGS = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
    pd.Timestamp('2028-04-01'),
]

PHASES = {
    'accumulation': (0, 180, 0.8),
    'early_bull': (180, 365, 1.0),
    'parabolic': (365, 540, 0.6),
    'distribution': (540, 730, -0.5),
    'early_bear': (730, 1095, -0.7),
    'late_bear': (1095, 1460, 0.2),
}


def add_features(df):
    """Add all features."""
    out = df.copy()

    # Halving
    halving_dir = []
    halving_phase = []
    for ts in out.index:
        ts = pd.Timestamp(ts)
        last_h = None
        for h in HALVINGS:
            if h <= ts:
                last_h = h
        if last_h is None:
            halving_dir.append(0)
            halving_phase.append(-1)
            continue
        days = (ts - last_h).days
        phase_idx = -1
        direction = 0
        for i, (name, (start, end, dir_val)) in enumerate(PHASES.items()):
            if start <= days < end:
                phase_idx = i
                direction = dir_val
                break
        halving_dir.append(direction)
        halving_phase.append(phase_idx)

    out['halving_direction'] = halving_dir
    out['halving_phase'] = halving_phase

    # Momentum
    out['momentum_12'] = out['close'].pct_change(12)
    out['momentum_24'] = out['close'].pct_change(24)

    # LFP
    try:
        from src.features_fourier import compute_fourier_features, FourierConfig
        config = FourierConfig(nperseg_grid=(128, 256))
        fourier = compute_fourier_features(out, config)
        out['LFP_ratio'] = fourier['LFP_ratio']
    except:
        out['LFP_ratio'] = 0.5

    # Funding
    try:
        from src.funding_rate import load_or_fetch_funding_rate, resample_funding_to_h2
        funding_df = load_or_fetch_funding_rate(verbose=False)
        out['funding_rate'] = resample_funding_to_h2(funding_df, out)
        out['funding_zscore'] = (
            (out['funding_rate'] - out['funding_rate'].rolling(36).mean()) /
            (out['funding_rate'].rolling(36).std() + 1e-10)
        )
    except:
        out['funding_rate'] = np.nan
        out['funding_zscore'] = np.nan

    return out


def generate_signal(row, use_momentum=True, use_funding=False, use_lfp=False):
    """Generate signal for a single row."""
    base = row['halving_direction']

    if base == 0:
        return 0

    signal = base

    # Momentum confirmation
    if use_momentum and not np.isnan(row['momentum_12']):
        mom_sign = np.sign(row['momentum_12'])
        if mom_sign == np.sign(base):
            signal *= 1.2  # Confirmation
        else:
            signal *= 0.7  # Disagreement

    # Funding adjustment (contrarian for extremes)
    if use_funding and not np.isnan(row.get('funding_zscore', np.nan)):
        fz = row['funding_zscore']
        if abs(fz) > 2:  # Extreme
            # Fade the crowd
            signal *= 0.5 if np.sign(fz) == np.sign(signal) else 1.5

    # LFP filter
    if use_lfp:
        lfp = row.get('LFP_ratio', 0.5)
        if lfp < 0.3:  # Choppy
            signal *= 0.5

    return np.sign(signal) if abs(signal) > 0.3 else 0


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
    print("HALVING CYCLE - OPTIMIZATION TESTS")
    print("="*70)

    # Load
    print("\nLoading data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = add_features(df)
    print(f"Loaded {len(df)} rows")

    results = []

    # ========================================
    # Test 1: Halving + Momentum (baseline)
    # ========================================
    print("\n" + "="*70)
    print("TEST 1: Halving + Momentum (baseline)")
    print("="*70)

    df_clean = df.dropna(subset=['halving_direction', 'momentum_12'])
    signals = df_clean.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=False, use_lfp=False), axis=1).values
    m = evaluate(df_clean, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
    results.append(('Halving+Mom', m))

    # ========================================
    # Test 2: Halving + Momentum + LFP filter
    # ========================================
    print("\n" + "="*70)
    print("TEST 2: Halving + Momentum + LFP filter")
    print("="*70)

    df_lfp = df_clean[df_clean['LFP_ratio'] > 0.4]  # Only trending
    signals = df_lfp.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=False, use_lfp=True), axis=1).values
    m = evaluate(df_lfp, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
    print("(Only trading when LFP > 0.4)")
    results.append(('Halving+Mom+LFP', m))

    # ========================================
    # Test 3: Only bullish phases (0-18 months post-halving)
    # ========================================
    print("\n" + "="*70)
    print("TEST 3: Only Bullish Phases (accumulation, early_bull, parabolic)")
    print("="*70)

    df_bull = df_clean[df_clean['halving_phase'].isin([0, 1, 2])]  # Bullish phases
    signals = df_bull.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=False, use_lfp=False), axis=1).values
    m = evaluate(df_bull, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
    print("(Only trading in phases 0-18 months post-halving)")
    results.append(('Bullish phases only', m))

    # ========================================
    # Test 4: Halving + Funding (2019+)
    # ========================================
    print("\n" + "="*70)
    print("TEST 4: Halving + Momentum + Funding (2019+)")
    print("="*70)

    df_2019 = df[df.index >= '2019-10-01'].dropna(subset=['halving_direction', 'momentum_12', 'funding_zscore'])
    if len(df_2019) > 100:
        signals = df_2019.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=True, use_lfp=False), axis=1).values
        m = evaluate(df_2019, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
        results.append(('Halving+Mom+Funding', m))
    else:
        print("Not enough data")

    # ========================================
    # Test 5: Bullish phases + LFP filter
    # ========================================
    print("\n" + "="*70)
    print("TEST 5: Bullish Phases + LFP filter (OPTIMAL?)")
    print("="*70)

    df_opt = df_clean[(df_clean['halving_phase'].isin([0, 1, 2])) & (df_clean['LFP_ratio'] > 0.4)]
    if len(df_opt) > 100:
        signals = df_opt.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=False, use_lfp=True), axis=1).values
        m = evaluate(df_opt, signals)
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
        print("(Bullish phases + trending market)")
        results.append(('Bullish+LFP', m))

    # ========================================
    # Test 6: Avoid bearish phases entirely
    # ========================================
    print("\n" + "="*70)
    print("TEST 6: Avoid Bearish Phases (no trading in distribution/bear)")
    print("="*70)

    df_nobear = df_clean[~df_clean['halving_phase'].isin([3, 4])]  # Exclude distribution, early_bear
    signals = df_nobear.apply(lambda r: generate_signal(r, use_momentum=True, use_funding=False, use_lfp=False), axis=1).values
    m = evaluate(df_nobear, signals)
    print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")
    print("(No trading 18-36 months post-halving)")
    results.append(('Avoid bear phases', m))

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

    best = results[0]
    print(f"\nBEST: {best[0]} with Sharpe {best[1]['sharpe']:.2f}")


if __name__ == '__main__':
    main()
