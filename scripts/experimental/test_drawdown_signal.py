#!/usr/bin/env python3
"""
Strategie basee sur le drawdown depuis l'ATH.

Pattern:
- Nouveau ATH = BULL (continuation)
- Drawdown < 20% = BULL (normal volatility)
- Drawdown 20-40% = NEUTRAL (correction)
- Drawdown > 40% = BEAR (crash)

C'est simple mais ca capture le vrai pattern:
- Long tant qu'on fait des nouveaux ATH
- Short quand on perd > 20% depuis le top
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def compute_drawdown(prices):
    """Compute drawdown from running max."""
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max
    return drawdown, running_max


def add_drawdown_features(df, lookback_days=365):
    """Add drawdown-based features."""
    out = df.copy()

    # Running max (all-time high)
    out['ath'] = out['close'].expanding().max()
    out['drawdown'] = (out['close'] - out['ath']) / out['ath']

    # Rolling max (local high over last N days)
    lookback_bars = lookback_days * 12  # H2 bars
    out['local_max'] = out['close'].rolling(lookback_bars, min_periods=1).max()
    out['local_drawdown'] = (out['close'] - out['local_max']) / out['local_max']

    # Is making new ATH?
    out['new_ath'] = (out['close'] >= out['ath'].shift(1)).astype(int)

    # Days since last ATH
    out['ath_date'] = out.index.where(out['new_ath'] == 1)
    out['ath_date'] = out['ath_date'].ffill()
    out['days_since_ath'] = (out.index - out['ath_date']).dt.days

    # Signal based on drawdown
    # Long: drawdown < 20% (still in uptrend)
    # Neutral: drawdown 20-35% (correction, wait)
    # Short: drawdown > 35% (bear market)
    out['dd_signal'] = 0.0
    out.loc[out['drawdown'] > -0.20, 'dd_signal'] = 1.0   # Long
    out.loc[(out['drawdown'] <= -0.20) & (out['drawdown'] > -0.35), 'dd_signal'] = 0.0  # Neutral
    out.loc[out['drawdown'] <= -0.35, 'dd_signal'] = -1.0  # Short

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
        'mean_ret': float(strat_ret.mean() * 100),
    }


def main():
    print("="*70)
    print("DRAWDOWN-BASED STRATEGY TEST")
    print("="*70)

    # Load
    print("\nLoading data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df['momentum_12'] = df['close'].pct_change(12)
    print(f"Loaded {len(df)} rows")

    # Add features
    print("\nAdding features...")
    df = add_drawdown_features(df)

    # Current state
    last = df.iloc[-1]
    print(f"\nCurrent state:")
    print(f"  Price: ${last['close']:,.0f}")
    print(f"  ATH: ${last['ath']:,.0f}")
    print(f"  Drawdown: {last['drawdown']:.1%}")
    print(f"  Days since ATH: {last['days_since_ath']:.0f}")
    print(f"  Signal: {last['dd_signal']:.0f}")

    # Distribution
    print("\nDrawdown distribution:")
    print(f"  < 20% (bull):    {(df['drawdown'] > -0.20).mean():.1%}")
    print(f"  20-35% (neutral):{((df['drawdown'] <= -0.20) & (df['drawdown'] > -0.35)).mean():.1%}")
    print(f"  > 35% (bear):    {(df['drawdown'] <= -0.35).mean():.1%}")

    # ========================================
    # Test 1: Drawdown signal only
    # ========================================
    print("\n" + "="*70)
    print("TEST 1: Drawdown Signal (long if DD < 20%, short if DD > 35%)")
    print("="*70)

    df_clean = df.dropna(subset=['dd_signal', 'momentum_12'])
    signals = df_clean['dd_signal'].values
    m = evaluate(df_clean, signals)
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")

    # ========================================
    # Test 2: Drawdown + Momentum
    # ========================================
    print("\n" + "="*70)
    print("TEST 2: Drawdown + Momentum confirmation")
    print("="*70)

    def combined_signal(row):
        dd_sig = row['dd_signal']
        if dd_sig == 0:
            return 0
        mom = row['momentum_12']
        if np.isnan(mom):
            return dd_sig
        # Confirm with momentum
        if np.sign(mom) == np.sign(dd_sig):
            return dd_sig * 1.2
        else:
            return dd_sig * 0.5
        return np.sign(dd_sig) if abs(dd_sig) > 0.3 else 0

    signals = df_clean.apply(lambda r: np.sign(combined_signal(r)) if abs(combined_signal(r)) > 0.3 else 0, axis=1).values
    m = evaluate(df_clean, signals)
    if m:
        print(f"HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}, Bull: {m['bull_pct']:.1%}")

    # ========================================
    # Test 3: Performance by drawdown zone
    # ========================================
    print("\n" + "="*70)
    print("TEST 3: Performance by Drawdown Zone")
    print("="*70)

    zones = [
        ('Near ATH (0-10%)', df_clean['drawdown'] > -0.10),
        ('Small DD (10-20%)', (df_clean['drawdown'] <= -0.10) & (df_clean['drawdown'] > -0.20)),
        ('Correction (20-35%)', (df_clean['drawdown'] <= -0.20) & (df_clean['drawdown'] > -0.35)),
        ('Bear (35-50%)', (df_clean['drawdown'] <= -0.35) & (df_clean['drawdown'] > -0.50)),
        ('Deep Bear (>50%)', df_clean['drawdown'] <= -0.50),
    ]

    for name, mask in zones:
        df_zone = df_clean[mask]
        if len(df_zone) < 100:
            print(f"{name:>20}: Not enough data ({len(df_zone)})")
            continue

        # Long signal in this zone
        signals = np.ones(len(df_zone))  # Always long
        m = evaluate(df_zone, signals)
        if m:
            print(f"{name:>20}: HR={m['hr']:.1%}, Sharpe={m['sharpe']:+.2f}, N={m['n']:,}")

    # ========================================
    # Test 4: Optimal thresholds
    # ========================================
    print("\n" + "="*70)
    print("TEST 4: Finding Optimal Thresholds")
    print("="*70)

    best_sharpe = -999
    best_params = None

    for long_thresh in [-0.10, -0.15, -0.20, -0.25]:
        for short_thresh in [-0.30, -0.35, -0.40, -0.50]:
            if short_thresh >= long_thresh:
                continue

            signals = np.zeros(len(df_clean))
            signals[df_clean['drawdown'].values > long_thresh] = 1
            signals[df_clean['drawdown'].values <= short_thresh] = -1

            m = evaluate(df_clean, signals)
            if m and m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_params = (long_thresh, short_thresh, m)

    if best_params:
        lt, st, m = best_params
        print(f"Best: Long if DD > {lt:.0%}, Short if DD < {st:.0%}")
        print(f"       HR: {m['hr']:.2%}, Sharpe: {m['sharpe']:.2f}, N: {m['n']:,}")

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Momentum only
    signals_mom = np.sign(df_clean['momentum_12'].values)
    m_mom = evaluate(df_clean, signals_mom)

    # Drawdown
    signals_dd = df_clean['dd_signal'].values
    m_dd = evaluate(df_clean, signals_dd)

    print(f"Momentum only:  HR={m_mom['hr']:.2%}, Sharpe={m_mom['sharpe']:.2f}")
    print(f"Drawdown:       HR={m_dd['hr']:.2%}, Sharpe={m_dd['sharpe']:.2f}")
    print(f"\nImprovement: HR {m_dd['hr'] - m_mom['hr']:+.2%}, Sharpe {m_dd['sharpe'] - m_mom['sharpe']:+.2f}")


if __name__ == '__main__':
    main()
