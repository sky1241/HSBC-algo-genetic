#!/usr/bin/env python3
"""
Test d'une approche simplifiee pour la prediction directionnelle.

Au lieu d'utiliser le modele Markov (qui detecte les regimes, pas la direction),
on utilise directement des signaux directionnels :
- Momentum (tendance)
- Funding rate (sentiment)
- LFP comme filtre (trending vs choppy)

C'est l'approche recommandee par la Deep Research.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

def load_data():
    """Load BTC data and funding."""
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    # Load funding
    try:
        from src.funding_rate import load_or_fetch_funding_rate, resample_funding_to_h2
        funding_df = load_or_fetch_funding_rate(verbose=False)
        df['funding_rate'] = resample_funding_to_h2(funding_df, df)
    except:
        df['funding_rate'] = np.nan

    return df


def compute_directional_features(df):
    """Compute features for directional prediction."""
    out = df.copy()

    # Momentum (directional)
    out['momentum_12'] = out['close'].pct_change(12)  # 24h
    out['momentum_24'] = out['close'].pct_change(24)  # 48h

    # Funding signal (mean-revert: high funding = crowded long = bearish)
    out['funding_zscore'] = (
        (out['funding_rate'] - out['funding_rate'].rolling(36).mean()) /
        (out['funding_rate'].rolling(36).std() + 1e-10)
    )

    # LFP (from Fourier)
    try:
        from src.features_fourier import compute_fourier_features, FourierConfig
        config = FourierConfig(nperseg_grid=(128, 256))
        fourier = compute_fourier_features(out, config)
        out['LFP_ratio'] = fourier['LFP_ratio']
    except:
        out['LFP_ratio'] = 0.5

    return out


def generate_signals(df, momentum_window=12, funding_weight=0.5, funding_mode='trend'):
    """
    Generate directional signals.

    Args:
        funding_mode: 'trend' = follow funding direction, 'contrarian' = fade extreme funding

    Signal = sign(momentum) * confidence
    """
    signals = np.zeros(len(df))

    momentum = df['momentum_12'].values
    funding = df.get('funding_zscore', pd.Series(0, index=df.index)).fillna(0).values
    lfp = df.get('LFP_ratio', pd.Series(0.5, index=df.index)).fillna(0.5).values

    for i in range(len(df)):
        if np.isnan(momentum[i]):
            continue

        # Base signal from momentum
        signal = np.sign(momentum[i])

        # Funding adjustment
        if not np.isnan(funding[i]) and funding_weight > 0:
            if funding_mode == 'trend':
                # Trend-following: positive funding = bullish sentiment = long
                # Use funding as a CONFIRMATION signal
                funding_signal = np.sign(funding[i])
                if signal == funding_signal:
                    # Momentum and funding agree -> stronger signal
                    signal *= 1.2
                else:
                    # Disagreement -> weaker signal
                    signal *= 0.7
            else:
                # Contrarian: extreme funding = reversal coming
                if abs(funding[i]) > 2:  # Extreme
                    signal *= -0.5  # Fade the crowd

        # LFP filter: reduce signal in choppy markets
        if lfp[i] < 0.3:
            signal *= 0.5

        signals[i] = np.sign(signal) if abs(signal) > 0.3 else 0

    return signals


def evaluate_signals(df, signals, forward_horizon=12):
    """Evaluate signal performance."""
    fwd_ret = df['close'].shift(-forward_horizon) / df['close'] - 1

    valid = ~fwd_ret.isna() & (signals != 0)
    if valid.sum() < 10:
        return {'error': 'Too few signals'}

    strat_ret = signals[valid] * fwd_ret.values[valid]

    return {
        'n_signals': int(valid.sum()),
        'hit_rate': float((strat_ret > 0).mean()),
        'mean_return': float(strat_ret.mean() * 100),
        'sharpe': float(strat_ret.mean() / (strat_ret.std() + 1e-10) * np.sqrt(252 * 12 / forward_horizon)),
        'bull_pct': float((signals[valid.values] == 1).mean()),
        'bear_pct': float((signals[valid.values] == -1).mean()),
    }


def main():
    print("="*60)
    print("Simple Directional Signal Test")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows")

    # Compute features
    print("Computing features...")
    df = compute_directional_features(df)

    # ========================================
    # Test 1: Full dataset (no funding)
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: Momentum only (2011-2025)")
    print("="*60)

    df_clean = df.dropna(subset=['momentum_12'])
    signals = generate_signals(df_clean, funding_weight=0)

    metrics = evaluate_signals(df_clean, signals)
    print(f"Hit rate: {metrics['hit_rate']:.2%}")
    print(f"Sharpe:   {metrics['sharpe']:.2f}")
    print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull, {metrics['bear_pct']:.1%} bear)")

    # ========================================
    # Test 2: Funding trend-following (2019+)
    # ========================================
    print("\n" + "="*60)
    print("TEST 2: Momentum + Funding TREND-FOLLOWING (2019+)")
    print("="*60)

    df_2019 = df[df.index >= '2019-10-01'].copy()
    df_2019 = df_2019.dropna(subset=['momentum_12', 'funding_zscore'])

    if len(df_2019) > 100:
        signals = generate_signals(df_2019, funding_weight=0.5, funding_mode='trend')
        metrics = evaluate_signals(df_2019, signals)
        print(f"Hit rate: {metrics['hit_rate']:.2%}")
        print(f"Sharpe:   {metrics['sharpe']:.2f}")
        print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull, {metrics['bear_pct']:.1%} bear)")
    else:
        print("Not enough data")

    # ========================================
    # Test 3: Funding contrarian (2019+)
    # ========================================
    print("\n" + "="*60)
    print("TEST 3: Momentum + Funding CONTRARIAN (2019+)")
    print("="*60)

    if len(df_2019) > 100:
        signals = generate_signals(df_2019, funding_weight=0.5, funding_mode='contrarian')
        metrics = evaluate_signals(df_2019, signals)
        print(f"Hit rate: {metrics['hit_rate']:.2%}")
        print(f"Sharpe:   {metrics['sharpe']:.2f}")
        print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull, {metrics['bear_pct']:.1%} bear)")

    # ========================================
    # Test 4: With LFP filter
    # ========================================
    print("\n" + "="*60)
    print("TEST 4: Momentum + Funding + LFP filter (2019+)")
    print("="*60)

    if len(df_2019) > 100:
        # Only trade when LFP > 0.4 (trending)
        df_trending = df_2019[df_2019['LFP_ratio'] > 0.4]

        if len(df_trending) > 100:
            signals = generate_signals(df_trending, funding_weight=0.5, funding_mode='trend')
            metrics = evaluate_signals(df_trending, signals)
            print(f"Hit rate: {metrics['hit_rate']:.2%}")
            print(f"Sharpe:   {metrics['sharpe']:.2f}")
            print(f"Signals:  {metrics['n_signals']:,} ({metrics['bull_pct']:.1%} bull, {metrics['bear_pct']:.1%} bear)")
            print(f"(Only trading when LFP > 0.4 = trending market)")
        else:
            print("Not enough trending periods")

    print("\nDone!")


if __name__ == '__main__':
    main()
