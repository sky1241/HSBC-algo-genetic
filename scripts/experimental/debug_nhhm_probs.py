#!/usr/bin/env python3
"""Debug NHHM probability distribution."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.regime_nhhm import NHHM, NHHMConfig, build_nhhm_features, get_recommended_tvtp_cols

# Load data
print("Loading data...")
df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
df = df.set_index('timestamp')

# Build features
print("Building features...")
df_feat = build_nhhm_features(df, include_fourier=True, include_funding=False)
tvtp_cols = get_recommended_tvtp_cols(include_fourier=True, include_funding=False)
df_feat = df_feat.dropna(subset=tvtp_cols)

# Fit
print("Fitting NHHM...")
config = NHHMConfig(n_regimes=2, forward_horizon=12)
nhhm = NHHM(config)
nhhm.fit(df_feat, exog_tvtp_cols=tvtp_cols, verbose=True)

# Get probabilities
result = nhhm.predict(df_feat, exog_tvtp_cols=tvtp_cols)

print("\n" + "="*50)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*50)

print(f"\nP(bull) statistics:")
print(f"  Min:    {result.p_bull.min():.3f}")
print(f"  Max:    {result.p_bull.max():.3f}")
print(f"  Mean:   {result.p_bull.mean():.3f}")
print(f"  Median: {np.median(result.p_bull):.3f}")
print(f"  Std:    {result.p_bull.std():.3f}")

print(f"\nP(bear) statistics:")
print(f"  Min:    {result.p_bear.min():.3f}")
print(f"  Max:    {result.p_bear.max():.3f}")
print(f"  Mean:   {result.p_bear.mean():.3f}")
print(f"  Median: {np.median(result.p_bear):.3f}")

# Distribution
print("\nP(bull) distribution:")
thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]
for t in thresholds:
    pct = (result.p_bull > t).mean() * 100
    print(f"  P(bull) > {t}: {pct:.1f}%")

print("\nP(bear) distribution:")
for t in thresholds:
    pct = (result.p_bear > t).mean() * 100
    print(f"  P(bear) > {t}: {pct:.1f}%")

# Calculate optimal threshold
print("\n" + "="*50)
print("TESTING DIFFERENT THRESHOLDS")
print("="*50)

# Align returns
fwd_ret = df_feat['close'].shift(-12) / df_feat['close'] - 1
n = len(result.signal)
fwd_ret_aligned = fwd_ret.iloc[-n:].values

for threshold in [0.5, 0.52, 0.55, 0.6]:
    signal = np.zeros(len(result.p_bull))
    signal[result.p_bull > threshold] = 1
    signal[result.p_bear > threshold] = -1

    valid = ~np.isnan(fwd_ret_aligned) & (signal != 0)
    if valid.sum() < 10:
        print(f"Threshold {threshold}: Not enough signals")
        continue

    strat_ret = signal[valid] * fwd_ret_aligned[valid]
    hit_rate = (strat_ret > 0).mean()
    sharpe = strat_ret.mean() / (strat_ret.std() + 1e-10) * np.sqrt(252 * 12 / 12)

    bull_pct = (signal == 1).mean() * 100
    bear_pct = (signal == -1).mean() * 100

    print(f"Threshold {threshold}: HR={hit_rate:.2%}, Sharpe={sharpe:.2f}, Bull={bull_pct:.1f}%, Bear={bear_pct:.1f}%")
