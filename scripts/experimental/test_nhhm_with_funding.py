#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test NHHM with Funding Rate features.

This test compares:
1. NHHM without funding rate (baseline)
2. NHHM with funding rate (should improve hit rate)

The funding rate is the key directional indicator that was missing
in the initial test. According to Deep Research recommendations,
adding funding should disambiguate bull vs bear momentum.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.regime_nhhm import NHHM, NHHMConfig, build_nhhm_features
from src.funding_rate import load_or_fetch_funding_rate


def compute_metrics(signal: np.ndarray, fwd_ret: np.ndarray, forward_horizon: int) -> dict:
    """Compute performance metrics for a signal."""
    valid_mask = ~np.isnan(fwd_ret)
    n_valid = min(len(signal), valid_mask.sum())

    signal_valid = signal[:n_valid]
    fwd_ret_valid = fwd_ret[:n_valid]

    # Strategy returns
    strategy_ret = signal_valid * fwd_ret_valid

    # Only count non-zero signals
    active_mask = signal_valid != 0
    n_active = active_mask.sum()

    if n_active == 0:
        return {"error": "No active signals"}

    active_ret = strategy_ret[active_mask]

    return {
        "n_signals": int(n_active),
        "hit_rate": float((active_ret > 0).mean()),
        "mean_return": float(active_ret.mean()),
        "std_return": float(active_ret.std()),
        "sharpe": float(active_ret.mean() / (active_ret.std() + 1e-10) * np.sqrt(252 * 12 / forward_horizon)),
        "long_pct": float((signal_valid == 1).mean()),
        "short_pct": float((signal_valid == -1).mean()),
    }


def main() -> int:
    print("=" * 70)
    print("NHHM Test: Baseline vs With Funding Rate")
    print("=" * 70)

    # Load OHLCV data
    csv_path = _ROOT / "data" / "BTC_FUSED_2h.csv"
    print(f"\n1. Loading OHLCV data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"   Loaded {len(df)} rows")

    # Load funding rate
    print("\n2. Loading funding rate data...")
    try:
        funding_df = load_or_fetch_funding_rate(verbose=True)
        has_funding = len(funding_df) > 0
    except Exception as e:
        print(f"   Could not load funding rate: {e}")
        has_funding = False
        funding_df = None

    # Config
    config = NHHMConfig(
        n_regimes=2,
        forward_horizon=12,  # 24h ahead
        switching_variance=True
    )

    # Split data
    train_ratio = 0.7
    n = len(df)
    split_idx = int(n * train_ratio)

    # ========================================
    # Test 1: Baseline (without funding)
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: NHHM Baseline (without funding rate)")
    print("=" * 70)

    # Build features without funding
    df_baseline = build_nhhm_features(df.copy(), include_funding=False)
    df_baseline = df_baseline.dropna()
    df_train_base = df_baseline.iloc[:int(len(df_baseline) * train_ratio)]

    feature_cols_base = ['momentum_12', 'vol_ratio', 'rsi_centered', 'dist_ma20']
    print(f"\n   Features: {feature_cols_base}")

    nhhm_base = NHHM(config)
    try:
        nhhm_base.fit(df_train_base, exog_tvtp_cols=feature_cols_base, verbose=True)
        result_base = nhhm_base.predict(df_train_base, exog_tvtp_cols=feature_cols_base)

        # Compute forward return
        fwd_ret_base = (df_train_base['close'].shift(-config.forward_horizon) / df_train_base['close'] - 1).values

        metrics_base = compute_metrics(result_base.signal, fwd_ret_base, config.forward_horizon)
        print(f"\n   Results (Baseline):")
        for k, v in metrics_base.items():
            if isinstance(v, float):
                print(f"   - {k}: {v:.4f}")
            else:
                print(f"   - {k}: {v}")
    except Exception as e:
        print(f"   ERROR: {e}")
        metrics_base = {"error": str(e)}

    # ========================================
    # Test 2: With Funding Rate
    # ========================================
    if has_funding:
        print("\n" + "=" * 70)
        print("TEST 2: NHHM with Funding Rate")
        print("=" * 70)

        # Build features with funding
        df_funding = build_nhhm_features(df.copy(), include_funding=True, funding_df=funding_df)
        df_funding = df_funding.dropna()
        df_train_fund = df_funding.iloc[:int(len(df_funding) * train_ratio)]

        # Check which funding features are available
        funding_cols = [c for c in df_train_fund.columns if 'funding' in c]
        print(f"\n   Available funding features: {funding_cols}")

        # Select key features
        feature_cols_fund = [
            'momentum_12',
            'vol_ratio',
            'rsi_centered',
            'dist_ma20',
        ]

        # Add funding features if available
        if 'funding_zscore' in df_train_fund.columns:
            feature_cols_fund.append('funding_zscore')
        if 'funding_polarity' in df_train_fund.columns:
            feature_cols_fund.append('funding_polarity')

        print(f"   Features: {feature_cols_fund}")

        nhhm_fund = NHHM(config)
        try:
            nhhm_fund.fit(df_train_fund, exog_tvtp_cols=feature_cols_fund, verbose=True)
            result_fund = nhhm_fund.predict(df_train_fund, exog_tvtp_cols=feature_cols_fund)

            # Compute forward return
            fwd_ret_fund = (df_train_fund['close'].shift(-config.forward_horizon) / df_train_fund['close'] - 1).values

            metrics_fund = compute_metrics(result_fund.signal, fwd_ret_fund, config.forward_horizon)
            print(f"\n   Results (With Funding):")
            for k, v in metrics_fund.items():
                if isinstance(v, float):
                    print(f"   - {k}: {v:.4f}")
                else:
                    print(f"   - {k}: {v}")
        except Exception as e:
            print(f"   ERROR: {e}")
            metrics_fund = {"error": str(e)}
    else:
        print("\n   SKIPPED: No funding rate data available")
        metrics_fund = {"error": "No data"}

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if "error" not in metrics_base and "error" not in metrics_fund:
        print(f"\n   {'Metric':<20} {'Baseline':<15} {'With Funding':<15} {'Diff':<10}")
        print("   " + "-" * 60)

        for key in ['hit_rate', 'sharpe', 'mean_return', 'n_signals']:
            base_val = metrics_base.get(key, 0)
            fund_val = metrics_fund.get(key, 0)
            diff = fund_val - base_val

            if isinstance(base_val, float):
                print(f"   {key:<20} {base_val:<15.4f} {fund_val:<15.4f} {diff:+.4f}")
            else:
                print(f"   {key:<20} {base_val:<15} {fund_val:<15} {diff:+}")

        # Verdict
        hit_diff = metrics_fund.get('hit_rate', 0) - metrics_base.get('hit_rate', 0)
        if hit_diff > 0.02:
            print(f"\n   [SUCCESS] Funding rate improved hit rate by {hit_diff*100:.1f}%!")
        elif hit_diff > 0:
            print(f"\n   [OK] Slight improvement (+{hit_diff*100:.1f}%), may need more tuning")
        else:
            print(f"\n   [WARN] No improvement. Consider:")
            print("         - Different funding features (zscore vs polarity)")
            print("         - Different forward horizon")
            print("         - More regimes (3 instead of 2)")
    else:
        print("\n   Could not compare due to errors")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
