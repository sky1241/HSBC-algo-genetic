#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NHHM labels to replace HMM K3/K5 labels.

Output format compatible with existing pipeline:
- timestamp, label
- label: 0=bear, 1=neutral, 2=bull (for K3) or binary 0/1

Usage:
    py -3 scripts/generate_nhhm_labels.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from src.regime_nhhm import (
    NHHM, NHHMConfig, build_nhhm_features, get_recommended_tvtp_cols
)


def generate_labels_rolling(
    df: pd.DataFrame,
    tvtp_cols: list,
    config: NHHMConfig,
    lookback_years: float = 3.0,
    step_months: int = 6,
    threshold_bull: float = 0.55,
    threshold_bear: float = 0.55,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate NHHM labels using rolling window training.

    This avoids lookahead bias by training only on past data.

    Args:
        df: DataFrame with features
        tvtp_cols: Columns for TVTP
        config: NHHM config
        lookback_years: Training window size
        step_months: Retrain every N months
        threshold_bull: P(bull) threshold for label=2
        threshold_bear: P(bear) threshold for label=0
        verbose: Print progress

    Returns:
        DataFrame with timestamp, label, p_bull, p_bear
    """
    df = df.copy()

    # Calculate step size in bars (H2 = 12 bars/day)
    bars_per_day = 12
    lookback_bars = int(lookback_years * 365 * bars_per_day)
    step_bars = int(step_months * 30 * bars_per_day)

    results = []
    n = len(df)

    if verbose:
        print(f"Rolling window training:")
        print(f"  Lookback: {lookback_years} years ({lookback_bars:,} bars)")
        print(f"  Step: {step_months} months ({step_bars:,} bars)")
        print(f"  Total: {n:,} bars")

    # Start from minimum lookback
    start_idx = lookback_bars

    current_model = None
    last_train_idx = 0

    for i in range(start_idx, n):
        # Retrain if needed
        need_retrain = (current_model is None) or (i - last_train_idx >= step_bars)

        if need_retrain:
            train_start = max(0, i - lookback_bars)
            train_end = i

            df_train = df.iloc[train_start:train_end].copy()

            # Drop NaN in TVTP columns
            valid_mask = df_train[tvtp_cols].notna().all(axis=1)
            df_train_clean = df_train[valid_mask]

            if len(df_train_clean) < config.min_observations:
                continue

            try:
                current_model = NHHM(config)
                current_model.fit(df_train_clean, exog_tvtp_cols=tvtp_cols, verbose=False)
                last_train_idx = i

                if verbose and i % (step_bars * 2) == 0:
                    print(f"  Trained at bar {i:,} ({df.index[i]})")

            except Exception as e:
                if verbose:
                    print(f"  Training failed at bar {i}: {e}")
                continue

        # Predict for current bar
        if current_model is not None:
            try:
                # Use filtered probabilities from training
                result = current_model.predict(
                    df.iloc[:i+1],
                    exog_tvtp_cols=tvtp_cols
                )

                p_bull = result.p_bull[-1] if len(result.p_bull) > 0 else 0.5
                p_bear = result.p_bear[-1] if len(result.p_bear) > 0 else 0.5

                # Determine label
                if p_bull >= threshold_bull:
                    label = 2  # Bull
                elif p_bear >= threshold_bear:
                    label = 0  # Bear
                else:
                    label = 1  # Neutral

                results.append({
                    'timestamp': df.index[i],
                    'label': label,
                    'p_bull': p_bull,
                    'p_bear': p_bear,
                })

            except Exception:
                pass

    if not results:
        return pd.DataFrame(columns=['timestamp', 'label', 'p_bull', 'p_bear'])

    return pd.DataFrame(results)


def generate_labels_simple(
    df: pd.DataFrame,
    tvtp_cols: list,
    config: NHHMConfig,
    threshold_bull: float = 0.55,
    threshold_bear: float = 0.55,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate NHHM labels using full-sample training (simpler, faster).

    WARNING: This has lookahead bias. Use only for exploration.

    Args:
        df: DataFrame with features
        tvtp_cols: Columns for TVTP
        config: NHHM config
        threshold_bull: P(bull) threshold for label=2
        threshold_bear: P(bear) threshold for label=0
        verbose: Print progress

    Returns:
        DataFrame with timestamp, label, p_bull, p_bear
    """
    # Drop NaN
    valid_mask = df[tvtp_cols].notna().all(axis=1)
    df_clean = df[valid_mask].copy()

    if len(df_clean) < config.min_observations:
        raise ValueError(f"Not enough data: {len(df_clean)} < {config.min_observations}")

    if verbose:
        print(f"Training NHHM on {len(df_clean):,} bars...")

    # Fit
    nhhm = NHHM(config)
    nhhm.fit(df_clean, exog_tvtp_cols=tvtp_cols, verbose=verbose)

    # Predict
    result = nhhm.predict(df_clean, exog_tvtp_cols=tvtp_cols)

    # Generate labels
    labels = np.ones(len(result.p_bull))  # Default neutral
    labels[result.p_bull >= threshold_bull] = 2  # Bull
    labels[result.p_bear >= threshold_bear] = 0  # Bear

    # Create output DataFrame
    output = pd.DataFrame({
        'timestamp': df_clean.index[-len(result.p_bull):],
        'label': labels,
        'p_bull': result.p_bull,
        'p_bear': result.p_bear,
    })

    return output


def main():
    print("="*60)
    print("NHHM Label Generator")
    print("="*60)

    # Configuration
    OUTPUT_DIR = Path("outputs/fourier/labels_frozen/BTC_FUSED_2h")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = NHHMConfig(
        n_regimes=2,
        forward_horizon=12,  # 24h
        min_observations=500
    )

    # Load data
    print("\nLoading BTC data...")
    df = pd.read_csv('data/BTC_FUSED_2h.csv', parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    print(f"Loaded {len(df):,} rows ({df.index[0]} to {df.index[-1]})")

    # Load funding
    print("\nLoading funding data...")
    try:
        from src.funding_rate import load_or_fetch_funding_rate
        funding_df = load_or_fetch_funding_rate(verbose=True)
    except Exception as e:
        print(f"Could not load funding: {e}")
        funding_df = None

    # Build features
    print("\nBuilding features...")
    df_features = build_nhhm_features(
        df,
        include_fourier=True,
        include_funding=(funding_df is not None),
        funding_df=funding_df
    )

    # ========================================
    # Generate labels: Fourier only (full history)
    # ========================================
    print("\n" + "="*60)
    print("Generating labels: NHHM + Fourier (2011-2025)")
    print("="*60)

    tvtp_cols_fourier = get_recommended_tvtp_cols(include_fourier=True, include_funding=False)
    print(f"Features: {tvtp_cols_fourier}")

    try:
        labels_fourier = generate_labels_simple(
            df_features,
            tvtp_cols_fourier,
            config,
            verbose=True
        )

        # Save
        output_path = OUTPUT_DIR / "NHHM_fourier.csv"
        labels_fourier[['timestamp', 'label']].to_csv(output_path, index=False)
        print(f"\nSaved {len(labels_fourier):,} labels to {output_path}")

        # Stats
        print(f"\nLabel distribution:")
        for label in sorted(labels_fourier['label'].unique()):
            count = (labels_fourier['label'] == label).sum()
            pct = count / len(labels_fourier) * 100
            name = {0: 'bear', 1: 'neutral', 2: 'bull'}.get(label, str(label))
            print(f"  {name}: {count:,} ({pct:.1f}%)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Generate labels: Fourier + Funding (2019+)
    # ========================================
    if funding_df is not None and 'funding_zscore' in df_features.columns:
        print("\n" + "="*60)
        print("Generating labels: NHHM + Fourier + Funding (2019+)")
        print("="*60)

        df_2019 = df_features[df_features.index >= '2019-10-01']
        tvtp_cols_funding = get_recommended_tvtp_cols(include_fourier=True, include_funding=True)
        print(f"Features: {tvtp_cols_funding}")

        try:
            labels_funding = generate_labels_simple(
                df_2019,
                tvtp_cols_funding,
                config,
                verbose=True
            )

            # Save
            output_path = OUTPUT_DIR / "NHHM_fourier_funding.csv"
            labels_funding[['timestamp', 'label']].to_csv(output_path, index=False)
            print(f"\nSaved {len(labels_funding):,} labels to {output_path}")

            # Stats
            print(f"\nLabel distribution:")
            for label in sorted(labels_funding['label'].unique()):
                count = (labels_funding['label'] == label).sum()
                pct = count / len(labels_funding) * 100
                name = {0: 'bear', 1: 'neutral', 2: 'bull'}.get(label, str(label))
                print(f"  {name}: {count:,} ({pct:.1f}%)")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
