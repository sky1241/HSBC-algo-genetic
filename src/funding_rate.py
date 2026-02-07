#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funding Rate Feature Engineering for NHHM.

Funding rate is a key directional indicator for crypto perpetual futures:
- Positive funding = longs pay shorts = bullish sentiment (crowded long)
- Negative funding = shorts pay longs = bearish sentiment (crowded short)

High funding extremes often predict reversals (mean reversion).

Sources:
- BIS Working Paper No. 1087 (2023) "Crypto Carry"
- He & Manela (2022) "Fundamentals of Perpetual Futures"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import time

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Binance API endpoints
BINANCE_FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_funding_rate_binance(
    symbol: str = "BTCUSDT",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch historical funding rate from Binance Futures API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Max records per request (max 1000)
        verbose: Print progress

    Returns:
        DataFrame with columns: timestamp, funding_rate
    """
    if not HAS_REQUESTS:
        raise ImportError("requests required. Install with: pip install requests")

    all_data = []
    current_start = start_time

    while True:
        params = {
            "symbol": symbol,
            "limit": limit,
        }
        if current_start:
            params["startTime"] = current_start
        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(BINANCE_FUNDING_RATE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            if verbose:
                print(f"Error fetching funding rate: {e}")
            break

        if not data:
            break

        all_data.extend(data)

        if len(data) < limit:
            break

        # Move start to after last record
        current_start = data[-1]["fundingTime"] + 1

        if verbose:
            print(f"Fetched {len(all_data)} funding rate records...")

        time.sleep(0.1)  # Rate limiting

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"Total: {len(df)} funding rate records from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def load_or_fetch_funding_rate(
    symbol: str = "BTCUSDT",
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load funding rate from cache or fetch from API.

    Args:
        symbol: Trading pair
        cache_path: Path to cache file (default: data/funding_rate_{symbol}.csv)
        force_refresh: Force re-fetch from API
        verbose: Print progress

    Returns:
        DataFrame with funding rate history
    """
    if cache_path is None:
        cache_path = Path("data") / f"funding_rate_{symbol}.csv"

    if cache_path.exists() and not force_refresh:
        if verbose:
            print(f"Loading cached funding rate from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return df

    if verbose:
        print(f"Fetching funding rate from Binance API...")

    # Fetch all available history (Binance funding started around 2019-09)
    start_time = int(pd.Timestamp("2019-09-01").timestamp() * 1000)
    end_time = int(pd.Timestamp.now().timestamp() * 1000)

    df = fetch_funding_rate_binance(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose
    )

    if len(df) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        if verbose:
            print(f"Saved to {cache_path}")

    return df


def resample_funding_to_h2(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame
) -> pd.Series:
    """
    Resample 8h funding rate to H2 timeframe.

    Funding rate is paid every 8 hours on Binance.
    We forward-fill to H2 bars (no lookahead bias).

    Args:
        funding_df: DataFrame with timestamp, funding_rate
        ohlcv_df: OHLCV DataFrame with DatetimeIndex

    Returns:
        Series aligned to ohlcv_df index
    """
    # Set index
    funding = funding_df.set_index("timestamp")["funding_rate"]

    # Reindex to H2, forward fill (use last known funding)
    aligned = funding.reindex(ohlcv_df.index, method="ffill")

    return aligned


def compute_funding_features(
    funding_series: pd.Series,
    windows: list[int] = [6, 12, 36]  # 12h, 24h, 3d in H2 bars
) -> pd.DataFrame:
    """
    Compute funding rate features for NHHM.

    Features:
    - funding_rate: Raw funding rate (annualized)
    - funding_zscore: Z-score vs rolling mean
    - funding_momentum: Change in funding
    - funding_extreme: Flag for extreme values (potential reversal)

    Args:
        funding_series: Funding rate aligned to H2
        windows: Rolling windows for statistics

    Returns:
        DataFrame with funding features
    """
    features = pd.DataFrame(index=funding_series.index)

    # Raw funding (annualized: 3x per day * 365)
    features["funding_rate"] = funding_series
    features["funding_annual"] = funding_series * 3 * 365 * 100  # As percentage

    # Momentum (change)
    features["funding_change"] = funding_series.diff()
    features["funding_accel"] = features["funding_change"].diff()  # Acceleration

    # Rolling statistics
    for w in windows:
        roll_mean = funding_series.rolling(w).mean()
        roll_std = funding_series.rolling(w).std()

        features[f"funding_zscore_{w}"] = (funding_series - roll_mean) / (roll_std + 1e-10)
        features[f"funding_ma_{w}"] = roll_mean

    # Primary z-score (use 36-bar = 3 days)
    features["funding_zscore"] = features.get("funding_zscore_36", features["funding_zscore_12"])

    # Extreme flags (potential reversal signals)
    features["funding_extreme_high"] = (features["funding_zscore"] > 2).astype(float)
    features["funding_extreme_low"] = (features["funding_zscore"] < -2).astype(float)

    # Polarity signal
    # Positive funding + rising = bullish momentum (but crowded)
    # Negative funding + falling = bearish momentum (but crowded)
    features["funding_polarity"] = np.sign(funding_series) * np.sign(features["funding_change"])

    # Reversal signal (extreme + opposite momentum = potential reversal)
    extreme_high = features["funding_extreme_high"].astype(bool)
    extreme_low = features["funding_extreme_low"].astype(bool)
    change_neg = features["funding_change"] < 0
    change_pos = features["funding_change"] > 0

    features["funding_reversal"] = (
        (extreme_high & change_neg) | (extreme_low & change_pos)
    ).astype(float)

    return features


def add_funding_features_to_df(
    ohlcv_df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    symbol: str = "BTCUSDT",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add funding rate features to OHLCV DataFrame.

    Args:
        ohlcv_df: OHLCV DataFrame (must have DatetimeIndex)
        funding_df: Pre-loaded funding data (optional)
        symbol: Symbol to fetch if funding_df not provided
        verbose: Print progress

    Returns:
        DataFrame with funding features added
    """
    if funding_df is None:
        funding_df = load_or_fetch_funding_rate(symbol=symbol, verbose=verbose)

    if len(funding_df) == 0:
        if verbose:
            print("WARNING: No funding rate data available")
        return ohlcv_df

    # Align to H2
    funding_aligned = resample_funding_to_h2(funding_df, ohlcv_df)

    # Compute features
    funding_features = compute_funding_features(funding_aligned)

    # Merge
    result = ohlcv_df.copy()
    for col in funding_features.columns:
        result[col] = funding_features[col]

    if verbose:
        n_valid = result["funding_rate"].notna().sum()
        print(f"Added funding features: {n_valid}/{len(result)} rows with data")

    return result


# Quick test
if __name__ == "__main__":
    print("Testing funding rate module...")

    # Test fetch
    df = load_or_fetch_funding_rate(verbose=True)
    print(f"\nFunding rate sample:")
    print(df.tail(10))

    # Test features
    if len(df) > 0:
        # Create dummy OHLCV for alignment test
        dates = pd.date_range("2024-01-01", periods=100, freq="2h")
        dummy_ohlcv = pd.DataFrame({
            "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000
        }, index=dates)

        aligned = resample_funding_to_h2(df, dummy_ohlcv)
        features = compute_funding_features(aligned)
        print(f"\nFunding features sample:")
        print(features.tail())
