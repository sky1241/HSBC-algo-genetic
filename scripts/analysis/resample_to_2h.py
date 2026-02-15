#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample 1h OHLCV CSV to 2h OHLCV CSV with proper OHLC aggregation.
Usage:
  py -3 scripts/resample_to_2h.py --in data/BTC_USD_1h.csv --out data/BTC_USD_2h.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from scipy import signal


def _fir_lowpass(df: pd.DataFrame, cutoff_per_day: float = 0.4) -> pd.DataFrame:
    """Apply a zero-phase FIR low-pass filter to all numeric columns.

    Parameters
    ----------
    df: pd.DataFrame
        Hourly data indexed by timestamp.
    cutoff_per_day: float
        Cut-off frequency in cycles per day (defaults to ~0.4 j^-1).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with the same index/columns.
    """
    # Hourly sampling → 24 samples per day
    fs = 24.0
    nyq = fs / 2.0
    b = signal.firwin(101, cutoff_per_day / nyq)
    filtered = {}
    for col in df.columns:
        filtered[col] = signal.filtfilt(b, [1.0], df[col].to_numpy())
    return pd.DataFrame(filtered, index=df.index)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='outp', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    assert 'timestamp' in df.columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    # Low-pass filter before downsampling to avoid aliasing
    df_filt = _fir_lowpass(df)
    df2 = df_filt.resample('2h').agg(agg).dropna()
    df2.to_csv(args.outp)
    print(f"Wrote {args.outp} rows={len(df2)} range={df2.index.min()} -> {df2.index.max()}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
