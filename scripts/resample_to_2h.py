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
    df2 = df.resample('2h').agg(agg).dropna()
    df2.to_csv(args.outp)
    print(f"Wrote {args.outp} rows={len(df2)} range={df2.index.min()} -> {df2.index.max()}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
