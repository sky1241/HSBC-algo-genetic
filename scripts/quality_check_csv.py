#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality checks for OHLCV CSV files:
- Timestamp monotonicity, duplicates
- Gaps vs expected frequency
- NaN in OHLCV
- OHLC consistency (low <= open/close/high, high >= open/close/low)
- Non-positive prices, zero/negative volumes
- Simple outlier check on log-returns (z-score)

Usage:
  py -3 scripts/quality_check_csv.py --csv data/BTC_USDT_2h.csv --freq 2h
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def check_csv(path: Path, freq: str) -> dict:
    report: dict = {'file': path.as_posix()}
    df = pd.read_csv(path)
    assert 'timestamp' in df.columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    # Basic
    report['rows'] = len(df)
    report['has_dupes'] = int(df.index.duplicated().any())
    report['has_nans'] = int(df[['open','high','low','close','volume']].isna().any().any())

    # Gaps
    full = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing = len(full.difference(df.index))
    report['missing_bars'] = int(missing)

    # OHLC consistency
    bad_low = (df['low'] > df[['open','close','high']].min(axis=1)).sum()
    bad_high = (df['high'] < df[['open','close','low']].max(axis=1)).sum()
    report['bad_low'] = int(bad_low)
    report['bad_high'] = int(bad_high)

    # Non-positive
    report['non_positive_prices'] = int((df[['open','high','low','close']] <= 0).any(axis=1).sum())
    report['non_positive_volume'] = int((df['volume'] < 0).sum())
    report['zero_volume'] = int((df['volume'] == 0).sum())

    # Outliers on returns
    df['log_close'] = np.log(df['close'])
    ret = df['log_close'].diff().dropna()
    if len(ret) > 10:
        z = (ret - ret.mean()) / ret.std(ddof=1)
        report['ret_outliers_4sigma'] = int((np.abs(z) > 4).sum())
    else:
        report['ret_outliers_4sigma'] = 0

    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--freq', required=True, help='Pandas offset alias (e.g., 2h, 1d)')
    args = p.parse_args()

    rep = check_csv(Path(args.csv), args.freq)
    out = Path('outputs') / 'quality_reports'
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / (Path(args.csv).stem + '_QUALITY.csv')
    pd.DataFrame([rep]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
