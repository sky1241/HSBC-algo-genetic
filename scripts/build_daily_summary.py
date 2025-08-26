#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build daily summaries (P1/P2/P3/LFP price and volume) from monthly-rolling Fourier CSVs.

Outputs per symbol/timeframe:
  outputs/fourier/DAILY_SUMMARY_<symbol>_<tf>.csv

Also writes a combined summary:
  outputs/fourier/DAILY_SUMMARY_ALL.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def summarize_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError(f"Missing timestamp in {csv_path}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    # Normalize column presence
    for col in ['P1_bars','P2_bars','P3_bars','P4_bars','P5_bars','P6_bars','P_bars','LFP','P1_vol','P2_vol','P3_vol','LFP_vol']:
        if col not in df.columns:
            df[col] = pd.NA
    # If only P_bars exists, alias to P1_bars
    df['P1_bars'] = df['P1_bars'].combine_first(df['P_bars'])
    # Resample to daily (take last valid value each day)
    daily = df.resample('1D').last().dropna(how='all')
    return daily[['P1_bars','P2_bars','P3_bars','P4_bars','P5_bars','P6_bars','LFP','P1_vol','P2_vol','P3_vol','LFP_vol']]


def main() -> int:
    root = Path('outputs') / 'fourier'
    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    # Known pairs/timeframes
    for sym in ['BTC_USDT','BTC_USD']:
        for tf in ['2h','1d']:
            path = root / f"FREQ_MONTHLY_{sym}_{tf}.csv"
            if path.exists():
                targets.append((sym, tf, path))

    all_rows = []
    for sym, tf, path in targets:
        daily = summarize_one(path)
        daily['symbol'] = sym
        daily['timeframe'] = tf
        out_csv = out_dir / f"DAILY_SUMMARY_{sym}_{tf}.csv"
        daily.reset_index().to_csv(out_csv, index=False)
        all_rows.append(daily.reset_index())

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full = full.sort_values(['timestamp','symbol','timeframe'])
        full.to_csv(out_dir / 'DAILY_SUMMARY_ALL.csv', index=False)
        print("Wrote:", out_dir / 'DAILY_SUMMARY_ALL.csv')
    else:
        print('No monthly Fourier CSVs found.')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


