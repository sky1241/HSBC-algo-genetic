#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_phase_price import (
    read_ohlcv,
    load_daily_summary,
    label_3_regimes,
)
from scripts.phase_aware_module import phase_snapshot


PAIRS_TF: List[Tuple[str, str]] = [
    ("BTC_USDT", "2h"),
    ("BTC_USDT", "1d"),
    ("BTC_USD", "2h"),
    ("BTC_USD", "1d"),
]


def bars_per_day(tf: str) -> int:
    if tf.endswith('h'):
        try:
            h = float(tf[:-1])
            return int(round(24.0 / h)) if h > 0 else 12
        except Exception:
            return 12
    return 1


def aggregate_monthly_regime3(symbol: str, timeframe: str) -> pd.DataFrame:
    ohlcv_path = Path('data') / f"{symbol}_{timeframe}.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(ohlcv_path)

    df = read_ohlcv(ohlcv_path)
    feats = phase_snapshot(df).reindex(df.index).ffill()
    daily = load_daily_summary(symbol, timeframe)
    daily_aligned = daily.reindex(df.index, method='ffill')

    labels = pd.Series(
        [label_3_regimes(m, lfp) for m, lfp in zip(feats['M'], daily_aligned['LFP'])],
        index=df.index,
        name='label',
    )

    df_lab = pd.DataFrame({'label': labels})
    df_lab['month'] = df_lab.index.to_period('M').to_timestamp()

    # Count bars per label per month
    counts = df_lab.pivot_table(index='month', columns='label', values='label', aggfunc='count').fillna(0)
    for col in ['up','down','range']:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[['up','down','range']].astype(int)

    bpd = bars_per_day(timeframe)
    # Convert to durations
    out = pd.DataFrame(index=counts.index).sort_index()
    out['symbol'] = symbol
    out['timeframe'] = timeframe
    out['up_h2_bars'] = counts['up'] * (12 if timeframe == '1d' else 1)
    out['down_h2_bars'] = counts['down'] * (12 if timeframe == '1d' else 1)
    out['range_h2_bars'] = counts['range'] * (12 if timeframe == '1d' else 1)
    out['up_days'] = counts['up'] / (12 if timeframe == '2h' else 1)
    out['down_days'] = counts['down'] / (12 if timeframe == '2h' else 1)
    out['range_days'] = counts['range'] / (12 if timeframe == '2h' else 1)

    # Totals and shares
    out['h2_bars_total'] = out['up_h2_bars'] + out['down_h2_bars'] + out['range_h2_bars']
    out['days_total'] = out['up_days'] + out['down_days'] + out['range_days']
    out['up_share'] = (out['up_h2_bars'] / out['h2_bars_total']).where(out['h2_bars_total'] > 0)
    out['down_share'] = (out['down_h2_bars'] / out['h2_bars_total']).where(out['h2_bars_total'] > 0)
    out['range_share'] = (out['range_h2_bars'] / out['h2_bars_total']).where(out['h2_bars_total'] > 0)

    # Monthly Fourier aggregation from daily summary
    daily_m = daily.copy()
    daily_m['month'] = daily_m.index.to_period('M').to_timestamp()
    agg = daily_m.groupby('month').agg({
        'LFP': ['mean','median'],
        'LFP_vol': ['mean','median'],
        'P1_bars': 'median',
        'P2_bars': 'median',
        'P3_bars': 'median',
        'P4_bars': 'median',
        'P5_bars': 'median',
        'P6_bars': 'median',
        'P1_vol': 'median',
        'P2_vol': 'median',
        'P3_vol': 'median',
    })
    # Flatten columns
    agg.columns = ['_'.join([c for c in col if c]).strip('_') for col in agg.columns.values]

    out = out.join(agg, how='left')
    out = out.reset_index().rename(columns={'index':'month'})
    out['month'] = pd.to_datetime(out['month']).dt.strftime('%Y-%m')
    return out


def main() -> int:
    rows: List[pd.DataFrame] = []
    for sym, tf in PAIRS_TF:
        try:
            rows.append(aggregate_monthly_regime3(sym, tf))
        except FileNotFoundError:
            continue

    if not rows:
        print('No data found.')
        return 0

    full = pd.concat(rows, ignore_index=True)
    full = full.sort_values(['month','symbol','timeframe'])
    out_dir = Path('outputs') / 'fourier' / 'phase_monthly'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'REGIME3_DURATIONS_WITH_FOURIER.csv'
    full.to_csv(out_csv, index=False)
    print('Wrote:', out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


