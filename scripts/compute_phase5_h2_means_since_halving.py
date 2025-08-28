#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


IN_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'PHASE5_MONTHLY_DURATIONS.csv'
OUT_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'PHASE5_H2_MEANS_SINCE_2012-11.csv'

PHASES5: List[str] = ['accumulation','expansion','euphoria','distribution','bear']


def main() -> int:
    if not IN_CSV.exists():
        print('Missing input:', IN_CSV)
        return 0
    df = pd.read_csv(IN_CSV)
    # month filter from first halving month (inclusive)
    df['month_dt'] = pd.to_datetime(df['month'] + '-01', errors='coerce')
    df = df[(df['timeframe'] == '2h') & (df['month_dt'] >= pd.Timestamp('2012-11-01'))].copy()
    if df.empty:
        print('No 2h monthly data since 2012-11.')
        return 0
    rows = []
    for (sym, tf), sub in df.groupby(['symbol','timeframe']):
        for ph in PHASES5:
            h2_col = f'phase5_{ph}_h2_bars'
            d_col = f'phase5_{ph}_days'
            if h2_col not in sub.columns or d_col not in sub.columns:
                continue
            h2_mean = pd.to_numeric(sub[h2_col], errors='coerce').fillna(0).mean()
            d_mean = pd.to_numeric(sub[d_col], errors='coerce').fillna(0).mean()
            rows.append({
                'symbol': sym,
                'timeframe': tf,
                'phase': ph,
                'mean_h2_bars_per_month': round(float(h2_mean), 2),
                'mean_days_per_month': round(float(d_mean), 2),
                'months_count': int(sub.shape[0]),
            })
    out = pd.DataFrame(rows).sort_values(['symbol','timeframe','phase'])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print('Wrote:', OUT_CSV)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


