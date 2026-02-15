#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SEG_GLOB = Path('outputs') / 'fourier' / 'phase_monthly'
AGG_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'PHASE5_H2_MEANS_SINCE_2012-11.csv'
OUT_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'PHASE5_H2_MEANS_VALIDATION.csv'

PHASES = ['accumulation','bear','distribution','euphoria','expansion']


def recompute_from_segments() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sym_dir in sorted([d for d in (SEG_GLOB).iterdir() if d.is_dir()]):
        sym = sym_dir.name
        tf_dir = sym_dir / '2h'
        if not tf_dir.exists():
            continue
        for ydir in sorted([d for d in tf_dir.iterdir() if d.is_dir()]):
            for mdir in sorted([d for d in ydir.iterdir() if d.is_dir()]):
                ym = mdir.name
                if ym < '2012-11':
                    continue
                seg_csv = mdir / 'PHASE5_SEGMENTS.csv'
                if not seg_csv.exists():
                    continue
                seg = pd.read_csv(seg_csv)
                grp = seg.groupby('phase').agg({'d1_candles': 'sum', 'h2_bars': 'sum'}).reset_index()
                row = {'symbol': sym, 'timeframe': '2h', 'month': ym}
                for ph in PHASES:
                    sub = grp[grp['phase'] == ph]
                    row[f'{ph}_days'] = int(float(sub['d1_candles'].iloc[0])) if not sub.empty else 0
                    row[f'{ph}_h2_bars'] = int(float(sub['h2_bars'].iloc[0])) if not sub.empty else 0
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    monthly = pd.DataFrame(rows)
    # Means per symbol over all months
    means: List[Dict[str, object]] = []
    for sym, sub in monthly.groupby('symbol'):
        d: Dict[str, object] = {'symbol': sym, 'timeframe': '2h'}
        for ph in PHASES:
            d[f'mean_{ph}_h2_bars_per_month'] = round(float(pd.to_numeric(sub[f'{ph}_h2_bars']).mean()), 2)
            d[f'mean_{ph}_days_per_month'] = round(float(pd.to_numeric(sub[f'{ph}_days']).mean()), 2)
        d['months_count'] = int(sub.shape[0])
        means.append(d)
    return pd.DataFrame(means).sort_values(['symbol'])


def compare_with_agg(recomp: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    # reshape agg to comparable columns
    pivot_rows: List[Dict[str, object]] = []
    for (sym, tf), sub in agg.groupby(['symbol','timeframe']):
        row: Dict[str, object] = {'symbol': sym, 'timeframe': tf}
        for _, r in sub.iterrows():
            ph = str(r['phase'])
            row[f'mean_{ph}_h2_bars_per_month__agg'] = float(r['mean_h2_bars_per_month'])
            row[f'mean_{ph}_days_per_month__agg'] = float(r['mean_days_per_month'])
        pivot_rows.append(row)
    agg_pivot = pd.DataFrame(pivot_rows)
    out = recomp.merge(agg_pivot, on=['symbol','timeframe'], how='left')
    return out


def main() -> int:
    recomp = recompute_from_segments()
    if recomp.empty:
        print('No segments found for validation.')
        return 0
    if AGG_CSV.exists():
        agg = pd.read_csv(AGG_CSV)
        out = compare_with_agg(recomp, agg)
    else:
        out = recomp
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print('Wrote:', OUT_CSV)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
