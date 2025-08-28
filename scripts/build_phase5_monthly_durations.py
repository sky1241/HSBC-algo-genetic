#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


OUT_DIR = Path('outputs') / 'fourier' / 'phase_monthly'


PHASES5: List[str] = [
    'accumulation', 'expansion', 'euphoria', 'distribution', 'bear'
]


def _parse_from_path(p: Path) -> Tuple[str, str, str]:
    """Return (symbol, timeframe, month) parsed from path
    outputs/fourier/phase_monthly/<SYM>/<TF>/<YYYY>/<YYYY-MM>/PHASE5_SEGMENTS.csv
    """
    # .../<SYM>/<TF>/<YYYY>/<YYYY-MM>/PHASE5_SEGMENTS.csv
    ym = p.parent.name
    tf = p.parents[2].name
    sym = p.parents[3].name
    return sym, tf, ym


def build_phase5_monthly_durations_from_segments() -> pd.DataFrame:
    """Scan monthly PHASE5_SEGMENTS.csv and aggregate durations per month/phase."""
    rows: List[dict] = []
    for seg_path in sorted(OUT_DIR.glob('*/*/*/*/PHASE5_SEGMENTS.csv')):
        try:
            sym, tf, ym = _parse_from_path(seg_path)
        except Exception:
            continue
        try:
            seg = pd.read_csv(seg_path)
        except Exception:
            continue
        # Normalize columns
        cols = {c.lower(): c for c in seg.columns}
        # Required: phase, d1_candles, h2_bars
        if 'phase' not in seg.columns:
            if 'phase' in cols:
                seg.rename(columns={cols['phase']: 'phase'}, inplace=True)
            else:
                continue
        if 'd1_candles' not in seg.columns and 'd1_candles' in cols:
            seg.rename(columns={cols['d1_candles']: 'd1_candles'}, inplace=True)
        if 'h2_bars' not in seg.columns and 'h2_bars' in cols:
            seg.rename(columns={cols['h2_bars']: 'h2_bars'}, inplace=True)
        # Aggregate by phase
        grp = seg.groupby('phase').agg({
            'd1_candles': 'sum',
            'h2_bars': 'sum',
        }).reset_index()
        row = {'month': ym, 'symbol': sym, 'timeframe': tf}
        for ph in PHASES5:
            ph_row = grp[grp['phase'] == ph]
            d_val = float(ph_row['d1_candles'].iloc[0]) if not ph_row.empty else 0.0
            h_val = float(ph_row['h2_bars'].iloc[0]) if not ph_row.empty else 0.0
            row[f'phase5_{ph}_days'] = int(d_val)
            row[f'phase5_{ph}_h2_bars'] = int(h_val)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out['month_dt'] = pd.to_datetime(out['month'] + '-01', errors='coerce')
    out = out.sort_values(['symbol','timeframe','month_dt']).drop(columns=['month_dt'])
    return out


def build_phase5_global_means(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (sym, tf), sub in monthly.groupby(['symbol','timeframe']):
        for ph in PHASES5:
            h2_col = f'phase5_{ph}_h2_bars'
            d_col = f'phase5_{ph}_days'
            if (h2_col not in sub.columns) or (d_col not in sub.columns):
                continue
            # include zeros; drop NaNs
            h2_mean = pd.to_numeric(sub[h2_col], errors='coerce').fillna(0).mean()
            d_mean = pd.to_numeric(sub[d_col], errors='coerce').fillna(0).mean()
            rows.append({
                'symbol': sym,
                'timeframe': tf,
                'phase': ph,
                'mean_h2_bars_per_month': round(float(h2_mean), 2),
                'mean_days_per_month': round(float(d_mean), 2),
            })
    return pd.DataFrame(rows).sort_values(['symbol','timeframe','phase'])


def main() -> int:
    monthly = build_phase5_monthly_durations_from_segments()
    if monthly.empty:
        print('No monthly segments found; nothing to write.')
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_monthly = OUT_DIR / 'PHASE5_MONTHLY_DURATIONS.csv'
    monthly.to_csv(out_monthly, index=False)
    print('Wrote:', out_monthly)

    means = build_phase5_global_means(monthly)
    out_means = OUT_DIR / 'PHASE5_GLOBAL_MEANS.csv'
    means.to_csv(out_means, index=False)
    print('Wrote:', out_means)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


