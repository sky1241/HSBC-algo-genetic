#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


IN_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv'
OUT_DIR = Path('outputs') / 'fourier' / 'phase_monthly'


PHASES5: List[str] = [
    'accumulation', 'expansion', 'euphoria', 'distribution', 'bear'
]


def build_phase5_monthly_durations(full: pd.DataFrame) -> pd.DataFrame:
    """Return a pivoted monthly table with H2 bars and days per phase5."""
    keep_cols: List[str] = ['month', 'symbol', 'timeframe']
    for ph in PHASES5:
        keep_cols.append(f'phase5_{ph}_h2_bars')
        keep_cols.append(f'phase5_{ph}_days')
    # Only keep rows where at least timeframe/symbol present
    cols_present = [c for c in keep_cols if c in full.columns]
    out = full[cols_present].copy()
    # Sort for readability
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
    if not IN_CSV.exists():
        print('Input missing:', IN_CSV)
        return 0
    full = pd.read_csv(IN_CSV)
    monthly = build_phase5_monthly_durations(full)
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


