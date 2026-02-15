#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agrège tous les CSV mensuels Fourier (2h et 1d) en un seul fichier:
- outputs/FOURIER_ALL_REPORTS.csv

Colonnes: date, timeframe, symbol, P1_bars, P2_bars, P3_bars, LFP, P1_vol, P2_vol, P3_vol, LFP_vol, csv_path
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> int:
    root = Path('outputs') / 'fourier' / 'monthly'
    rows = []
    for ydir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for mdir in sorted([d for d in ydir.iterdir() if d.is_dir()]):
            for csv in sorted(mdir.glob('FREQ_*_*.csv')):
                try:
                    parts = csv.stem.split('_')
                    # FREQ_<SYM>_<TF>_<YYYY-MM>
                    sym = parts[1]
                    tf = parts[2]
                    ym = parts[3]
                    df = pd.read_csv(csv)
                    # Read first non-NA row
                    first = df.dropna(how='all').iloc[0]
                    def g(col):
                        return first[col] if col in df.columns and pd.notna(first[col]) else None
                    rows.append({
                        'date': ym,
                        'timeframe': tf,
                        'symbol': sym,
                        'P1_bars': g('P1_bars') if 'P1_bars' in df.columns else g('P_bars'),
                        'P2_bars': g('P2_bars'),
                        'P3_bars': g('P3_bars'),
                        'LFP': g('LFP'),
                        'P1_vol': g('P1_vol'),
                        'P2_vol': g('P2_vol'),
                        'P3_vol': g('P3_vol'),
                        'LFP_vol': g('LFP_vol'),
                        'csv_path': csv.as_posix(),
                    })
                except Exception:
                    continue
    out = Path('outputs') / 'FOURIER_ALL_REPORTS.csv'
    # Tri: plus ancien -> plus récent (date), puis timeframe/symbol pour stabilité
    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values(['date','timeframe','symbol'])
    df_all.to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
