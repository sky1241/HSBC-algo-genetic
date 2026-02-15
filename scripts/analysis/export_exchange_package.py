#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description='Package Fourier outputs for a single exchange')
    p.add_argument('--exchange', required=True, choices=['binance', 'bitstamp'])
    args = p.parse_args()

    sym_by_ex = {
        'binance': 'BTC_USDT',
        'bitstamp': 'BTC_USD',
    }
    sym = sym_by_ex[args.exchange]

    root = Path('outputs')
    monthly_root = root / 'fourier' / 'monthly'
    export_root = root / 'export' / args.exchange
    export_csv_dir = export_root / 'csv'
    export_root.mkdir(parents=True, exist_ok=True)
    export_csv_dir.mkdir(parents=True, exist_ok=True)

    # 1) Filter master CSV rows
    master_csv = root / 'FOURIER_ALL_REPORTS.csv'
    df = pd.read_csv(master_csv)
    df_ex = df[df['symbol'] == sym].copy()
    df_ex = df_ex.sort_values(['date', 'timeframe'])
    filtered_csv = export_root / f'FOURIER_ALL_REPORTS_{args.exchange}.csv'
    df_ex.to_csv(filtered_csv, index=False)

    # 2) Write XLSX (single sheet)
    xlsx_path = export_root / f'FOURIER_ALL_REPORTS_{args.exchange}.xlsx'
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_ex.to_excel(writer, sheet_name='ALL', index=False)

    # 3) Copy monthly CSVs for this exchange (both timeframes)
    for ydir in sorted([d for d in monthly_root.iterdir() if d.is_dir()]):
        for mdir in sorted([d for d in ydir.iterdir() if d.is_dir()]):
            for f in mdir.glob('FREQ_*.csv'):
                name = f.name  # FREQ_<SYM>_<TF>_<YYYY-MM>.csv
                parts = name.split('_')
                if len(parts) >= 4 and parts[1] == sym:
                    dst = export_csv_dir / name
                    shutil.copy2(f, dst)

    print(f'Wrote {filtered_csv}')
    print(f'Wrote {xlsx_path}')
    print(f'CSV copies in: {export_csv_dir}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


