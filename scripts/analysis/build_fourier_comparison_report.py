#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit un rapport comparatif H2 vs D1 (BTC/USDT) pour P1/P2/P3 et LFP
sur la période 2020-01-01 → fin des données.

Sorties:
- docs/FOURIER_COMPARAISON_H2_vs_D1.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_series(csv_path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    return df[cols]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTC/USDT')
    p.add_argument('--from', dest='date_from', default='2020-01-01')
    args = p.parse_args()

    root = Path('outputs') / 'fourier'
    sym = args.symbol.replace('/', '_')

    # Annual
    h2_ann = load_series(root / f"FREQ_ANNUAL_{sym}_2h.csv", ['P1_bars','P2_bars','P3_bars','LFP'])
    d1_ann = load_series(root / f"FREQ_ANNUAL_{sym}_1d.csv", ['P1_bars','P2_bars','P3_bars','LFP'])

    h2_ann = h2_ann.loc[h2_ann.index >= pd.Timestamp(args.date_from)]
    d1_ann = d1_ann.loc[d1_ann.index >= pd.Timestamp(args.date_from)]

    idx_ann = h2_ann.index.intersection(d1_ann.index)
    h2_ann = h2_ann.loc[idx_ann]
    d1_ann = d1_ann.loc[idx_ann]

    diffs_ann = pd.DataFrame(index=idx_ann)
    for col in ['P1_bars','P2_bars','P3_bars','LFP']:
        diffs_ann[col] = h2_ann[col] - d1_ann[col]

    summary_ann = pd.DataFrame({
        'H2_mean': h2_ann.mean(),
        'D1_mean': d1_ann.mean(),
        'H2_minus_D1_mean': diffs_ann.mean(),
        'H2_minus_D1_median': diffs_ann.median(),
        'H2_minus_D1_std': diffs_ann.std(),
        'count': diffs_ann.count(),
    })

    # Monthly (rolling window)
    h2_mon = load_series(root / f"FREQ_MONTHLY_{sym}_2h.csv", ['P1_bars','P2_bars','P3_bars','LFP'])
    d1_mon = load_series(root / f"FREQ_MONTHLY_{sym}_1d.csv", ['P1_bars','P2_bars','P3_bars','LFP'])

    h2_mon = h2_mon.loc[h2_mon.index >= pd.Timestamp(args.date_from)]
    d1_mon = d1_mon.loc[d1_mon.index >= pd.Timestamp(args.date_from)]

    idx_mon = h2_mon.index.intersection(d1_mon.index)
    h2_mon = h2_mon.loc[idx_mon]
    d1_mon = d1_mon.loc[idx_mon]

    diffs_mon = pd.DataFrame(index=idx_mon)
    for col in ['P1_bars','P2_bars','P3_bars','LFP']:
        diffs_mon[col] = h2_mon[col] - d1_mon[col]

    summary_mon = pd.DataFrame({
        'H2_mean': h2_mon.mean(),
        'D1_mean': d1_mon.mean(),
        'H2_minus_D1_mean': diffs_mon.mean(),
        'H2_minus_D1_median': diffs_mon.median(),
        'H2_minus_D1_std': diffs_mon.std(),
        'count': diffs_mon.count(),
    })

    out_md = Path('docs') / 'FOURIER_COMPARAISON_H2_vs_D1.md'
    lines: list[str] = []
    lines.append(f"# Comparaison Fourier — H2 vs D1 ({args.symbol})\n")
    lines.append(f"Période: {args.date_from} → {idx_ann.max().date()}\n")
    lines.append("\n## Synthèse statistique (rolling annual)\n")
    lines.append(summary_ann.to_markdown())
    lines.append("\n\n## Synthèse statistique (rolling monthly)\n")
    lines.append(summary_mon.to_markdown())
    lines.append("\n\n## Pointeurs graphiques\n")
    base_plots = Path('outputs') / 'fourier' / 'plots'
    lines.append(f"- H2 P (annual): { (base_plots / f'{sym}_2h_P_annual.png').as_posix() }\n")
    lines.append(f"- D1 P (annual): { (base_plots / f'{sym}_1d_P_annual.png').as_posix() }\n")
    lines.append(f"- H2 LFP (annual): { (base_plots / f'{sym}_2h_LFP_annual.png').as_posix() }\n")
    lines.append(f"- D1 LFP (annual): { (base_plots / f'{sym}_1d_LFP_annual.png').as_posix() }\n")
    lines.append(f"- H2 P (monthly): { (base_plots / f'{sym}_2h_P_monthly.png').as_posix() }\n")
    lines.append(f"- D1 P (monthly): { (base_plots / f'{sym}_1d_P_monthly.png').as_posix() }\n")
    lines.append(f"- H2 LFP (monthly): { (base_plots / f'{sym}_2h_LFP_monthly.png').as_posix() }\n")
    lines.append(f"- D1 LFP (monthly): { (base_plots / f'{sym}_1d_LFP_monthly.png').as_posix() }\n")

    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote {out_md}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


