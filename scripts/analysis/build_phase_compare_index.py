#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def find_summary_files(base: Path) -> list[Path]:
    files: list[Path] = []
    if not base.exists():
        return files
    for sub in base.glob('*_*_*'):
        if not sub.is_dir():
            continue
        for f in sub.glob('SUMMARY_*.csv'):
            files.append(f)
    return sorted(files)


def parse_context_from_path(p: Path) -> tuple[str, str, str]:
    # examples: outputs/fourier/compare/BTC_USDT_2h/SUMMARY_phase6.csv
    sym_tf = p.parent.name
    try:
        symbol, timeframe = sym_tf.rsplit('_', 1)
    except ValueError:
        symbol, timeframe = sym_tf, ''
    labelset = p.stem.replace('SUMMARY_', '')
    return symbol, timeframe, labelset


def build_aggregated_csv() -> Path:
    base = Path('outputs') / 'fourier' / 'compare'
    out_csv = base / 'ALL_SUMMARIES_BY_LABELSET.csv'
    rows: list[pd.DataFrame] = []
    for f in find_summary_files(base):
        symbol, timeframe, labelset = parse_context_from_path(f)
        df = pd.read_csv(f)
        df.insert(0, 'symbol', symbol)
        df.insert(1, 'timeframe', timeframe)
        df.insert(2, 'labelset', labelset)
        rows.append(df)
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    all_df.to_csv(out_csv, index=False)
    return out_csv


def write_markdown_index(agg_csv: Path) -> Path:
    docs_dir = Path('docs') / 'PHASE_LABELS'
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / 'COMPARATIF_DUREES_FOURIER_HALVING.md'

    if not agg_csv.exists():
        md_path.write_text('# Comparatif introuvable\nFichier agrégé manquant.')
        return md_path

    df = pd.read_csv(agg_csv)
    if df.empty:
        md_path.write_text('# Comparatif vide\nAucune donnée agrégée.')
        return md_path

    parts: list[str] = []
    parts.append('## Comparatif durées et Fourier vs phases (3/5/6)')
    parts.append('Données: symboles BTC, timeframes 2h/1d, sources Binance (USDT) et Bitstamp (USD).')

    for (symbol, timeframe), g in df.groupby(['symbol', 'timeframe']):
        parts.append(f"\n### {symbol} — {timeframe}")
        for labelset, gl in g.groupby('labelset'):
            parts.append(f"\n#### Label set: {labelset}")
            cols = [c for c in gl.columns if c in (
                'label','dsh_mid_median','duration_days','duration_h2_bars',
                'P1_med','P2_med','P3_med','P4_med','P5_med','P6_med','LFP_mean'
            )]
            # ensure ordering
            ordered = ['label','dsh_mid_median','duration_days','duration_h2_bars',
                       'P1_med','P2_med','P3_med','P4_med','P5_med','P6_med','LFP_mean']
            cols = [c for c in ordered if c in cols]
            tbl = gl[cols].sort_values('label')
            # pretty rounding
            for c in tbl.columns:
                if c in ('label',):
                    continue
                tbl[c] = np.round(tbl[c].astype(float), 2)
            parts.append(tbl.to_markdown(index=False))

    md_path.write_text('\n'.join(parts), encoding='utf-8')
    return md_path


def main() -> int:
    agg_csv = build_aggregated_csv()
    md_path = write_markdown_index(agg_csv)
    print('Wrote:', agg_csv)
    print('Wrote:', md_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())


