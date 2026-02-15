#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


IN_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv'
OUT_MD = Path('docs') / 'PHASE_LABELS' / 'MONTHLY_SUMMARY.md'


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ''
    return f"{x*100:.1f}%"


def table_markdown(headers: List[str], rows: List[List[str]]) -> str:
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)
    return '\n'.join([head, sep, body])


def build_section(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    sec: List[str] = []
    sec.append(f"### {symbol} {timeframe}")

    # Ensure expected columns exist
    for c in ['regime3_up_share','regime3_down_share','regime3_range_share','LFP_mean','LFP_vol_mean']:
        if c not in df.columns:
            df[c] = pd.NA

    # Tops (5 meilleurs mois) par part de phase
    tops = []
    for label in ['up','down','range']:
        col = f'regime3_{label}_share'
        sub = df[['month', col, 'LFP_mean','LFP_vol_mean']].dropna(subset=[col]).sort_values(col, ascending=False).head(5)
        for _, r in sub.iterrows():
            tops.append([
                label,
                r['month'],
                fmt_pct(r[col]),
                f"{r['LFP_mean']:.2f}" if pd.notna(r['LFP_mean']) else '',
                f"{r['LFP_vol_mean']:.2f}" if pd.notna(r['LFP_vol_mean']) else '',
            ])
    if tops:
        sec.append('Top 5 mois par part de phase (regime3)')
        sec.append(table_markdown(['Phase','Mois','Part','LFP_mean','LFP_vol_mean'], tops))
        sec.append('')

    # Corrélations (Pearson) entre parts et LFP
    corr_rows = []
    corr_df = df[['regime3_up_share','regime3_down_share','regime3_range_share','LFP_mean','LFP_vol_mean']].dropna()
    if len(corr_df) >= 3:
        for phase_col in ['regime3_up_share','regime3_down_share','regime3_range_share']:
            for lfp_col in ['LFP_mean','LFP_vol_mean']:
                try:
                    val = corr_df[phase_col].corr(corr_df[lfp_col])
                except Exception:
                    val = float('nan')
                corr_rows.append([phase_col.replace('regime3_',''), lfp_col, f"{val:.3f}" if pd.notna(val) else ''])
    if corr_rows:
        sec.append('Corrélations (Pearson) entre parts de phases (regime3) et LFP')
        sec.append(table_markdown(['Phase','Metric','Corr'], corr_rows))
        sec.append('')

    return '\n'.join(sec)


def main() -> int:
    if not IN_CSV.exists():
        print('Input CSV missing:', IN_CSV)
        return 0
    full = pd.read_csv(IN_CSV)
    # Ensure month ordering
    full['month_dt'] = pd.to_datetime(full['month'] + '-01', errors='coerce')
    full = full.sort_values(['symbol','timeframe','month_dt'])

    parts: List[str] = []
    parts.append('## Synthèse mensuelle (tops et corrélations)')
    parts.append('Source: `outputs/fourier/phase_monthly/ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv`')
    parts.append('')

    for sym in ['BTC_USDT','BTC_USD']:
        for tf in ['2h','1d']:
            sub = full[(full['symbol'] == sym) & (full['timeframe'] == tf)].copy()
            if sub.empty:
                continue
            parts.append(build_section(sub, sym, tf))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(parts), encoding='utf-8')
    print('Wrote:', OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


