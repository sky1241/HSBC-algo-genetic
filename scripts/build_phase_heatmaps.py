#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import sys

import pandas as pd
import matplotlib.pyplot as plt


IN_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv'
OUT_DIR = Path('outputs') / 'fourier' / 'phase_monthly' / 'HEATMAPS'
OUT_MD = Path('docs') / 'PHASE_LABELS' / 'MONTHLY_HEATMAPS.md'


REGIME3_COLS = ['regime3_up_share','regime3_down_share','regime3_range_share']
PHASE5_COLS = [
    'phase5_accumulation_share','phase5_expansion_share','phase5_euphoria_share',
    'phase5_distribution_share','phase5_bear_share',
]
PHASE6_COLS = PHASE5_COLS + ['phase6_capitulation_share']


def _prepare_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    mat = df[['month'] + cols].copy()
    # Order months chronologically
    mat['month_dt'] = pd.to_datetime(mat['month'] + '-01', errors='coerce')
    mat = mat.sort_values('month_dt')
    # Convert to percent 0..100
    for c in cols:
        if c in mat.columns:
            mat[c] = mat[c].astype(float) * 100.0
        else:
            mat[c] = 0.0
    labels = mat['month'].tolist()
    mat = mat[cols]
    return mat, labels


def _heatmap(ax: plt.Axes, data: pd.DataFrame, months: List[str], title: str, cmap: str = 'viridis') -> None:
    im = ax.imshow(data.values, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(len(months)))
    # show fewer y-ticks for readability
    yticks = list(range(0, len(months), max(1, len(months)//12)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([months[i] for i in yticks], fontsize=8)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([c.replace('regime3_','').replace('phase5_','').replace('phase6_','').replace('_share','') for c in data.columns], rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('% part', rotation=90, fontsize=8)


def build_heatmaps() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(IN_CSV)
    full = pd.read_csv(IN_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = []
    md_lines.append('## Heatmaps mensuelles des parts de phases')
    md_lines.append('Source: `outputs/fourier/phase_monthly/ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv`')
    md_lines.append('')

    for sym in ['BTC_USDT','BTC_USD']:
        for tf in ['2h','1d']:
            sub = full[(full['symbol'] == sym) & (full['timeframe'] == tf)].copy()
            if sub.empty:
                continue
            base = OUT_DIR / sym / tf
            base.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

            # regime3
            mat, months = _prepare_matrix(sub, REGIME3_COLS)
            _heatmap(axes[0], mat, months, f'{sym} {tf} — regime3')

            # phase5
            mat5, months5 = _prepare_matrix(sub, PHASE5_COLS)
            _heatmap(axes[1], mat5, months5, f'{sym} {tf} — phase5')

            # phase6
            mat6, months6 = _prepare_matrix(sub, PHASE6_COLS)
            _heatmap(axes[2], mat6, months6, f'{sym} {tf} — phase6')

            out_png = base / f'{sym}_{tf}_phase_shares_heatmaps.png'
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

            md_lines.append(f'### {sym} {tf}')
            md_lines.append(f'![{out_png.name}]({out_png.as_posix()})')
            md_lines.append('')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Wrote:', OUT_MD)


def main() -> int:
    build_heatmaps()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


