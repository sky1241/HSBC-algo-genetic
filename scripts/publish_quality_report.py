#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> int:
    qdir = Path('outputs') / 'quality_reports'
    rows = []
    for f in qdir.glob('*_QUALITY.csv'):
        df = pd.read_csv(f)
        df['report'] = f.name
        rows.append(df)
    if not rows:
        print('No quality reports found.')
        return 1
    full = pd.concat(rows, ignore_index=True)
    md = ['# Rapports de qualité OHLCV', '']
    md.append(full.to_markdown(index=False))
    out_md = Path('docs') / 'QUALITY_REPORTS.md'
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(f'Wrote {out_md}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
