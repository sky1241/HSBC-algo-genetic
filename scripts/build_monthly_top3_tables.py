#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import pandas as pd


def extract_month_row(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    # Expect columns: timestamp, P1_bars, P2_bars, P3_bars, LFP, P1_vol, P2_vol, P3_vol, LFP_vol (some optional)
    cols = df.columns
    # Read first non-null values
    def first(col: str):
        return (df[col].dropna().iloc[0] if (col in cols and df[col].dropna().shape[0] > 0) else None)
    parts = csv_path.stem.split('_')  # FREQ_<SYM>_<TF>_<YYYY-MM>
    sym = parts[1]
    tf = parts[2]
    ym = parts[3]
    return {
        'date': ym,
        'symbol': sym,
        'timeframe': tf,
        'P1_bars': first('P1_bars') or first('P_bars'),
        'P2_bars': first('P2_bars'),
        'P3_bars': first('P3_bars'),
        'P4_bars': first('P4_bars'),
        'P5_bars': first('P5_bars'),
        'P6_bars': first('P6_bars'),
        'LFP': first('LFP'),
        'P1_vol': first('P1_vol'),
        'P2_vol': first('P2_vol'),
        'P3_vol': first('P3_vol'),
        'LFP_vol': first('LFP_vol'),
        'source_csv': csv_path.as_posix(),
    }


def build_table(root: Path, sym: str, tf: str) -> pd.DataFrame:
    rows = []
    for ydir in sorted([d for d in (root / 'monthly').iterdir() if d.is_dir()]):
        for mdir in sorted([d for d in ydir.iterdir() if d.is_dir()]):
            csv = mdir / f"FREQ_{sym}_{tf}_{ydir.name}-{mdir.name}.csv"
            if csv.exists():
                try:
                    rows.append(extract_month_row(csv))
                except Exception:
                    continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['date'])
    return df


def write_markdown(df: pd.DataFrame, md_path: Path, title: str) -> None:
    lines = [f"### {title}", "", df.to_markdown(index=False)]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    root = Path('outputs') / 'fourier'
    out_docs = Path('docs') / 'CYCLES_EMERGENCE'
    combos = [
        ('BTC_USDT','2h'),
        ('BTC_USDT','1d'),
        ('BTC_USD','2h'),
        ('BTC_USD','1d'),
    ]
    for sym, tf in combos:
        df = build_table(root, sym, tf)
        if df.empty:
            continue
        csv_out = out_docs / f"TABLE_TOP3_{sym}_{tf}.csv"
        md_out = out_docs / f"TABLE_TOP3_{sym}_{tf}.md"
        df.to_csv(csv_out, index=False)
        write_markdown(df[['date','P1_bars','P2_bars','P3_bars','P4_bars','P5_bars','P6_bars','LFP','P1_vol','P2_vol','P3_vol','LFP_vol']], md_out, f"Top‑6 mensuel — {sym} {tf}")
        print('Wrote', csv_out)
        print('Wrote', md_out)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


