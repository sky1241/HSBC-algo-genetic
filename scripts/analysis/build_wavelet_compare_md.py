#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import pandas as pd


WAVELET_CSV = Path('outputs') / 'fourier' / 'wavelets' / 'WAVELET_STFT_MONTHLY_METRICS.csv'
FOURIER_CSV = Path('outputs') / 'fourier' / 'phase_monthly' / 'ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv'
OUT_MD = Path('docs') / 'FOURIER_WAVELETS_COMPARE.md'


def to_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    sub = df[cols].head(max_rows)
    header = '| ' + ' | '.join(sub.columns) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(sub.columns)) + ' |'
    rows = ['| ' + ' | '.join(map(lambda x: '' if pd.isna(x) else str(x), r)) + ' |' for r in sub.values.tolist()]
    return '\n'.join([header, sep] + rows)


def main() -> int:
    if not WAVELET_CSV.exists() or not FOURIER_CSV.exists():
        print('Missing input CSVs')
        return 0
    wav = pd.read_csv(WAVELET_CSV)
    fou = pd.read_csv(FOURIER_CSV)

    # Merge on month/symbol/timeframe
    merged = pd.merge(
        fou[['month','symbol','timeframe','LFP_mean','P1_bars_median','P2_bars_median','P3_bars_median']],
        wav,
        on=['month','symbol','timeframe'],
        how='inner'
    )
    # Simple correlations per pair/timeframe
    parts: list[str] = []
    parts.append('## Comparaison LFP (Fourier) vs STFT/CWT')
    parts.append('Sources: `outputs/fourier/phase_monthly/ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv`, `outputs/fourier/wavelets/WAVELET_STFT_MONTHLY_METRICS.csv`')
    parts.append('')
    parts.append('### Méthodes')
    parts.append('- STFT: fenêtre Hann nperseg=128, chevauchement=96; période=1/f, exclusion DC/Nyquist; bande LF: périodes ∈ [64, 4096] barres, normalisation par tranche temporelle.')
    parts.append("- CWT: ondelette Morlet (pywt), num_scales=96; conversion période=1/f avec f=scale2frequency('morl')*fs; bande LF identique [64, 4096] barres.")
    parts.append('- Agrégations mensuelles: médiane des périodes dominantes; moyenne du ratio LF (par tranche).')
    parts.append('')

    for sym in ['BTC_USDT','BTC_USD']:
        for tf in ['2h','1d']:
            sub = merged[(merged['symbol'] == sym) & (merged['timeframe'] == tf)].copy()
            if sub.empty:
                continue
            parts.append(f"### {sym} {tf}")
            # correlations
            corr_cols = ['LFP_mean','STFT_LFP_like','CWT_LFP_like','STFT_domP_bars','CWT_domScale_bars','P1_bars_median']
            corr_df = sub[corr_cols].dropna()
            corr_val = corr_df.corr(method='pearson')
            parts.append('Corrélations (Pearson)')
            parts.append(to_table(corr_val.reset_index().rename(columns={'index':'metric'}), ['metric'] + corr_cols, max_rows=20))
            parts.append('')
            # sample rows
            parts.append('Extraits (mois)')
            parts.append(to_table(sub.sort_values('month'), ['month','LFP_mean','STFT_LFP_like','CWT_LFP_like','P1_bars_median','STFT_domP_bars','CWT_domScale_bars'], max_rows=20))
            parts.append('')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(parts), encoding='utf-8')
    print('Wrote:', OUT_MD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


