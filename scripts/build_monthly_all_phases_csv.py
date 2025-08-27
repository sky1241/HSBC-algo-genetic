#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_phase_price import (
    read_ohlcv,
    load_daily_summary,
    label_3_regimes,
    label_6_from_5,
)
from scripts.phase_aware_module import phase_snapshot


PAIRS_TF: List[Tuple[str, str]] = [
    ("BTC_USDT", "2h"),
    ("BTC_USDT", "1d"),
    ("BTC_USD", "2h"),
    ("BTC_USD", "1d"),
]

PHASE5_KEYS: List[str] = [
    "accumulation", "expansion", "euphoria", "distribution", "bear"
]
PHASE6_KEYS: List[str] = PHASE5_KEYS + ["capitulation"]


def _counts_to_durations(counts: pd.DataFrame, tf: str, col_prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=counts.index).sort_index()
    # Days/H2 bars conversion
    if tf == '2h':
        # counts are in 2h candles; 12 bars = 1 day
        for col in counts.columns:
            out[f"{col_prefix}_{col}_h2_bars"] = counts[col].astype(int)
            out[f"{col_prefix}_{col}_days"] = counts[col] / 12.0
    else:
        # counts are in daily candles
        for col in counts.columns:
            out[f"{col_prefix}_{col}_days"] = counts[col].astype(int)
            out[f"{col_prefix}_{col}_h2_bars"] = counts[col] * 12
    # Shares (over total bars)
    h2_cols = [c for c in out.columns if c.endswith('_h2_bars')]
    out[f"{col_prefix}_h2_bars_total"] = out[h2_cols].sum(axis=1)
    for col in counts.columns:
        out[f"{col_prefix}_{col}_share"] = (
            out[f"{col_prefix}_{col}_h2_bars"] / out[f"{col_prefix}_h2_bars_total"]
        ).where(out[f"{col_prefix}_h2_bars_total"] > 0)
    return out


def aggregate_monthly_all(symbol: str, timeframe: str) -> pd.DataFrame:
    ohlcv_path = Path('data') / f"{symbol}_{timeframe}.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(ohlcv_path)

    df = read_ohlcv(ohlcv_path)
    feats = phase_snapshot(df).reindex(df.index).ffill()
    # Daily Fourier summary for regime3 LFP
    daily = load_daily_summary(symbol, timeframe)
    daily_aligned = daily.reindex(df.index, method='ffill')

    # 3-regimes labels (up/down/range)
    reg3 = pd.Series(
        [label_3_regimes(m, lfp) for m, lfp in zip(feats['M'], daily_aligned['LFP'])],
        index=df.index,
    ).astype(str)
    # 5-phase labels
    ph5 = feats['phase'].astype(str)
    # 6-phase labels from 5-phase + conditions
    ph6 = pd.Series(
        [label_6_from_5(p5, m, v, dd) for p5, m, v, dd in zip(feats['phase'], feats['M'], feats['V_ann'], feats['DD'])],
        index=feats.index,
    ).astype(str)

    # Month index
    mon_idx = df.index.to_period('M').to_timestamp()

    # Counts per month for each label set
    df_reg3 = pd.DataFrame({'month': mon_idx, 'lab': reg3})
    reg3_counts = df_reg3.pivot_table(index='month', columns='lab', values='lab', aggfunc='count').fillna(0)
    for k in ['up','down','range']:
        if k not in reg3_counts.columns:
            reg3_counts[k] = 0
    reg3_counts = reg3_counts[['up','down','range']]
    reg3_dur = _counts_to_durations(reg3_counts, timeframe, 'regime3')

    df_ph5 = pd.DataFrame({'month': mon_idx, 'lab': ph5})
    ph5_counts = df_ph5.pivot_table(index='month', columns='lab', values='lab', aggfunc='count').fillna(0)
    for k in PHASE5_KEYS:
        if k not in ph5_counts.columns:
            ph5_counts[k] = 0
    ph5_counts = ph5_counts[PHASE5_KEYS]
    ph5_dur = _counts_to_durations(ph5_counts, timeframe, 'phase5')

    df_ph6 = pd.DataFrame({'month': mon_idx, 'lab': ph6})
    ph6_counts = df_ph6.pivot_table(index='month', columns='lab', values='lab', aggfunc='count').fillna(0)
    for k in PHASE6_KEYS:
        if k not in ph6_counts.columns:
            ph6_counts[k] = 0
    ph6_counts = ph6_counts[PHASE6_KEYS]
    ph6_dur = _counts_to_durations(ph6_counts, timeframe, 'phase6')

    # Fourier monthly aggregates
    daily_m = daily.copy()
    daily_m['month'] = daily_m.index.to_period('M').to_timestamp()
    agg = daily_m.groupby('month').agg({
        'LFP': ['mean','median'],
        'LFP_vol': ['mean','median'],
        'P1_bars': 'median', 'P2_bars': 'median', 'P3_bars': 'median',
        'P4_bars': 'median', 'P5_bars': 'median', 'P6_bars': 'median',
        'P1_vol': 'median', 'P2_vol': 'median', 'P3_vol': 'median',
    })
    agg.columns = ['_'.join([c for c in col if c]).strip('_') for col in agg.columns.values]

    # Merge
    out = (
        reg3_dur.join(ph5_dur, how='outer')
                .join(ph6_dur, how='outer')
                .join(agg, how='left')
                .reset_index()
    )
    out['symbol'] = symbol
    out['timeframe'] = timeframe
    out['month'] = pd.to_datetime(out['month']).dt.strftime('%Y-%m')
    # Order columns
    cols_front = ['month','symbol','timeframe']
    other_cols = [c for c in out.columns if c not in cols_front]
    out = out[cols_front + other_cols]
    return out


def main() -> int:
    frames: List[pd.DataFrame] = []
    for sym, tf in PAIRS_TF:
        try:
            frames.append(aggregate_monthly_all(sym, tf))
        except FileNotFoundError:
            continue
    if not frames:
        print('No data found.')
        return 0
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(['month','symbol','timeframe'])
    out_dir = Path('outputs') / 'fourier' / 'phase_monthly'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'ALL_PHASESETS_MONTHLY_WITH_FOURIER.csv'
    full.to_csv(out_csv, index=False)
    print('Wrote:', out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


