#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Local imports
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.phase_aware_module import phase_snapshot  # type: ignore


OHLCV_PATHS: Dict[Tuple[str,str], Path] = {
    ("BTC_USDT","2h"): Path('data') / 'BTC_USDT_2h.csv',
    ("BTC_USDT","1d"): Path('data') / 'BTC_USDT_1d.csv',
    ("BTC_USD","2h"): Path('data') / 'BTC_USD_2h.csv',
    ("BTC_USD","1d"): Path('data') / 'BTC_USD_1d.csv',
}


def bars_per_day(timeframe: str) -> int:
    if timeframe.endswith('h'):
        h = float(timeframe[:-1])
        return max(1, int(round(24.0 / h)))
    return 1


def read_ohlcv(sym: str, tf: str) -> pd.DataFrame:
    p = OHLCV_PATHS.get((sym, tf))
    if p is None or not p.exists():
        raise FileNotFoundError(f"Missing OHLCV: {(sym, tf)}")
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']]


def load_daily_summary(sym: str, tf: str) -> pd.DataFrame:
    p = Path('outputs') / 'fourier' / f'DAILY_SUMMARY_{sym}_{tf}.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df


def label_3_regimes(M: float, LFP: float, m_up: float = 0.05, m_down: float = -0.05, lfp_trend: float = 0.80) -> str:
    if np.isnan(M) or np.isnan(LFP):
        return 'unknown'
    if (M >= m_up) and (LFP >= lfp_trend):
        return 'up'
    if (M <= m_down) and (LFP >= lfp_trend):
        return 'down'
    return 'range'


def label_6_phases(phase5: str, M: float, V_ann: float, DD: float) -> str:
    if not (np.isnan(M) or np.isnan(V_ann) or np.isnan(DD)):
        if (M <= -0.15) and (V_ann >= 1.0) and (DD <= -0.50):
            return 'capitulation'
    return phase5


def segment_durations(labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, int]]:
    # Returns list of (start, end, label, length_bars)
    segs: List[Tuple[pd.Timestamp, pd.Timestamp, str, int]] = []
    if labels.empty:
        return segs
    start = labels.index[0]
    prev = labels.iloc[0]
    count = 1
    for t, val in labels.iloc[1:].items():
        if val == prev:
            count += 1
        else:
            segs.append((start, t, prev, count))
            start = t
            prev = val
            count = 1
    segs.append((start, labels.index[-1], prev, count))
    return segs


def summarize(sym: str, tf: str) -> None:
    df = read_ohlcv(sym, tf)
    feats = phase_snapshot(df)  # includes M, V_ann, DD, last_halving, days_since_halving
    # Align feats to OHLCV index
    feats = feats.reindex(df.index).ffill()
    # Daily Fourier summary for P/LFP
    daily = load_daily_summary(sym, tf)
    # Align daily to OHLCV index by forward-fill
    daily_aligned = daily.reindex(df.index, method='ffill')

    # Labels
    l3 = pd.Series([label_3_regimes(m, l) for m, l in zip(feats['M'], daily_aligned['LFP'])], index=df.index, name='regime3')
    l5 = feats['phase'].rename('phase5')
    l6 = pd.Series([label_6_phases(p5, m, v, dd) for p5, m, v, dd in zip(l5, feats['M'], feats['V_ann'], feats['DD'])], index=df.index, name='phase6')

    # Segments and stats
    bpd = bars_per_day(tf)
    out_dir = Path('outputs') / 'fourier' / 'compare' / f'{sym}_{tf}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in [('regime3', l3), ('phase5', l5), ('phase6', l6)]:
        segs = segment_durations(s)
        rows = []
        for start, end, label, length_bars in segs:
            if label in ('unknown', np.nan, None):
                continue
            # Slice
            sli = df.loc[(df.index >= start) & (df.index <= end)]
            if sli.empty:
                continue
            days = max(1, (end - start).days)
            h2_bars = length_bars if tf == '2h' else int(length_bars * (24/2))
            # Fourier stats in segment (using daily_aligned)
            sld = daily_aligned.loc[sli.index]
            # Distance au halving (début/médiane/fin)
            dsh = feats.loc[sli.index, 'days_since_halving']
            dsh_start = float(dsh.iloc[0]) if not dsh.empty else np.nan
            dsh_end = float(dsh.iloc[-1]) if not dsh.empty else np.nan
            dsh_mid = float(dsh.median(skipna=True)) if not dsh.empty else np.nan
            stats = {
                'P1_med': sld['P1_bars'].median(skipna=True),
                'P2_med': sld['P2_bars'].median(skipna=True),
                'P3_med': sld['P3_bars'].median(skipna=True),
                'P4_med': sld['P4_bars'].median(skipna=True) if 'P4_bars' in sld.columns else np.nan,
                'P5_med': sld['P5_bars'].median(skipna=True) if 'P5_bars' in sld.columns else np.nan,
                'P6_med': sld['P6_bars'].median(skipna=True) if 'P6_bars' in sld.columns else np.nan,
                'LFP_mean': sld['LFP'].mean(skipna=True),
                'days_since_halving_start': dsh_start,
                'days_since_halving_mid': dsh_mid,
                'days_since_halving_end': dsh_end,
            }
            rows.append({
                'labelset': name,
                'label': label,
                'start': start,
                'end': end,
                'duration_days': days,
                'duration_bars': length_bars,
                'duration_h2_bars': h2_bars,
                **stats,
            })
        seg_df = pd.DataFrame(rows)
        seg_path = out_dir / f'SEGMENTS_{name}.csv'
        seg_df.to_csv(seg_path, index=False)
        # Aggregate by label
        if not seg_df.empty:
            agg = seg_df.groupby('label').agg({
                'duration_days':'median',
                'duration_h2_bars':'median',
                'days_since_halving_mid':'median',
                'P1_med':'median','P2_med':'median','P3_med':'median','P4_med':'median','P5_med':'median','P6_med':'median','LFP_mean':'mean'
            }).reset_index().rename(columns={'days_since_halving_mid':'dsh_mid_median'})
            agg_path = out_dir / f'SUMMARY_{name}.csv'
            agg.to_csv(agg_path, index=False)
            print('Wrote:', seg_path)
            print('Wrote:', agg_path)


def main() -> int:
    ap = argparse.ArgumentParser(description='Comparative analysis of 3/5/6 labels vs time since halving and durations')
    ap.add_argument('--symbol', default='BTC_USDT')
    ap.add_argument('--timeframe', default='2h')
    args = ap.parse_args()
    summarize(args.symbol, args.timeframe)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


