#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import calendar
from pathlib import Path
from typing import List, Tuple
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_phase_price import read_ohlcv  # type: ignore
from scripts.phase_aware_module import phase_snapshot  # type: ignore


PAIRS_TF: List[Tuple[str, str]] = [
    ("BTC_USDT", "2h"),
    ("BTC_USDT", "1d"),
    ("BTC_USD", "2h"),
    ("BTC_USD", "1d"),
]


def iter_month_ranges(index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    idx = index.sort_values()
    if idx.empty:
        return []
    start = pd.Timestamp(year=idx[0].year, month=idx[0].month, day=1)
    end = idx[-1]
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        m_start = cur
        m_end = pd.Timestamp(year=cur.year, month=cur.month, day=last_day, hour=23, minute=59, second=59)
        ranges.append((m_start, min(m_end, end)))
        # next month
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1)
    return ranges


def segment_series(index: pd.DatetimeIndex, labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if index.empty:
        return segments
    s = labels.reindex(index, method='ffill').astype(str)
    start = s.index[0]
    cur = s.iloc[0]
    for ts, v in zip(s.index[1:], s.iloc[1:]):
        if v != cur:
            segments.append((start, ts, cur))
            start = ts
            cur = v
    segments.append((start, s.index[-1], cur))
    return segments


def _daily_candle_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.floor('D') - start.floor('D')).days + 1


def build_monthly_segments(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, str]:
    ohlcv_path = Path('data') / f"{symbol}_{timeframe}.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(ohlcv_path)

    df = read_ohlcv(ohlcv_path)
    feats = phase_snapshot(df).reindex(df.index).ffill()
    phase5 = feats['phase'].astype(str)

    all_rows: List[dict] = []
    current_phase: str = ''
    for m_start, m_end in iter_month_ranges(df.index):
        sli = df.loc[(df.index >= m_start) & (df.index <= m_end)]
        if len(sli) < 2:
            continue
        segs = segment_series(sli.index, phase5)
        ym = m_start.strftime('%Y-%m')
        for start, end, ph in segs:
            # Clip to month window
            s = max(start, m_start)
            e = min(end, m_end)
            sli_seg = df.loc[(df.index >= s) & (df.index <= e)]
            if sli_seg.empty:
                continue
            d1 = _daily_candle_count(s, e)
            h2 = len(sli_seg) if timeframe == '2h' else d1 * 12
            all_rows.append({
                'month': ym,
                'symbol': symbol,
                'timeframe': timeframe,
                'phase': ph,
                'start': s.strftime('%Y-%m-%d %H:%M:%S'),
                'end': e.strftime('%Y-%m-%d %H:%M:%S'),
                'd1_candles': int(d1),
                'h2_bars': int(h2),
            })
        # phase at end of month
        if not segs:
            continue
        current_phase = segs[-1][2]

    seg_df = pd.DataFrame(all_rows)
    return seg_df, current_phase


def write_per_month(seg_df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    # Write per-month CSVs under the existing monthly report directories
    for ym, sub in seg_df.groupby('month'):
        y = ym.split('-')[0]
        out_dir = Path('outputs') / 'fourier' / 'phase_monthly' / symbol / timeframe / y / ym
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'PHASE5_SEGMENTS.csv'
        sub[['phase','start','end','d1_candles','h2_bars']].to_csv(csv_path, index=False)
        # Also write a small markdown snippet for convenience
        md_path = out_dir / 'SEGMENTS.md'
        try:
            md = sub[['phase','start','end','d1_candles','h2_bars']].to_markdown(index=False)
        except Exception:
            md = ''
        md_lines = [f"### Segments phase5 — {symbol} {timeframe} — {ym}", '', md]
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')


def main() -> int:
    current_rows: List[dict] = []
    all_concat: List[pd.DataFrame] = []
    for sym, tf in PAIRS_TF:
        try:
            seg_df, cur = build_monthly_segments(sym, tf)
        except FileNotFoundError:
            continue
        if not seg_df.empty:
            all_concat.append(seg_df)
            write_per_month(seg_df, sym, tf)
        if cur:
            current_rows.append({'symbol': sym, 'timeframe': tf, 'current_phase5': cur})

    if all_concat:
        out_dir = Path('outputs') / 'fourier' / 'phase_monthly'
        out_dir.mkdir(parents=True, exist_ok=True)
        seg_all = pd.concat(all_concat, ignore_index=True)
        seg_all = seg_all.sort_values(['symbol','timeframe','month','start'])
        seg_all.to_csv(out_dir / 'ALL_PHASE5_SEGMENTS.csv', index=False)

    if current_rows:
        cur_df = pd.DataFrame(current_rows)
        cur_df.to_csv(Path('docs') / 'PHASE_LABELS' / 'CURRENT_PHASE5.csv', index=False)
        # Lightweight MD for quick viewing
        md_lines = [
            '### Phase actuelle (phase5) par paire/timeframe',
            '',
            cur_df.to_markdown(index=False),
        ]
        (Path('docs') / 'PHASE_LABELS' / 'CURRENT_PHASE5.md').write_text('\n'.join(md_lines), encoding='utf-8')

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


