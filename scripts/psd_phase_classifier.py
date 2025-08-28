#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch, get_window  # type: ignore[reportMissingImports]

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_phase_price import read_ohlcv  # type: ignore


PAIRS_TF: List[Tuple[str, str]] = [
    ("BTC_USDT", "2h"),
    ("BTC_USD", "2h"),
]

# Bands in cycles/day for H2 (fs=12/day, Nyquist=6/day)
BANDS: Dict[str, Tuple[float, float]] = {
    "accumulation": (0.00, 0.10),
    "bear":         (0.10, 0.30),
    "distribution": (0.30, 1.00),
    "expansion":    (1.00, 3.00),
    "euphoria":     (3.00, 6.00),
}


@dataclass
class PSDParams:
    fs: float = 12.0            # samples per day (2h bars)
    nperseg: int = 84           # ~7 days
    noverlap: float = 0.8       # 80% overlap
    window: str = "hann"
    scaling: str = "density"
    smooth_k: int = 3           # MA over windows
    margin: float = 0.07        # 7 percentage points
    min_persist_windows: int = 2


def _log_returns(close: pd.Series) -> pd.Series:
    rt = np.log(close).diff()
    return rt - rt.mean()


def _welch_psd_segment(x: np.ndarray, params: PSDParams) -> Tuple[np.ndarray, np.ndarray]:
    w = get_window(params.window, params.nperseg, fftbins=True)
    f, Pxx = welch(
        x,
        fs=params.fs,
        window=w,
        nperseg=params.nperseg,
        noverlap=int(params.noverlap * params.nperseg),
        detrend='constant',
        scaling=params.scaling,
        return_onesided=True,
    )
    return f, Pxx


def _band_energy(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _band_shares(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    energies: Dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        energies[name] = _band_energy(freqs, psd, lo, hi)
    total = float(sum(energies.values()))
    if total <= 0:
        return {k: 0.0 for k in energies}
    return {k: (v / total) for k, v in energies.items()}


def _winner_with_guards(shares_hist: List[Dict[str, float]], params: PSDParams, prev_label: Optional[str]) -> str:
    cur = shares_hist[-1]
    # smoothing over last K windows
    if len(shares_hist) >= params.smooth_k:
        keys = list(cur.keys())
        avg = {k: float(np.mean([h[k] for h in shares_hist[-params.smooth_k:]])) for k in keys}
    else:
        avg = cur
    # winner and second
    ranked = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_val = ranked[0]
    second_val = ranked[1][1] if len(ranked) > 1 else 0.0
    if (top_val - second_val) < params.margin and prev_label is not None:
        return prev_label
    return top_label


def classify_psd_h2(close: pd.Series, params: PSDParams) -> pd.Series:
    rt = _log_returns(close).dropna()
    n = len(rt)
    if n < params.nperseg:
        return pd.Series(index=close.index, dtype=str)
    step = max(1, int(round(params.nperseg * (1.0 - params.noverlap))))
    labels_at: List[Tuple[pd.Timestamp, str]] = []
    shares_hist: List[Dict[str, float]] = []
    prev_label: Optional[str] = None
    persist_win = 0
    idx = rt.index
    # iterate windows
    for end_i in range(params.nperseg, n + 1, step):
        seg = rt.iloc[end_i - params.nperseg: end_i].values.astype(float)
        f, Pxx = _welch_psd_segment(seg, params)
        shares = _band_shares(f, Pxx)
        shares_hist.append(shares)
        # tentative winner with guards
        cand = _winner_with_guards(shares_hist, params, prev_label)
        if prev_label is None or cand == prev_label:
            persist_win += 1
            prev_label = cand
        else:
            if persist_win >= params.min_persist_windows:
                prev_label = cand
                persist_win = 1
            else:
                persist_win += 1  # keep previous label
        ts = idx[end_i - 1]
        labels_at.append((ts, prev_label))
    # build per-bar series via forward-fill
    lab_series = pd.Series({ts: lab for ts, lab in labels_at}).sort_index()
    lab_series = lab_series.reindex(close.index, method='ffill')
    return lab_series.astype(str)


def _segments_from_labels(index: pd.DatetimeIndex, labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    s = labels.reindex(index, method='ffill').astype(str)
    if s.empty:
        return []
    out: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    start = s.index[0]
    cur = s.iloc[0]
    for ts, v in zip(s.index[1:], s.iloc[1:]):
        if v != cur:
            out.append((start, ts, cur))
            start = ts
            cur = v
    out.append((start, s.index[-1], cur))
    return out


def _month_ranges(index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    idx = index.sort_values()
    if idx.empty:
        return []
    cur = pd.Timestamp(year=idx[0].year, month=idx[0].month, day=1)
    end = idx[-1]
    out: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    while cur <= end:
        last = pd.Timestamp(year=cur.year, month=cur.month, day=1) + pd.offsets.MonthEnd(0)
        ym = cur.strftime('%Y-%m')
        out.append((cur, min(last, end), ym))
        cur = (last + pd.offsets.Day(1)).normalize()
    return out


def _write_month_segments(sym: str, tf: str, ym: str, month_idx: pd.DatetimeIndex, segs: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> None:
    y = ym.split('-')[0]
    out_dir = Path('outputs') / 'fourier' / 'phase_monthly' / sym / tf / y / ym
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for start, end, ph in segs:
        s = max(start, month_idx.min())
        e = min(end, month_idx.max())
        sli = month_idx[(month_idx >= s) & (month_idx <= e)]
        if len(sli) == 0:
            continue
        d1 = (e.floor('D') - s.floor('D')).days + 1
        h2 = len(sli)
        rows.append({
            'phase': ph,
            'start': s.strftime('%Y-%m-%d %H:%M:%S'),
            'end': e.strftime('%Y-%m-%d %H:%M:%S'),
            'd1_candles': int(d1),
            'h2_bars': int(h2),
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / 'PHASE5_PSD_SEGMENTS.csv', index=False)


def _aggregate_monthly(sym: str, tf: str) -> pd.DataFrame:
    base = Path('outputs') / 'fourier' / 'phase_monthly' / sym / tf
    rows: List[dict] = []
    if not base.exists():
        return pd.DataFrame()
    for ydir in sorted(d for d in base.iterdir() if d.is_dir()):
        for mdir in sorted(d for d in ydir.iterdir() if d.is_dir()):
            seg_csv = mdir / 'PHASE5_PSD_SEGMENTS.csv'
            if not seg_csv.exists():
                continue
            ym = mdir.name
            seg = pd.read_csv(seg_csv)
            grp = seg.groupby('phase').agg({'d1_candles': 'sum', 'h2_bars': 'sum'}).reset_index()
            row = {'month': ym, 'symbol': sym, 'timeframe': tf}
            for ph in BANDS.keys():
                ph_row = grp[grp['phase'] == ph]
                d_val = float(ph_row['d1_candles'].iloc[0]) if not ph_row.empty else 0.0
                h_val = float(ph_row['h2_bars'].iloc[0]) if not ph_row.empty else 0.0
                row[f'phase5psd_{ph}_days'] = int(d_val)
                row[f'phase5psd_{ph}_h2_bars'] = int(h_val)
            rows.append(row)
    return pd.DataFrame(rows)


def _global_means(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (sym, tf), sub in monthly.groupby(['symbol','timeframe']):
        for ph in BANDS.keys():
            h2_col = f'phase5psd_{ph}_h2_bars'
            d_col = f'phase5psd_{ph}_days'
            if h2_col not in sub.columns or d_col not in sub.columns:
                continue
            h2_mean = pd.to_numeric(sub[h2_col], errors='coerce').fillna(0).mean()
            d_mean = pd.to_numeric(sub[d_col], errors='coerce').fillna(0).mean()
            rows.append({
                'symbol': sym,
                'timeframe': tf,
                'phase': ph,
                'mean_h2_bars_per_month': round(float(h2_mean), 2),
                'mean_days_per_month': round(float(d_mean), 2),
            })
    return pd.DataFrame(rows).sort_values(['symbol','timeframe','phase'])


def main() -> int:
    params = PSDParams()
    current_rows: List[dict] = []
    for sym, tf in PAIRS_TF:
        path = Path('data') / f'{sym}_{tf}.csv'
        if not path.exists():
            continue
        df = read_ohlcv(path)
        close = df['close']
        labels = classify_psd_h2(close, params=params)
        if labels.empty:
            continue
        # current label
        current_rows.append({'symbol': sym, 'timeframe': tf, 'current_phase5_psd': str(labels.iloc[-1])})
        # write per-month segments
        segs_all = _segments_from_labels(df.index, labels)
        # split by month
        for m_start, m_end, ym in _month_ranges(df.index):
            month_idx = df.index[(df.index >= m_start) & (df.index <= m_end)]
            if len(month_idx) == 0:
                continue
            # clip segments to month
            segs_month: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
            for s, e, ph in segs_all:
                if e < m_start or s > m_end:
                    continue
                segs_month.append((max(s, m_start), min(e, m_end), ph))
            _write_month_segments(sym, tf, ym, month_idx, segs_month)

    # aggregate monthly durations
    frames: List[pd.DataFrame] = []
    for sym, tf in PAIRS_TF:
        agg = _aggregate_monthly(sym, tf)
        if not agg.empty:
            frames.append(agg)
    if frames:
        monthly = pd.concat(frames, ignore_index=True)
        monthly = monthly.sort_values(['symbol','timeframe','month'])
        out_dir = Path('outputs') / 'fourier' / 'phase_monthly'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_monthly = out_dir / 'PSD_PHASE5_MONTHLY_DURATIONS.csv'
        monthly.to_csv(out_monthly, index=False)
        print('Wrote:', out_monthly)
        means = _global_means(monthly)
        out_means = out_dir / 'PSD_PHASE5_GLOBAL_MEANS.csv'
        means.to_csv(out_means, index=False)
        print('Wrote:', out_means)

    # current label report
    if current_rows:
        cur_df = pd.DataFrame(current_rows)
        cur_dir = Path('docs') / 'PHASE_LABELS'
        cur_dir.mkdir(parents=True, exist_ok=True)
        (cur_dir / 'CURRENT_PHASE5_PSD.csv').write_text(cur_df.to_csv(index=False), encoding='utf-8')
        try:
            md = cur_df.to_markdown(index=False)
        except Exception:
            md = ''
        (cur_dir / 'CURRENT_PHASE5_PSD.md').write_text('\n'.join(['### Phase actuelle (phase5 PSD) par paire/timeframe', '', md]), encoding='utf-8')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


