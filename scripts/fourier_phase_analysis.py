#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-stratified analysis of Fourier features (P1/P2/P3/LFP, price and volume).

Steps:
- Load DAILY_SUMMARY_* and OHLCV CSVs
- Compute halving-aligned phases (using phase_aware_module)
- Resample to daily and join with Fourier daily summaries
- Aggregate by phase; write tables and plots
- Export scheduler JSON mapping P1→Ichimoku ranges per phase
- Calibrate LFP/LFP_vol gating thresholds

Outputs:
- outputs/fourier/phase/<symbol>_<tf>/PHASE_STATS_<symbol>_<tf>.csv
- outputs/fourier/phase/<symbol>_<tf>/*.png (bar plots)
- outputs/fourier/phase/<symbol>_<tf>/SCHEDULER_FOURIER_<symbol>_<tf>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.phase_aware_module import phase_snapshot


OHLCV_PATHS: Dict[Tuple[str,str], Path] = {
    ("BTC_USDT","2h"): Path('data') / 'BTC_USDT_2h.csv',
    ("BTC_USDT","1d"): Path('data') / 'BTC_USDT_1d.csv',
    ("BTC_USD","2h"): Path('data') / 'BTC_USD_2h.csv',
    ("BTC_USD","1d"): Path('data') / 'BTC_USD_1d.csv',
}


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']]


def load_daily_summary(root: Path, sym: str, tf: str) -> pd.DataFrame:
    p = root / f'DAILY_SUMMARY_{sym}_{tf}.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df


def summarize_by_phase(daily: pd.DataFrame, feats_daily: pd.DataFrame) -> pd.DataFrame:
    joined = daily.join(feats_daily[['phase']], how='inner')
    # Aggregate by phase
    agg = joined.groupby('phase').agg({
        'P1_bars':'median','P2_bars':'median','P3_bars':'median','LFP':'mean',
        'P1_vol':'median','P2_vol':'median','P3_vol':'median','LFP_vol':'mean'
    }).reset_index()
    return agg


def plot_phase_bars(agg: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # P1/P2/P3 price
    fig, ax = plt.subplots(figsize=(8,4))
    phases = agg['phase']
    w = 0.25
    x = np.arange(len(phases))
    ax.bar(x - w, agg['P1_bars'], width=w, label='P1')
    ax.bar(x,       agg['P2_bars'], width=w, label='P2')
    ax.bar(x + w,   agg['P3_bars'], width=w, label='P3')
    ax.set_xticks(x); ax.set_xticklabels(phases, rotation=20)
    ax.set_title(f"{title_prefix} — P1/P2/P3 (prix)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / 'phase_Pbars.png', dpi=140); plt.close(fig)

    # LFP price vs volume
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(phases, agg['LFP'], label='LFP prix')
    ax.bar(phases, agg['LFP_vol'], alpha=0.6, label='LFP vol')
    ax.set_title(f"{title_prefix} — LFP prix vs volume")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / 'phase_LFP.png', dpi=140); plt.close(fig)


def ichimoku_ranges_from_P(P: float) -> Dict[str, Tuple[int,int]]:
    # Simple ranges around heuristics (±20%) and minimal bounds
    if not np.isfinite(P):
        return {
            'tenkan': (6, 12),
            'kijun': (26, 55),
            'senkou_b': (52, 120),
            'shift': (26, 30),
        }
    tenkan_c = max(6.0, P/8.0)
    kijun_c = max(26.0, P/2.0)
    senkou_c = max(52.0, P)
    def rng(c: float, pct: float, mn: int) -> Tuple[int,int]:
        a = int(round(max(mn, c*(1-pct))))
        b = int(round(max(mn, c*(1+pct))))
        return (min(a,b), max(a,b))
    return {
        'tenkan': rng(tenkan_c, 0.2, 6),
        'kijun': rng(kijun_c, 0.2, 26),
        'senkou_b': rng(senkou_c, 0.2, 52),
        'shift': (26, 30),
    }


def gating_thresholds_from_lfp(agg: pd.DataFrame) -> Dict[str, float]:
    # Use overall distribution to set low/medium/high thresholds (quantiles)
    vals = agg['LFP'].dropna().values
    if len(vals) < 5:
        return {'low': 0.80, 'medium': 0.85, 'high': 0.90}
    q = np.quantile(vals, [0.33, 0.66, 0.80])
    return {'low': float(q[0]), 'medium': float(q[1]), 'high': float(q[2])}


def main() -> int:
    ap = argparse.ArgumentParser(description='Fourier phase-stratified analysis')
    ap.add_argument('--symbol', default='BTC_USDT')
    ap.add_argument('--timeframe', default='2h')
    args = ap.parse_args()

    sym, tf = args.symbol, args.timeframe
    root = Path('outputs') / 'fourier'
    out_phase = root / 'phase' / f'{sym}_{tf}'
    out_phase.mkdir(parents=True, exist_ok=True)

    daily = load_daily_summary(root, sym, tf)
    ohlcv_path = OHLCV_PATHS.get((sym, tf))
    if ohlcv_path is None or not ohlcv_path.exists():
        raise FileNotFoundError(f'No OHLCV for {(sym, tf)}')
    df = read_ohlcv(ohlcv_path)

    feats = phase_snapshot(df)

    def _lowpass_subsample(df: pd.DataFrame, q: int) -> pd.DataFrame:
        """Low-pass filter then sub-sample by q using a FIR filter."""
        fs = 12.0  # 2h bars → 12 samples/day
        nyq = fs / 2.0
        b = signal.firwin(101, 0.4 / nyq)
        filt = {c: signal.filtfilt(b, [1.0], df[c].to_numpy()) for c in df.columns}
        df_filt = pd.DataFrame(filt, index=df.index)
        return df_filt.iloc[::q]

    feats_filt = _lowpass_subsample(feats, 12)
    feats_daily = feats_filt.resample('1D').last().dropna(subset=['phase'])

    agg = summarize_by_phase(daily, feats_daily)
    agg_path = out_phase / f'PHASE_STATS_{sym}_{tf}.csv'
    agg.to_csv(agg_path, index=False)

    plot_phase_bars(agg, out_phase, f'{sym} {tf}')

    # Scheduler JSON: ranges per phase using median P1
    sched = {}
    for _, row in agg.iterrows():
        phase = str(row['phase'])
        P1 = float(row['P1_bars']) if pd.notna(row['P1_bars']) else np.nan
        sched[phase] = ichimoku_ranges_from_P(P1)
    # Gating thresholds
    sched['gating_lfp'] = gating_thresholds_from_lfp(agg)

    with open(out_phase / f'SCHEDULER_FOURIER_{sym}_{tf}.json', 'w', encoding='utf-8') as f:
        json.dump(sched, f, ensure_ascii=False, indent=2)

    print('Wrote:', agg_path)
    print('Wrote:', out_phase / f'SCHEDULER_FOURIER_{sym}_{tf}.json')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


