#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule et met en cache les signaux Fourier (annuels et mensuels) à partir d'un CSV OHLCV H2.
Exports:
  - CSV annuel: outputs/fourier/FREQ_ANNUAL_<symbol>_<tf>.csv
  - CSV mensuel: outputs/fourier/FREQ_MONTHLY_<symbol>_<tf>.csv
  - Graphiques: outputs/fourier/plots/<...>.png (spectres & timelines P/LFP)

Usage:
  py -3 scripts/fourier_signals_batch.py --csv data/BTC_USDT_2h.csv --symbol BTC/USDT --timeframe 2h
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from scipy.signal import welch as scipy_welch  # type: ignore
except Exception:
    scipy_welch = None


def _periodogram_psd(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    if n < 8:
        return np.array([]), np.array([])
    w = np.hanning(n)
    xw = np.where(np.isnan(x), 0.0, x) * w
    fft = np.fft.rfft(xw)
    psd = (np.abs(fft) ** 2) / np.sum(w**2)
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs, psd


def _welch_psd(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if scipy_welch is None or len(x) < 32:
        return _periodogram_psd(x)
    nperseg = max(64, min(1024, len(x) // 4))
    f, pxx = scipy_welch(x, fs=1.0, nperseg=nperseg)
    return f, pxx


def _dominant_period(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    seg = close.iloc[-window:]
    f, psd = _welch_psd(seg.to_numpy())
    if len(f) == 0:
        return None
    idx = int(np.nanargmax(psd[1:]) + 1) if len(psd) > 1 else int(np.nanargmax(psd))
    fstar = float(f[idx])
    return (1.0 / fstar) if fstar > 0 else None


def _top_k_periods_windowed(close: pd.Series, window: int, k: int = 3, min_rel_separation: float = 0.1) -> list[float | None]:
    if len(close) < window:
        return [None] * k
    seg = close.iloc[-window:]
    f, psd = _welch_psd(seg.to_numpy())
    if len(f) == 0:
        return [None] * k
    # Exclure la composante DC
    f = f[1:]
    psd = psd[1:]
    if len(f) == 0:
        return [None] * k
    periods = 1.0 / np.where(f > 0, f, np.nan)
    order = np.argsort(psd)[::-1]
    selected: list[float] = []
    for idx in order:
        p = float(periods[idx]) if np.isfinite(periods[idx]) else None
        if p is None or p <= 0:
            continue
        too_close = False
        for q in selected:
            if q <= 0:
                continue
            if abs(p - q) / max(q, 1e-9) < min_rel_separation:
                too_close = True
                break
        if not too_close:
            selected.append(p)
        if len(selected) >= k:
            break
    while len(selected) < k:
        selected.append(None)
    return selected


def _lfp(close: pd.Series, window: int, bars_per_day: int, min_days: int = 5) -> float | None:
    if len(close) < window:
        return None
    seg = close.iloc[-window:]
    f, psd = _welch_psd(seg.to_numpy())
    if len(f) == 0:
        return None
    f0 = 1.0 / float(max(1, min_days * bars_per_day))
    num = float(np.nansum(psd[f < f0]))
    den = float(np.nansum(psd))
    return (num / den) if den > 0 else None


def compute_fourier_signals(df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Colonnes attendues: timestamp,open,high,low,close,volume
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
    close = df['close']
    volume = df['volume'] if 'volume' in df.columns else None
    # Bars per day
    if timeframe.endswith('h'):
        h = float(timeframe[:-1])
        bars_per_day = max(1, int(round(24.0 / h)))
    else:
        bars_per_day = 1

    # Fenêtres
    win_annual = int(round(365 * bars_per_day))
    win_month = int(round(30 * bars_per_day))

    # Rolling annual
    annual_rows = []
    for ts in close.index:
        sub = close.loc[:ts]
        P = _dominant_period(sub, win_annual)
        P1, P2, P3 = _top_k_periods_windowed(sub, win_annual, k=3)
        L = _lfp(sub, win_annual, bars_per_day)
        row = {'timestamp': ts, 'P_bars': P1 or P, 'P1_bars': P1 or P, 'P2_bars': P2, 'P3_bars': P3, 'LFP': L}
        if volume is not None:
            sub_v = volume.loc[:ts]
            Pv = _dominant_period(sub_v, win_annual)
            Pv1, Pv2, Pv3 = _top_k_periods_windowed(sub_v, win_annual, k=3)
            Lv = _lfp(sub_v, win_annual, bars_per_day)
            row.update({'P1_vol': Pv1 or Pv, 'P2_vol': Pv2, 'P3_vol': Pv3, 'LFP_vol': Lv})
        annual_rows.append(row)
    df_annual = pd.DataFrame(annual_rows).set_index('timestamp')

    # Rolling monthly
    monthly_rows = []
    for ts in close.index:
        sub = close.loc[:ts]
        P = _dominant_period(sub, win_month)
        P1, P2, P3 = _top_k_periods_windowed(sub, win_month, k=3)
        L = _lfp(sub, win_month, bars_per_day)
        row = {'timestamp': ts, 'P_bars': P1 or P, 'P1_bars': P1 or P, 'P2_bars': P2, 'P3_bars': P3, 'LFP': L}
        if volume is not None:
            sub_v = volume.loc[:ts]
            Pv = _dominant_period(sub_v, win_month)
            Pv1, Pv2, Pv3 = _top_k_periods_windowed(sub_v, win_month, k=3)
            Lv = _lfp(sub_v, win_month, bars_per_day)
            row.update({'P1_vol': Pv1 or Pv, 'P2_vol': Pv2, 'P3_vol': Pv3, 'LFP_vol': Lv})
        monthly_rows.append(row)
    df_month = pd.DataFrame(monthly_rows).set_index('timestamp')

    return df_annual, df_month


def plot_signals(df_annual: pd.DataFrame, df_month: pd.DataFrame, out_dir: Path, symbol: str, timeframe: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Timeline P and LFP
    for name, df in [('annual', df_annual), ('monthly', df_month)]:
        fig, ax = plt.subplots(figsize=(10, 3))
        if 'P1_bars' in df.columns:
            ax.plot(df.index, df['P1_bars'], label='P1', color='#0D47A1', linewidth=1.2)
        if 'P2_bars' in df.columns:
            ax.plot(df.index, df['P2_bars'], label='P2', color='#FF6F00', linewidth=1.0)
        if 'P3_bars' in df.columns:
            ax.plot(df.index, df['P3_bars'], label='P3', color='#2E7D32', linewidth=0.9)
        ax.set_title(f"P1/P2/P3 ({name}) — {symbol} {timeframe}")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol.replace('/','_')}_{timeframe}_P_{name}.png", dpi=120)
        plt.close(fig)

        # Volume cycles if present
        if 'P1_vol' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df.index, df['P1_vol'], label='P1_vol', color='#4A148C', linewidth=1.2)
            if 'P2_vol' in df.columns:
                ax.plot(df.index, df['P2_vol'], label='P2_vol', color='#AD1457', linewidth=1.0)
            if 'P3_vol' in df.columns:
                ax.plot(df.index, df['P3_vol'], label='P3_vol', color='#006064', linewidth=0.9)
            ax.set_title(f"P1/P2/P3 (Volume) {name} — {symbol} {timeframe}")
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_dir / f"{symbol.replace('/','_')}_{timeframe}_PVOL_{name}.png", dpi=120)
            plt.close(fig)

        if 'LFP_vol' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df.index, df['LFP'], label='LFP_price', color='#1B5E20', linewidth=1.2)
            ax.plot(df.index, df['LFP_vol'], label='LFP_vol', color='#6A1B9A', linewidth=1.0)
            ax.set_title(f"LFP prix vs volume ({name}) — {symbol} {timeframe}")
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_dir / f"{symbol.replace('/','_')}_{timeframe}_LFPv_{name}.png", dpi=120)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df['LFP'], label='LFP', color='#1B5E20', linewidth=1.2)
        ax.set_title(f"LFP ({name}) — {symbol} {timeframe}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f"{symbol.replace('/','_')}_{timeframe}_LFP_{name}.png", dpi=120)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Batch Fourier signals (annual/monthly) with CSV + plots")
    p.add_argument('--csv', required=True, help='Path to OHLCV CSV (timestamp,open,high,low,close,volume)')
    p.add_argument('--symbol', default='BTC/USDT')
    p.add_argument('--timeframe', default='2h')
    p.add_argument('--out-dir', default=str(Path('outputs') / 'fourier'))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df_annual, df_month = compute_fourier_signals(df, args.timeframe)

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / 'plots'
    # CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    annual_csv = out_dir / f"FREQ_ANNUAL_{args.symbol.replace('/','_')}_{args.timeframe}.csv"
    monthly_csv = out_dir / f"FREQ_MONTHLY_{args.symbol.replace('/','_')}_{args.timeframe}.csv"
    df_annual.to_csv(annual_csv, index=True)
    df_month.to_csv(monthly_csv, index=True)
    # Plots
    plot_signals(df_annual, df_month, plots_dir, args.symbol, args.timeframe)
    print(f"Saved: {annual_csv}")
    print(f"Saved: {monthly_csv}")
    print(f"Plots: {plots_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


