#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produit un rapport Fourier mensuel (H2) sur toute l'historique disponible:
- Exports CSV par mois (P_bars, LFP) dans outputs/fourier/monthly/YYYY/MM/
- Graphiques P/LFP par mois (PNG), triés du plus ancien au plus récent.

Usage:
  py -3 scripts/fourier_monthly_report.py --csv data/BTC_USDT_2h.csv --symbol BTC/USDT --timeframe 2h
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


def _welch_psd(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if scipy_welch is None or len(x) < 32:
        # fallback periodogram
        n = len(x)
        if n < 8:
            return np.array([]), np.array([])
        w = np.hanning(n)
        xw = np.where(np.isnan(x), 0.0, x) * w
        fft = np.fft.rfft(xw)
        psd = (np.abs(fft) ** 2) / np.sum(w**2)
        freqs = np.fft.rfftfreq(n, d=1.0)
        return freqs, psd
    nperseg = max(64, min(1024, len(x) // 4))
    f, pxx = scipy_welch(x, fs=1.0, nperseg=nperseg)
    return f, pxx


def _dominant_period(series: pd.Series) -> float | None:
    f, psd = _welch_psd(series.to_numpy())
    if len(f) == 0:
        return None
    idx = int(np.nanargmax(psd[1:]) + 1) if len(psd) > 1 else int(np.nanargmax(psd))
    fstar = float(f[idx])
    return (1.0 / fstar) if fstar > 0 else None


def _top_k_periods(series: pd.Series, k: int = 3, min_rel_separation: float = 0.1) -> list[float | None]:
    """
    Retourne jusqu'à k périodes dominantes (en barres) triées par puissance décroissante,
    en écartant les doublons trop proches (min_rel_separation sur la période).
    """
    f, psd = _welch_psd(series.to_numpy())
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
        # Vérifie la séparation relative vs périodes déjà choisies
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
    # Complète avec None si moins de k trouvées
    while len(selected) < k:
        selected.append(None)
    return selected


def _lfp(series: pd.Series, bars_per_day: int, min_days: int = 5) -> float | None:
    f, psd = _welch_psd(series.to_numpy())
    if len(f) == 0:
        return None
    f0 = 1.0 / float(max(1, min_days * bars_per_day))
    num = float(np.nansum(psd[f < f0]))
    den = float(np.nansum(psd))
    return (num / den) if den > 0 else None


def main() -> int:
    p = argparse.ArgumentParser(description="Monthly Fourier report (H2)")
    p.add_argument('--csv', required=True)
    p.add_argument('--symbol', default='BTC/USDT')
    p.add_argument('--timeframe', default='2h')
    p.add_argument('--out-dir', default=str(Path('outputs') / 'fourier' / 'monthly'))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
    close = df['close']

    # Bars/day from timeframe
    if args.timeframe.endswith('h'):
        h = float(args.timeframe[:-1])
        bars_per_day = max(1, int(round(24.0 / h)))
    else:
        bars_per_day = 1

    # Iterate months
    months = pd.period_range(start=close.index.min().to_period('M'), end=close.index.max().to_period('M'), freq='M')
    out_root = Path(args.out_dir)
    for period in months:
        y = period.year
        m = period.month
        # Slice monthly window
        start = pd.Timestamp(year=y, month=m, day=1)
        end = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=0)
        sub = close.loc[(close.index >= start) & (close.index <= end)]
        if len(sub) < 32:
            continue
        P = _dominant_period(sub)
        Pvals = _top_k_periods(sub, k=6)
        P1, P2, P3, P4, P5, P6 = (Pvals + [None]*6)[:6]
        L = _lfp(sub, bars_per_day)

        # CSV per month
        month_dir = out_root / f"{y:04d}" / f"{m:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        csv_path = month_dir / f"FREQ_{args.symbol.replace('/','_')}_{args.timeframe}_{y:04d}-{m:02d}.csv"
        pd.DataFrame({
            'timestamp': sub.index,
            'P_bars': [P]*len(sub),         # alias pour compatibilité (P1)
            'P1_bars': [P1]*len(sub),
            'P2_bars': [P2]*len(sub),
            'P3_bars': [P3]*len(sub),
            'P4_bars': [P4]*len(sub),
            'P5_bars': [P5]*len(sub),
            'P6_bars': [P6]*len(sub),
            'LFP': [L]*len(sub)
        }).set_index('timestamp').to_csv(csv_path)

        # Plots per month
        fig, ax = plt.subplots(figsize=(10, 3))
        if P1 is not None:
            ax.plot(sub.index, [P1]*len(sub), color='#0D47A1', label='P1')
        if P2 is not None:
            ax.plot(sub.index, [P2]*len(sub), color='#FF6F00', label='P2')
        if P3 is not None:
            ax.plot(sub.index, [P3]*len(sub), color='#2E7D32', label='P3')
        # P4-P6 disponibles au CSV; on garde l'affichage P1-P3 pour lisibilité
        ax.set_title(f"P1/P2/P3 (bars) — {args.symbol} {args.timeframe} — {y:04d}-{m:02d}")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(month_dir / f"P_{args.symbol.replace('/','_')}_{args.timeframe}_{y:04d}-{m:02d}.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sub.index, [L]*len(sub), color='#1B5E20')
        ax.set_title(f"LFP — {args.symbol} {args.timeframe} — {y:04d}-{m:02d}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(month_dir / f"LFP_{args.symbol.replace('/','_')}_{args.timeframe}_{y:04d}-{m:02d}.png", dpi=120)
        plt.close(fig)

        print(f"Saved: {csv_path}")

    print(f"Monthly reports in: {out_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


