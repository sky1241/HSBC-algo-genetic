#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate phase shares (percentages per phase) WITHOUT Welch and WITHOUT anti-alias filtering,
then save a reference figure as docs/IMAGES/PHASES_NO_WELCH_NO_ANTIALIAS_2025-08-28.png

This approximates the historical reference described in docs/JOURNAL_2025-08-28.md.

Input: data/BTC_USDT_2h.csv (or BTC_USD_2h.csv as fallback)
Output: docs/IMAGES/PHASES_NO_WELCH_NO_ANTIALIAS_2025-08-28.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Matplotlib only for saving the figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

# Minimal bands classifier (no Welch, no AA): classify by FFT band energy shares on raw returns

BANDS: Dict[str, Tuple[float, float]] = {
    "accumulation": (0.00, 0.10),
    "bear":         (0.10, 0.30),
    "distribution": (0.30, 1.00),
    "expansion":    (1.00, 3.00),
    "euphoria":     (3.00, 6.00),
}


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try common datetime column names
    for col in ["timestamp", "date", "time", "datetime"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            # make tz-naive (UTC implicit)
            try:
                ts = ts.dt.tz_localize(None)
            except Exception:
                try:
                    ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
                except Exception:
                    pass
            df = df.set_index(ts)
            break
    cols = {c.lower(): c for c in df.columns}
    ren = {}
    for k in ["open","high","low","close","volume"]:
        if k in cols and cols[k] != k:
            ren[cols[k]] = k
    if ren:
        df = df.rename(columns=ren)
    return df[["open","high","low","close","volume"]].sort_index()


def dominant_fft_band_returns(close: pd.Series, fs: float = 12.0, win: int = 84, step: int | None = None) -> pd.Series:
    rt = np.log(close).diff().dropna()
    if len(rt) < win:
        return pd.Series(index=close.index, dtype=str)
    if step is None:
        step = max(1, int(round(win * 0.2)))
    labels_at = []
    idx = rt.index
    for end_i in range(win, len(rt) + 1, step):
        seg = rt.iloc[end_i - win: end_i].values.astype(float)
        # raw FFT PSD (no Welch)
        seg = seg - np.nanmean(seg)
        n = len(seg)
        fft = np.fft.rfft(seg)
        psd = (np.abs(fft) ** 2) / max(1.0, float(n))
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        # band shares
        energies = {}
        for name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            energies[name] = float(np.nansum(psd[mask]))
        total = float(sum(energies.values()))
        shares = {k: (v/total if total > 0 else 0.0) for k, v in energies.items()}
        top = max(shares.items(), key=lambda kv: kv[1])[0]
        labels_at.append((idx[end_i - 1], top))
    lab = pd.Series({ts: ph for ts, ph in labels_at}).sort_index()
    return lab.reindex(close.index, method="ffill").astype(str)


def compute_monthly_phase_shares(labels: pd.Series) -> pd.DataFrame:
    mon = labels.index.to_period('M').to_timestamp()
    df = pd.DataFrame({'month': mon, 'lab': labels.values})
    counts = df.pivot_table(index='month', columns='lab', values='lab', aggfunc='count').fillna(0)
    for k in BANDS.keys():
        if k not in counts.columns:
            counts[k] = 0
    counts = counts[list(BANDS.keys())]
    counts['total'] = counts.sum(axis=1)
    shares = counts.div(counts['total'].where(counts['total'] > 0), axis=0).fillna(0)
    return shares.drop(columns=['total'], errors='ignore')


def plot_global_shares(shares: pd.DataFrame, out_path: Path) -> None:
    mean_shares = shares.mean(axis=0).reindex(list(BANDS.keys())).fillna(0)
    plt.figure(figsize=(8, 5))
    mean_shares.plot(kind='bar', color=['#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f0e'])
    plt.ylabel('Part moyenne par mois')
    plt.title('Parts % par phase (sans Welch, sans anti‑alias) — référence 2025‑08‑28')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    # Prefer USDT; fallback USD
    for sym in ["BTC_FUSED", "BTC_USDT", "BTC_USD"]:
        path = Path('data') / f"{sym}_2h.csv"
        if path.exists():
            df = read_ohlcv(path)
            break
    else:
        raise FileNotFoundError("data/BTC_USDT_2h.csv or data/BTC_USD_2h.csv not found")

    close = df['close']
    labels = dominant_fft_band_returns(close, fs=12.0, win=84, step=None)
    if labels.empty:
        raise RuntimeError("Not enough data to compute labels")
    shares = compute_monthly_phase_shares(labels)
    out = Path('docs') / 'IMAGES' / 'PHASES_NO_WELCH_NO_ANTIALIAS_2025-08-28.png'
    plot_global_shares(shares, out)
    print('Saved:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


