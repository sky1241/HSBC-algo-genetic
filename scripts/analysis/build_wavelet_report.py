#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.wavelet_utils import compute_stft, compute_cwt  # type: ignore


OHLCV_PATHS = {
    ("BTC_USDT", "2h"): Path("data") / "BTC_USDT_2h.csv",
    ("BTC_USDT", "1d"): Path("data") / "BTC_USDT_1d.csv",
    ("BTC_USD", "2h"): Path("data") / "BTC_USD_2h.csv",
    ("BTC_USD", "1d"): Path("data") / "BTC_USD_1d.csv",
}

HALVINGS: List[Tuple[pd.Timestamp, str]] = [
    (pd.Timestamp("2012-11-28"), "Halving 2012"),
    (pd.Timestamp("2016-07-09"), "Halving 2016"),
    (pd.Timestamp("2020-05-11"), "Halving 2020"),
    (pd.Timestamp("2024-04-20"), "Halving 2024"),
]


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open","high","low","close","volume"]]


def fs_per_bar(timeframe: str) -> float:
    # We use sampling frequency of 1 sample per bar
    return 1.0


def timeframe_label(timeframe: str) -> str:
    return timeframe


def annotate_halvings(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    start, end = index.min(), index.max()
    ymin, ymax = ax.get_ylim()
    for dt, label in HALVINGS:
        if dt < start or dt > end:
            continue
        ax.axvline(dt, color="black", linestyle="--", linewidth=1.2, alpha=0.9, zorder=5)
        ax.text(
            dt, ymax, label, rotation=90, va="bottom", ha="right",
            fontsize=8, color="black", alpha=0.95,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="#333", linewidth=0.4, pad=1.2),
            zorder=6,
        )
    ax.set_ylim(ymin, ymax)


def stft_figure(close: pd.Series, sym: str, tf: str, out_dir: Path) -> Path:
    fs = fs_per_bar(tf)
    res = compute_stft(close.values, fs=fs, nperseg=128, noverlap=96)
    # Period in bars = 1 / freq (avoid zero)
    freqs = res.freqs
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_bars = np.where(freqs > 0, 1.0 / freqs, np.nan)
    # Align time axis to index
    t0 = close.index[0]
    times = [t0 + pd.Timedelta(seconds=float(t)*3600*0) for t in res.times]  # res.times already in samples/fs; we map via index length below
    # For 1 sample per bar, we can map column index to timestamps subset
    time_idx = np.linspace(0, len(close) - 1, num=res.power.shape[1])
    ts = close.index[np.clip(time_idx.astype(int), 0, len(close)-1)]

    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    import matplotlib.dates as mdates
    # Map x extent to Matplotlib date numbers
    x_min = mdates.date2num(pd.to_datetime(ts[0]).to_pydatetime())
    x_max = mdates.date2num(pd.to_datetime(ts[-1]).to_pydatetime())
    y_min = np.nanmin(periods_bars)
    y_max = np.nanmax(periods_bars)
    im = ax.imshow(
        res.power,
        aspect='auto',
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='magma'
    )
    ax.set_ylabel('Période (barres)')
    ax.set_title(f"{sym} {tf} — Spectrogramme (STFT)")
    # x as dates
    ax.set_xlabel('Date')
    ax.set_ylim(y_min, y_max)
    # Format x ticks as dates
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Puissance', rotation=90)
    # Halvings
    annotate_halvings(ax, close.index)
    out = out_dir / f"{sym}_{tf}_stft.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def cwt_figure(close: pd.Series, sym: str, tf: str, out_dir: Path) -> Path:
    fs = fs_per_bar(tf)
    res = compute_cwt(close.values, fs=fs, wavelet='morl', num_scales=72)
    if res is None:
        # PyWavelets missing → skip
        return out_dir / f"{sym}_{tf}_cwt_NOT_AVAILABLE.png"
    # Convert scales to period using pywt.scale2frequency
    import pywt
    freqs = pywt.scale2frequency('morl', res.scales) * fs
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = np.where(freqs > 0, 1.0 / freqs, np.nan)
    # Time mapping similar to STFT
    time_idx = np.linspace(0, len(close) - 1, num=res.power.shape[1])
    ts = close.index[np.clip(time_idx.astype(int), 0, len(close)-1)]

    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    import matplotlib.dates as mdates
    x_min = mdates.date2num(pd.to_datetime(ts[0]).to_pydatetime())
    x_max = mdates.date2num(pd.to_datetime(ts[-1]).to_pydatetime())
    y = periods
    y = y[np.isfinite(y)]
    y_min = float(np.nanpercentile(y, 1))
    y_max = float(np.nanpercentile(y, 99))
    im = ax.imshow(
        res.power,
        aspect='auto',
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='viridis'
    )
    ax.set_ylabel('Echelle ~ Période (barres)')
    ax.set_title(f"{sym} {tf} — Scalogramme (CWT Morlet)")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Puissance', rotation=90)
    annotate_halvings(ax, close.index)
    out = out_dir / f"{sym}_{tf}_cwt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def price_figure(close: pd.Series, sym: str, tf: str, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(close.index, close.values, color='black', linewidth=1.2)
    ax.set_title(f"{sym} {tf} — Prix")
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.grid(True, alpha=0.2)
    annotate_halvings(ax, close.index)
    out = out_dir / f"{sym}_{tf}_price.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_one(sym: str, tf: str) -> List[Path]:
    path = OHLCV_PATHS.get((sym, tf))
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    df = read_ohlcv(path)
    out_dir = Path('outputs') / 'fourier' / 'wavelets' / sym / tf
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: List[Path] = []
    figs.append(price_figure(df['close'], sym, tf, out_dir))
    figs.append(stft_figure(df['close'], sym, tf, out_dir))
    figs.append(cwt_figure(df['close'], sym, tf, out_dir))
    return figs


def build_markdown(figs_map: dict[tuple[str,str], List[Path]]) -> Path:
    md_lines: List[str] = []
    md_lines.append('## Rapport ondelettes/STFT (séparé)')
    md_lines.append('Ce rapport présente STFT (spectrogramme) et ondelettes (scalogramme) pour évaluer l’intérêt temporel des périodes dominantes.')
    md_lines.append('')
    for (sym, tf), paths in figs_map.items():
        md_lines.append(f"### {sym} {tf}")
        for p in paths:
            md_lines.append(f"![{p.name}]({p.as_posix()})")
        md_lines.append('')
    out_md = Path('docs') / 'FOURIER_WAVELETS_REPORT.md'
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    return out_md


def main() -> int:
    tasks = [("BTC_USDT","2h"),("BTC_USDT","1d"),("BTC_USD","2h"),("BTC_USD","1d")]
    figs_map: dict[tuple[str,str], List[Path]] = {}
    for sym, tf in tasks:
        try:
            figs_map[(sym, tf)] = generate_one(sym, tf)
        except FileNotFoundError:
            continue
    md = build_markdown(figs_map)
    print('Wrote:', md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


