#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot price with phase-colored bands.

Usage:
    python scripts/plot_phase_price.py --symbol BTC_USD --timeframe 1d

The script loads OHLCV data from ``data/<symbol>_<timeframe>.csv``,
computes phase features via :func:`phase_snapshot`, and saves a PNG
with the price curve and colored regions for each phase.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.phase_aware_module import phase_snapshot  # type: ignore

OHLCV_PATHS: Dict[Tuple[str, str], Path] = {
    ("BTC_USDT", "2h"): Path("data") / "BTC_USDT_2h.csv",
    ("BTC_USDT", "1d"): Path("data") / "BTC_USDT_1d.csv",
    ("BTC_USD", "2h"): Path("data") / "BTC_USD_2h.csv",
    ("BTC_USD", "1d"): Path("data") / "BTC_USD_1d.csv",
}

# Color palettes per label set
PHASE_COLORS_6 = {
    # High-contrast qualitative palette
    "accumulation": "#1f77b4",   # blue
    "expansion": "#2ca02c",      # green
    "euphoria": "#ff7f0e",       # orange
    "distribution": "#9467bd",   # purple
    "bear": "#d62728",           # red
    "capitulation": "#e377c2",   # magenta
}

PHASE_COLORS_5 = {
    k: v for k, v in PHASE_COLORS_6.items() if k != "capitulation"
}

PHASE_COLORS_3 = {
    "up": "#2ca02c",
    "down": "#d62728",
    "range": "#7f7f7f",
}

# Known historical BTC halving dates
HALVINGS: List[Tuple[pd.Timestamp, str]] = [
    (pd.Timestamp("2012-11-28"), "Halving 2012"),
    (pd.Timestamp("2016-07-09"), "Halving 2016"),
    (pd.Timestamp("2020-05-11"), "Halving 2020"),
    (pd.Timestamp("2024-04-20"), "Halving 2024"),
]

def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def load_daily_summary(symbol: str, timeframe: str) -> pd.DataFrame:
    p = Path('outputs') / 'fourier' / f'DAILY_SUMMARY_{symbol}_{timeframe}.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    return df

def segment_phases(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    phases = df["phase"].astype(str)
    segments = []
    start = df.index[0]
    current = phases.iloc[0]
    for ts, ph in zip(df.index[1:], phases.iloc[1:]):
        if ph != current:
            segments.append((start, ts, current))
            start = ts
            current = ph
    segments.append((start, df.index[-1], current))
    return segments

def annotate_halvings(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    start, end = index.min(), index.max()
    ymin, ymax = ax.get_ylim()
    for dt, label in HALVINGS:
        if dt < start or dt > end:
            continue
        ax.axvline(dt, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
        ax.text(
            dt, ymax, label, rotation=90, va="bottom", ha="right",
            fontsize=8, color="black", alpha=0.9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
        )


def _bars_per_day(timeframe: str) -> float:
    if timeframe.endswith("h"):
        try:
            hours = float(timeframe[:-1])
            return 24.0 / hours if hours > 0 else 12.0
        except Exception:
            return 12.0
    # default for 1d
    return 1.0


def _format_duration_days_h2(start: pd.Timestamp, end: pd.Timestamp, tf: str, seg_len_rows: int) -> Tuple[str, int, float]:
    duration_days = max(0.0, (end - start).total_seconds() / 86400.0)
    if tf == "2h":
        h2_bars = seg_len_rows
    else:
        h2_bars = int(round(duration_days * 12.0))
    text = f"{int(round(duration_days))}j | {h2_bars} H2"
    return text, h2_bars, duration_days


def annotate_halvings(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    start, end = index.min(), index.max()
    ymin, ymax = ax.get_ylim()
    for dt, label in HALVINGS:
        if dt < start or dt > end:
            continue
        ax.axvline(dt, color="black", linestyle="--", linewidth=1.6, alpha=0.95, zorder=5)
        ax.text(
            dt, ymax, label, rotation=90, va="bottom", ha="right",
            fontsize=8.5, color="black", alpha=0.98,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#333333", linewidth=0.4, pad=1.5),
            zorder=6,
        )
    # keep axis limits unchanged (avoid autoscale by text)
    ax.set_ylim(ymin, ymax)


def _annotate_segments(ax: plt.Axes, df: pd.DataFrame, segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]], tf: str) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    for start, end, ph in segments:
        sli = df.loc[(df.index >= start) & (df.index <= end)]
        if sli.empty:
            continue
        text, h2_bars, days = _format_duration_days_h2(start, end, tf, len(sli))
        t_mid = start + (end - start) / 2
        # Position near top of the band for readability
        y_pos = ymax - 0.06 * yr
        label = f"{ph} | {start.date()} → {end.date()} | {text}"
        ax.text(
            t_mid, y_pos, label, rotation=0, ha="center", va="top",
            fontsize=8, color="#111111", zorder=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#444444", linewidth=0.4, pad=1.5),
        )
    ax.set_ylim(ymin, ymax)


def label_3_regimes(M: float, LFP: float, m_up: float = 0.05, m_down: float = -0.05, lfp_trend: float = 0.80) -> str:
    if pd.isna(M) or pd.isna(LFP):
        return 'range'
    if (M >= m_up) and (LFP >= lfp_trend):
        return 'up'
    if (M <= m_down) and (LFP >= lfp_trend):
        return 'down'
    return 'range'


def label_6_from_5(phase5: str, M: float, V_ann: float, DD: float) -> str:
    if not (pd.isna(M) or pd.isna(V_ann) or pd.isna(DD)):
        if (M <= -0.15) and (V_ann >= 1.0) and (DD <= -0.50):
            return 'capitulation'
    return str(phase5)


def _daily_candle_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    # Inclusive count of daily bars assuming 24/7 crypto (no market close)
    return (end.floor('D') - start.floor('D')).days + 1


def _build_segments_table(df: pd.DataFrame, segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]], tf: str) -> Tuple[List[str], List[List[str]]]:
    headers = ["Phase", "Start", "End", "D1 candles", "H2 bars"]
    rows: List[List[str]] = []
    for start, end, ph in segments:
        sli = df.loc[(df.index >= start) & (df.index <= end)]
        if sli.empty:
            continue
        d1 = _daily_candle_count(start, end)
        h2 = len(sli) if tf == '2h' else d1 * 12
        rows.append([ph, str(start.date()), str(end.date()), str(d1), str(h2)])
    return headers, rows


def plot_with_phases(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    daily: pd.DataFrame,
    sym: str,
    tf: str,
    labelset: str,
    out_dir_override: Optional[Path] = None,
    out_suffix: Optional[str] = None,
) -> Path:
    # Build labels by set
    if labelset == 'phase5':
        phases = feats['phase'].astype(str)
        palette = PHASE_COLORS_5
    elif labelset == 'phase6':
        phases = pd.Series(
            [label_6_from_5(p5, m, v, dd) for p5, m, v, dd in zip(feats['phase'], feats['M'], feats['V_ann'], feats['DD'])],
            index=feats.index
        )
        palette = PHASE_COLORS_6
    elif labelset == 'regime3':
        daily_aligned = daily.reindex(df.index, method='ffill')
        phases = pd.Series(
            [label_3_regimes(m, lfp) for m, lfp in zip(feats.reindex(df.index)['M'], daily_aligned['LFP'])],
            index=df.index
        )
        palette = PHASE_COLORS_3
    else:
        raise ValueError("labelset must be one of: regime3, phase5, phase6")

    # Align to df index and segment
    phases = phases.reindex(df.index, method='ffill').astype(str)
    tmp = df.copy()
    tmp['__lab__'] = phases
    def seg_from_series(s: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
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
    segments = seg_from_series(tmp['__lab__'])
    # Compose figure with space for a summary table below
    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3.5, 1.5], hspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    # Draw colored phase bands behind the price (higher contrast)
    for start, end, ph in segments:
        color = palette.get(ph, "grey")
        ax.axvspan(start, end, color=color, alpha=0.38, zorder=0)
    # Optional: thin separators between segments for extra contrast
    for _, end, _ in segments[:-1]:
        ax.axvline(end, color="white", linewidth=0.6, alpha=0.9, zorder=1)
    # Price on top
    ax.plot(df.index, df["close"], color="black", linewidth=1.35, zorder=3)
    handles = [mpatches.Patch(color=c, label=p) for p, c in palette.items()]
    ax.legend(handles=handles, title="Phase", loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"{sym} {tf} — {labelset} — price with phases")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True, which="both", axis="both", alpha=0.2)
    # Annotate halvings and segment durations (after plotting so y-lims are known)
    annotate_halvings(ax, df.index)
    _annotate_segments(ax, df, segments, tf)
    # Build and render the table under the chart
    headers, rows = _build_segments_table(df, segments, tf)
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis('off')
    if rows:
        table = ax_tbl.table(cellText=rows, colLabels=headers, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
    ax_tbl.set_title("Durées par phase (D1 et H2)", fontsize=10, pad=6)
    fig.tight_layout()
    out_dir = (out_dir_override if out_dir_override is not None else (Path("outputs") / "fourier" / "phase_plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{out_suffix}" if out_suffix else ""
    out_file = out_dir / f"{sym}_{tf}_{labelset}{suffix}_phase_price.png"
    fig.savefig(out_file)
    plt.close(fig)
    print("Saved plot to", out_file)
    return out_file

def main() -> int:
    ap = argparse.ArgumentParser(description="Plot price with phase-colored bands.")
    ap.add_argument("--symbol", default="BTC_USD")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--labelset", default="phase6", choices=["regime3","phase5","phase6"], help="Which label set to use")
    args = ap.parse_args()

    path = OHLCV_PATHS.get((args.symbol, args.timeframe))
    if path is None or not path.exists():
        raise FileNotFoundError(f"OHLCV not found for {(args.symbol, args.timeframe)}")
    df = read_ohlcv(path)
    feats = phase_snapshot(df)
    feats = feats.reindex(df.index).ffill()
    daily = load_daily_summary(args.symbol, args.timeframe)
    plot_with_phases(df, feats, daily, args.symbol, args.timeframe, args.labelset)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
