#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import calendar
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.plot_phase_price import (
    read_ohlcv,
    load_daily_summary,
    plot_with_phases,
)
from scripts.phase_aware_module import phase_snapshot


LABELSETS = ["regime3", "phase5", "phase6"]


def month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


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


def build_monthly_reports(symbol: str, timeframe: str) -> List[Path]:
    # Load
    ohlcv_path = Path("data") / f"{symbol}_{timeframe}.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(ohlcv_path)
    df = read_ohlcv(ohlcv_path)
    feats = phase_snapshot(df).reindex(df.index).ffill()
    daily = load_daily_summary(symbol, timeframe)

    out_png: List[Path] = []
    for m_start, m_end in iter_month_ranges(df.index):
        sli = df.loc[(df.index >= m_start) & (df.index <= m_end)]
        if len(sli) < 10:
            continue
        feats_m = feats.loc[sli.index]
        daily_m = daily.reindex(sli.index, method='ffill')
        y = m_start.strftime("%Y")
        ym = m_start.strftime("%Y-%m")
        out_dir = Path("outputs") / "fourier" / "phase_monthly" / symbol / timeframe / y / ym
        out_dir.mkdir(parents=True, exist_ok=True)
        # Skip regeneration if all three labelset images already exist
        existing = {ls: (out_dir / f"{symbol}_{timeframe}_{ls}_{ym}_phase_price.png").exists() for ls in LABELSETS}
        if not all(existing.values()):
            for ls in LABELSETS:
                if existing.get(ls, False):
                    continue
                png = plot_with_phases(sli, feats_m, daily_m, symbol, timeframe, ls, out_dir_override=out_dir, out_suffix=ym)
                out_png.append(png)
    return out_png


def main() -> int:
    # Build for both Binance and Bitstamp, 2h and 1d
    tasks = [("BTC_USDT","2h"),("BTC_USDT","1d"),("BTC_USD","2h"),("BTC_USD","1d")]
    all_pngs: List[Path] = []
    for sym, tf in tasks:
        all_pngs.extend(build_monthly_reports(sym, tf))
    # Build simple index markdown per symbol/timeframe
    for sym, tf in tasks:
        base = Path("outputs") / "fourier" / "phase_monthly" / sym / tf
        if not base.exists():
            continue
        md = [f"## Rapports mensuels — {sym} {tf}", ""]
        for year_dir in sorted(base.iterdir()):
            if not year_dir.is_dir():
                continue
            md.append(f"### {year_dir.name}")
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                md.append(f"#### {month_dir.name}")
                # three labelsets and per-month report page
                month_md = [f"### {sym} {tf} — {month_dir.name}", ""]
                for ls in LABELSETS:
                    png = month_dir / f"{sym}_{tf}_{ls}_{month_dir.name}_phase_price.png"
                    if png.exists():
                        month_md.append(f"- {ls}")
                        month_md.append(f"![{png.name}]({png.as_posix()})")
                report_path = month_dir / "REPORT.md"
                report_path.write_text("\n".join(month_md), encoding="utf-8")
                md.append(f"- Rapport: {report_path.as_posix()}")
                md.append("")
        idx_path = base / "INDEX.md"
        idx_path.write_text("\n".join(md), encoding="utf-8")
        print("Wrote:", idx_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


