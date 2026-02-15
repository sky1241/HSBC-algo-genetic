#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC gaps for 2h OHLCV. Detect missing 2h bars and optionally fill synthetic bars.

Usage:
  python scripts/qc_detect_and_fix_gaps.py --path data/BTC_FUSED_2h.csv --out data/BTC_FUSED_2h_clean.csv --fill ffill

Fill modes:
  - none  : do not modify, only report
  - ffill : insert missing 2h timestamps; set open/high/low/close to previous close; volume=0
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd  # type: ignore


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Find datetime column
    for col in ["timestamp", "date", "time", "datetime"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
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


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.sort_values()
    delta = (idx[1:] - idx[:-1]).to_series(index=idx[1:])
    return delta[delta > pd.Timedelta(hours=2)]


def fill_missing_bars(df: pd.DataFrame) -> pd.DataFrame:
    start, end = df.index.min(), df.index.max()
    full = pd.date_range(start=start, end=end, freq='2H')
    out = df.reindex(full)
    prev_close = out['close'].ffill()
    out['open'] = out['open'].fillna(prev_close)
    out['high'] = out['high'].fillna(prev_close)
    out['low'] = out['low'].fillna(prev_close)
    out['close'] = out['close'].fillna(prev_close)
    out['volume'] = out['volume'].fillna(0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect/fix 2h gaps in OHLCV CSV")
    ap.add_argument("--path", required=True, help="Input CSV path (e.g., data/BTC_FUSED_2h.csv)")
    ap.add_argument("--out", default="", help="Output CSV path for cleaned data")
    ap.add_argument("--fill", choices=["none","ffill"], default="none")
    args = ap.parse_args()

    inp = Path(args.path)
    if not inp.exists():
        raise FileNotFoundError(inp)
    df = read_ohlcv(inp)
    # Report
    gaps = detect_gaps(df)
    print(f"Rows: {len(df)}, Range: {df.index.min()} -> {df.index.max()}")
    print(f"Extreme gaps (>2h): {len(gaps)}")
    if len(gaps):
        print(gaps.head(20))

    if args.fill != "none":
        cleaned = fill_missing_bars(df)
        outp = Path(args.out) if args.out else inp.with_name(inp.stem + "_clean.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(outp, index=True, index_label='timestamp')
        print("Saved:", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


