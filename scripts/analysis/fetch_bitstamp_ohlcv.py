#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch OHLCV from Bitstamp with pagination, independent of pipeline helpers.

Usage:
  py -3 scripts/fetch_bitstamp_ohlcv.py --symbol BTC/USD --timeframe 1h --since 2011-08-18 --out data/BTC_USD_1h.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import ccxt  # type: ignore
import pandas as pd


def parse_date(s: str) -> int:
    if s == 'now':
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTC/USD')
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--since', default='2011-08-18')
    p.add_argument('--until', default='now')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    ex = ccxt.bitstamp({'enableRateLimit': True})
    since_ms = parse_date(args.since)
    until_ms = parse_date(args.until)
    limit = 1000

    out_path = Path(args.out) if args.out else Path('data') / f"{args.symbol.replace('/', '_')}_{args.timeframe}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file if any
    existing = None
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path, parse_dates=['timestamp']).set_index('timestamp').sort_index()
            if not existing.empty:
                last_ts = int(existing.index.max().timestamp() * 1000)
                since_ms = max(since_ms, last_ts)
        except Exception:
            existing = None

    all_rows = []
    cursor = since_ms
    while cursor < until_ms:
        candles = ex.fetch_ohlcv(args.symbol, timeframe=args.timeframe, since=cursor, limit=limit)
        if not candles:
            break
        # Prevent infinite loop if exchange returns same starting ts
        next_cursor = candles[-1][0] + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        all_rows.extend(candles)
        time.sleep(ex.rateLimit / 1000.0)

    if not all_rows and existing is None:
        print('No data fetched and no existing file. Abort.')
        return 1

    df_new = pd.DataFrame(all_rows, columns=['ms','open','high','low','close','volume'])
    if not df_new.empty:
        df_new['timestamp'] = pd.to_datetime(df_new['ms'], unit='ms', utc=True).dt.tz_convert(None)
        df_new = df_new[['timestamp','open','high','low','close','volume']]
        df_new = df_new.set_index('timestamp').sort_index()

    if existing is not None and not existing.empty:
        merged = pd.concat([existing, df_new]).drop_duplicates(keep='last') if not df_new.empty else existing
    else:
        merged = df_new

    if merged is None or merged.empty:
        print('No data to write.')
        return 1

    merged = merged[~merged.index.duplicated(keep='last')]
    merged = merged.sort_index()
    merged.to_csv(out_path)
    print(f"Wrote {out_path} rows={len(merged)} range={merged.index.min()} -> {merged.index.max()}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


