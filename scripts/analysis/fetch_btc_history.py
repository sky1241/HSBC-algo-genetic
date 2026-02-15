#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Télécharge un historique BTC/USDT en 2h depuis Binance et le stocke en cache data/BTC_USDT_2h.csv
Utilise la fonction fetch_ohlcv_range du pipeline pour assurer le même format de cache.

Usage:
  py -3 scripts/fetch_btc_history.py --years-back 8
  py -3 scripts/fetch_btc_history.py --since 2017-08-01 --until now
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import ccxt

import sys
from pathlib import Path as _Path
# Ensure project root is importable
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from ichimoku_pipeline_web_v4_8_fixed import fetch_ohlcv_range, utc_ms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="2h")
    p.add_argument("--exchange", default="binance", choices=["binance","bitstamp"], help="Exchange CCXT à utiliser")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--years-back", type=int, default=8)
    g.add_argument("--since", help="YYYY-MM-DD")
    p.add_argument("--until", default="now", help="YYYY-MM-DD ou 'now'")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    ex = ccxt.binance({"enableRateLimit": True}) if args.exchange == "binance" else ccxt.bitstamp({"enableRateLimit": True})

    if args.since:
        since_dt = datetime.fromisoformat(args.since)
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(days=int(365.25 * args.years_back))
    until_dt = datetime.now(timezone.utc) if args.until == "now" else datetime.fromisoformat(args.until)

    since_ms = utc_ms(since_dt)
    until_ms = utc_ms(until_dt)

    df = fetch_ohlcv_range(
        ex,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        cache_dir="data",
        use_cache=not args.no_cache,
    )
    if df.empty:
        print("No data fetched.")
        return 1
    print(f"Fetched {len(df)} rows: {df.index.min()} → {df.index.max()}")
    print("Cache updated: data/{}_{}.csv".format(args.symbol.replace('/', '_'), args.timeframe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


