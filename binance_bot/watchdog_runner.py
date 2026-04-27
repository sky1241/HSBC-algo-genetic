#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watchdog runner: lance les health checks et persiste le rapport."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from bot.watchdog import run_health_checks


def main():
    repo_root = ROOT.parent  # /home/ludov/HSBC-algo-genetic

    # Initialiser Binance client (best effort — si .env absent, skip)
    binance_client = None
    try:
        from services.data_fetcher import DataFetcher
        df = DataFetcher("BTC/USDT", "2h")
        binance_client = df.exchange
    except Exception as e:
        print(f"⚠️ Pas de client Binance pour watchdog: {e}")

    report = run_health_checks(repo_root=repo_root, binance_client=binance_client)

    # Log JSON
    out_path = repo_root / "binance_bot" / "logs" / "watchdog.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(report.to_dict()) + "\n")

    # Health snapshot pour le dashboard
    snap_path = repo_root / "binance_bot" / "data" / "health.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Print pour le journal systemd
    icon = {"OK": "✅", "WARN": "⚠️", "CRITICAL": "🛑"}[report.status]
    print(f"{icon} {report.status} — {report.timestamp_iso}")
    for a in report.anomalies:
        print(f"  {a}")
    for k, v in report.metrics.items():
        print(f"  {k}: {v}")

    # Exit code reflects severity (0=OK/WARN, 1=CRITICAL pour systemd alerts)
    return 0 if report.status != "CRITICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
