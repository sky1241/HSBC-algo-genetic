"""P6.2 — Binance Futures takerlongshortRatio polling + persistence.

REST endpoint Binance USDM Futures (vérifié 27/04/2026) :
    GET https://fapi.binance.com/futures/data/takerlongshortRatio
    Params: symbol (BTCUSDT), period (5m/15m/30m/1h/2h/4h/6h/12h/1d),
            limit (default 30, max 500)
    Réponse: liste d'objets {buySellRatio, buyVol, sellVol, timestamp}
    Historique disponible: 30 derniers jours rolling
    Rate limit: 1000 req/5min/IP

Mesure l'AGRESSIVITÉ taker (volume buy-aggressive vs sell-aggressive). Distinct
de topLongShortPositionRatio (P6.1) qui mesure le NOTIONAL position long/short
des top traders. Les 2 sont complémentaires dans le composite_signal (P6.5).

Stockage JSONL append-only avec dedup par ts_ms.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests


_REST_BASE_URL = "https://fapi.binance.com"
_ENDPOINT_TAKER_RATIO = "/futures/data/takerlongshortRatio"
_DEFAULT_PERIOD = "5m"
_DEFAULT_LIMIT = 500
_REQUEST_TIMEOUT_SEC = 10
_BACKOFF_INITIAL_SEC = 1.0
_BACKOFF_MAX_SEC = 60.0


def _parse_records(payload: list[dict], symbol: str) -> list[dict]:
    """Convertit la réponse Binance en records normalisés.

    Returns:
        Liste de dicts {ts_ms, symbol, buy_sell_ratio, buy_vol, sell_vol}.
        Skip silencieusement les entrées malformées.
    """
    records: list[dict] = []
    for item in payload:
        try:
            records.append({
                "ts_ms": int(item["timestamp"]),
                "symbol": str(item.get("symbol", symbol)),
                "buy_sell_ratio": float(item["buySellRatio"]),
                "buy_vol": float(item["buyVol"]),
                "sell_vol": float(item["sellVol"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return records


def fetch_taker_ratio(
    symbol: str,
    period: str = _DEFAULT_PERIOD,
    limit: int = _DEFAULT_LIMIT,
    end_time_ms: Optional[int] = None,
    timeout: float = _REQUEST_TIMEOUT_SEC,
) -> list[dict]:
    """Appelle l'endpoint REST 1 fois et retourne les records normalisés.

    Comportement identique à flow_top_ls.fetch_top_ls_ratio (backoff sur 429,
    empty list sur autres erreurs HTTP, pas de raise).
    """
    params = {
        "symbol": symbol.replace("/", "").upper(),
        "period": period,
        "limit": int(max(1, min(limit, 500))),
    }
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    url = _REST_BASE_URL + _ENDPOINT_TAKER_RATIO
    backoff = _BACKOFF_INITIAL_SEC
    while True:
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            return []
        if r.status_code == 200:
            try:
                payload = r.json()
            except ValueError:
                return []
            if not isinstance(payload, list):
                return []
            return _parse_records(payload, symbol=params["symbol"])
        if r.status_code == 429:
            if backoff > _BACKOFF_MAX_SEC:
                return []
            time.sleep(backoff)
            backoff *= 2
            continue
        return []


def _load_existing_ts(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts = int(rec.get("ts_ms", -1))
                if ts > 0:
                    seen.add(ts)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return seen


def _append_records(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def store_records_dedup(records: Iterable[dict], store_path: Path) -> int:
    existing = _load_existing_ts(store_path)
    new_records = [r for r in records if int(r.get("ts_ms", -1)) not in existing]
    return _append_records(new_records, store_path)


def backfill_30d(
    symbol: str,
    store_path: Path,
    period: str = _DEFAULT_PERIOD,
    fetch_func=fetch_taker_ratio,
) -> int:
    """Backfill jusqu'à 30j d'historique. Pagine vers le passé via endTime."""
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
    end_time_ms: Optional[int] = None
    total_written = 0
    seen_ts = _load_existing_ts(store_path)
    max_pages = 50

    for _ in range(max_pages):
        page = fetch_func(symbol=symbol, period=period, limit=500, end_time_ms=end_time_ms)
        if not page:
            break
        new_records = [r for r in page if int(r["ts_ms"]) not in seen_ts]
        if not new_records:
            break
        n = _append_records(new_records, store_path)
        total_written += n
        for r in new_records:
            seen_ts.add(int(r["ts_ms"]))
        oldest_ts = min(int(r["ts_ms"]) for r in page)
        if oldest_ts <= cutoff_ms:
            break
        end_time_ms = oldest_ts - 1
    return total_written


def poll_and_store(
    symbol: str,
    store_path: Path,
    period: str = _DEFAULT_PERIOD,
    limit: int = 30,
    fetch_func=fetch_taker_ratio,
) -> int:
    """1 poll synchrone + append dedup. Pour cron 5min."""
    records = fetch_func(symbol=symbol, period=period, limit=limit, end_time_ms=None)
    return store_records_dedup(records, store_path)


__all__ = [
    "fetch_taker_ratio",
    "backfill_30d",
    "poll_and_store",
    "store_records_dedup",
]
