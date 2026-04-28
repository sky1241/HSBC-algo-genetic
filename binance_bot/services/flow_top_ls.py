"""P6.1 — Binance Futures topLongShortPositionRatio polling + persistence.

REST endpoint Binance USDM Futures (vérifié 27/04/2026) :
    GET https://fapi.binance.com/futures/data/topLongShortPositionRatio
    Params: symbol (e.g. BTCUSDT), period (5m/15m/30m/1h/2h/4h/6h/12h/1d),
            limit (default 30, max 500)
    Réponse: liste d'objets {longShortRatio, longAccount, shortAccount, timestamp, symbol}
    Historique disponible: 30 derniers jours rolling
    Rate limit: 1000 req/5min/IP

Stockage JSONL append-only avec dedup par timestamp (évite double-insertion
sur reruns du backfill ou du polling).

Le module est SOURCE de données. P6.5 agrégera ce flux + 3 autres dans un
composite_signal. Pour P6.1 minimum-viable, pas de hook signal_engine.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests


_REST_BASE_URL = "https://fapi.binance.com"
_ENDPOINT_TOP_LS = "/futures/data/topLongShortPositionRatio"
_DEFAULT_PERIOD = "5m"
_DEFAULT_LIMIT = 500           # max permis par Binance
_BACKFILL_PERIODS_30D_5M = 8640  # 30 jours × 24h × 12 (= 30j de bars 5min)
_REQUEST_TIMEOUT_SEC = 10
_BACKOFF_INITIAL_SEC = 1.0
_BACKOFF_MAX_SEC = 60.0


def _parse_records(payload: list[dict], symbol: str) -> list[dict]:
    """Convertit la réponse Binance en records normalisés.

    Returns:
        Liste de dicts {ts_ms, symbol, long_short_ratio, long_account, short_account}.
        Skip silencieusement les entrées malformées.
    """
    records: list[dict] = []
    for item in payload:
        try:
            records.append({
                "ts_ms": int(item["timestamp"]),
                "symbol": str(item.get("symbol", symbol)),
                "long_short_ratio": float(item["longShortRatio"]),
                "long_account": float(item["longAccount"]),
                "short_account": float(item["shortAccount"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return records


def fetch_top_ls_ratio(
    symbol: str,
    period: str = _DEFAULT_PERIOD,
    limit: int = _DEFAULT_LIMIT,
    end_time_ms: Optional[int] = None,
    timeout: float = _REQUEST_TIMEOUT_SEC,
) -> list[dict]:
    """Appelle l'endpoint REST 1 fois et retourne les records normalisés.

    Args:
        symbol: paire futures (e.g. "BTCUSDT" — Binance fmt sans slash).
        period: granularité (5m/15m/30m/1h/2h/4h/6h/12h/1d).
        limit: nombre max d'observations (1..500).
        end_time_ms: fin de fenêtre (epoch ms); None = maintenant.
        timeout: timeout HTTP (sec).

    Returns:
        Liste de records (dict). Liste vide en cas d'erreur HTTP non-429.

    Raises:
        Aucune exception ne propage : 429 (rate limit) → backoff exponentiel,
        autres codes → empty list. Le caller voit le résultat (potentiellement vide).
    """
    params = {
        "symbol": symbol.replace("/", "").upper(),
        "period": period,
        "limit": int(max(1, min(limit, 500))),
    }
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    url = _REST_BASE_URL + _ENDPOINT_TOP_LS
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
            # Rate limit: backoff exponentiel jusqu'à _BACKOFF_MAX_SEC, puis abandon
            if backoff > _BACKOFF_MAX_SEC:
                return []
            time.sleep(backoff)
            backoff *= 2
            continue
        # Autre erreur HTTP (4xx hors 429, 5xx) : on n'insiste pas
        return []


def _load_existing_ts(path: Path) -> set[int]:
    """Charge les timestamps déjà présents dans le jsonl (dedup helper)."""
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
    """Append records (jsonl, ascii-safe). Retourne le nb effectivement écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def store_records_dedup(records: Iterable[dict], store_path: Path) -> int:
    """Append-only avec dedup par ts_ms. Retourne nb d'enregistrements écrits."""
    existing = _load_existing_ts(store_path)
    new_records = [r for r in records if int(r.get("ts_ms", -1)) not in existing]
    return _append_records(new_records, store_path)


def backfill_30d(
    symbol: str,
    store_path: Path,
    period: str = _DEFAULT_PERIOD,
    fetch_func=fetch_top_ls_ratio,
) -> int:
    """Backfill jusqu'à 30 jours d'historique (paginé 500 records/call).

    Pagine vers le passé via le param `endTime` Binance jusqu'à atteindre
    30 jours ou une page vide.

    Args:
        symbol: paire (BTCUSDT).
        store_path: jsonl destination.
        period: granularité (default 5m → 30j ≈ 8640 records).
        fetch_func: hook pour testabilité (default = fetch_top_ls_ratio réel).

    Returns:
        Nombre d'enregistrements nouveaux ajoutés au jsonl.
    """
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
    end_time_ms: Optional[int] = None
    total_written = 0
    seen_ts = _load_existing_ts(store_path)
    max_pages = 50  # safety cap : 50 × 500 = 25000 > 8640 attendus

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
        # Paginer vers le passé : end_time_ms = oldest - 1
        end_time_ms = oldest_ts - 1
    return total_written


def poll_and_store(
    symbol: str,
    store_path: Path,
    period: str = _DEFAULT_PERIOD,
    limit: int = 30,
    fetch_func=fetch_top_ls_ratio,
) -> int:
    """1 poll synchrone + append dedup. Pour cron 5min.

    Returns: nombre de nouveaux records écrits.
    """
    records = fetch_func(symbol=symbol, period=period, limit=limit, end_time_ms=None)
    return store_records_dedup(records, store_path)


__all__ = [
    "fetch_top_ls_ratio",
    "backfill_30d",
    "poll_and_store",
    "store_records_dedup",
]
