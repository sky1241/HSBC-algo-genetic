"""P6.2 — Tests flow_taker_ratio.py (Binance takerlongshortRatio)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_taker_ratio import (
    _parse_records,
    backfill_30d,
    fetch_taker_ratio,
    poll_and_store,
    store_records_dedup,
)


def test_parse_payload_correct_format():
    """Payload Binance bien formé → records normalisés buy_sell_ratio/buy_vol/sell_vol."""
    payload = [
        {
            "buySellRatio": "1.234",
            "buyVol": "1500.5",
            "sellVol": "1216.0",
            "timestamp": 1700000000000,
        },
        {
            "buySellRatio": "0.85",
            "buyVol": "850.0",
            "sellVol": "1000.0",
            "timestamp": 1700000300000,
        },
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 2
    assert records[0]["ts_ms"] == 1700000000000
    assert records[0]["buy_sell_ratio"] == pytest.approx(1.234)
    assert records[0]["buy_vol"] == pytest.approx(1500.5)
    assert records[0]["sell_vol"] == pytest.approx(1216.0)


def test_parse_payload_skips_malformed_entries():
    """Entrées sans clé requise ou valeurs non-numeric → skip silencieux."""
    payload = [
        {"timestamp": 100, "buySellRatio": "1.0", "buyVol": "100", "sellVol": "100"},
        {"timestamp": 200, "buySellRatio": "BAD", "buyVol": "100", "sellVol": "100"},
        {"timestamp": 300},  # missing fields
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 1
    assert records[0]["ts_ms"] == 100


def test_backfill_30d_writes_records(tmp_path):
    """backfill_30d() pagine et écrit les records jusqu'à 30j ou page vide."""
    store = tmp_path / "flow_taker.jsonl"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    one_hour = 3600 * 1000

    page1 = [
        {"ts_ms": now_ms - 1 * one_hour, "symbol": "BTCUSDT", "buy_sell_ratio": 1.0, "buy_vol": 1000.0, "sell_vol": 1000.0},
        {"ts_ms": now_ms - 2 * one_hour, "symbol": "BTCUSDT", "buy_sell_ratio": 0.95, "buy_vol": 950.0, "sell_vol": 1000.0},
    ]
    page2 = [
        {"ts_ms": now_ms - 3 * one_hour, "symbol": "BTCUSDT", "buy_sell_ratio": 1.05, "buy_vol": 1050.0, "sell_vol": 1000.0},
    ]
    pages = [page1, page2, []]

    def mock_fetch(symbol, period, limit, end_time_ms=None):
        return pages.pop(0) if pages else []

    n = backfill_30d("BTCUSDT", store_path=store, period="5m", fetch_func=mock_fetch)
    assert n == 3
    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_persistence_dedup(tmp_path):
    """store_records_dedup ne ré-écrit pas un ts_ms déjà présent."""
    store = tmp_path / "flow_taker.jsonl"
    records_v1 = [
        {"ts_ms": 100, "symbol": "BTCUSDT", "buy_sell_ratio": 1.0, "buy_vol": 100.0, "sell_vol": 100.0},
        {"ts_ms": 200, "symbol": "BTCUSDT", "buy_sell_ratio": 1.1, "buy_vol": 110.0, "sell_vol": 100.0},
    ]
    n1 = store_records_dedup(records_v1, store)
    assert n1 == 2

    records_v2 = records_v1 + [
        {"ts_ms": 300, "symbol": "BTCUSDT", "buy_sell_ratio": 0.95, "buy_vol": 95.0, "sell_vol": 100.0},
    ]
    n2 = store_records_dedup(records_v2, store)
    assert n2 == 1


def test_rate_limit_429_triggers_backoff_then_returns_empty():
    """HTTP 429 → backoff exponentiel; abandon après _BACKOFF_MAX_SEC."""
    with patch("services.flow_taker_ratio.requests.get") as mock_get, \
         patch("services.flow_taker_ratio.time.sleep") as mock_sleep:
        resp = MagicMock()
        resp.status_code = 429
        mock_get.return_value = resp
        result = fetch_taker_ratio("BTCUSDT", period="5m", limit=30)
        assert result == []
        assert mock_get.call_count >= 5
        assert mock_sleep.call_count >= 5


def test_fetch_handles_request_exception_gracefully():
    """RequestException → empty list, pas de crash."""
    import requests as _r
    with patch("services.flow_taker_ratio.requests.get", side_effect=_r.RequestException("simulated")):
        assert fetch_taker_ratio("BTCUSDT") == []


def test_fetch_handles_non_200_returns_empty():
    """HTTP 500 ou 404 → empty list."""
    with patch("services.flow_taker_ratio.requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp
        assert fetch_taker_ratio("BTCUSDT") == []


def test_poll_and_store_uses_fetch_func(tmp_path):
    """poll_and_store accepte fetch_func injectable."""
    store = tmp_path / "flow_taker.jsonl"
    mock_fetch = MagicMock(return_value=[
        {"ts_ms": 100, "symbol": "BTCUSDT", "buy_sell_ratio": 1.0, "buy_vol": 100.0, "sell_vol": 100.0}
    ])
    n = poll_and_store("BTCUSDT", store_path=store, period="5m", limit=30, fetch_func=mock_fetch)
    assert n == 1
    mock_fetch.assert_called_once()
