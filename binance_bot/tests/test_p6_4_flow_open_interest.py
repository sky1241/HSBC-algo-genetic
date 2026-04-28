"""P6.4 — Tests flow_open_interest.py (Binance openInterestHist)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_open_interest import (
    _parse_records,
    backfill_30d,
    fetch_open_interest,
    poll_and_store,
    store_records_dedup,
)


def test_parse_payload_correct_format():
    """Payload bien formé → records normalisés sum_oi + sum_oi_value floats."""
    payload = [
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "20000.5",
            "sumOpenInterestValue": "1000000000.0",
            "timestamp": 1700000000000,
        },
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "21000.0",
            "sumOpenInterestValue": "1050000000.0",
            "timestamp": 1700000300000,
        },
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 2
    assert records[0]["sum_oi"] == pytest.approx(20000.5)
    assert records[0]["sum_oi_value"] == pytest.approx(1_000_000_000.0)
    assert records[1]["ts_ms"] == 1700000300000


def test_parse_payload_skips_malformed_entries():
    payload = [
        {"timestamp": 100, "sumOpenInterest": "1000", "sumOpenInterestValue": "5e7"},
        {"timestamp": 200, "sumOpenInterest": "BAD", "sumOpenInterestValue": "5e7"},
        {"timestamp": 300},
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 1
    assert records[0]["ts_ms"] == 100


def test_backfill_30d_writes_records(tmp_path):
    """backfill_30d() pagine et écrit jusqu'à 30j ou page vide."""
    store = tmp_path / "flow_oi.jsonl"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    one_hour = 3600 * 1000

    page1 = [
        {"ts_ms": now_ms - 1 * one_hour, "symbol": "BTCUSDT", "sum_oi": 20000.0, "sum_oi_value": 1e9},
        {"ts_ms": now_ms - 2 * one_hour, "symbol": "BTCUSDT", "sum_oi": 19500.0, "sum_oi_value": 0.95e9},
    ]
    page2 = [
        {"ts_ms": now_ms - 3 * one_hour, "symbol": "BTCUSDT", "sum_oi": 19000.0, "sum_oi_value": 0.9e9},
    ]
    pages = [page1, page2, []]

    def mock_fetch(symbol, period, limit, end_time_ms=None):
        return pages.pop(0) if pages else []

    n = backfill_30d("BTCUSDT", store_path=store, period="5m", fetch_func=mock_fetch)
    assert n == 3
    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_persistence_dedup(tmp_path):
    store = tmp_path / "flow_oi.jsonl"
    records_v1 = [
        {"ts_ms": 100, "symbol": "BTCUSDT", "sum_oi": 1000.0, "sum_oi_value": 5e7},
        {"ts_ms": 200, "symbol": "BTCUSDT", "sum_oi": 1100.0, "sum_oi_value": 5.5e7},
    ]
    n1 = store_records_dedup(records_v1, store)
    assert n1 == 2

    records_v2 = records_v1 + [
        {"ts_ms": 300, "symbol": "BTCUSDT", "sum_oi": 1050.0, "sum_oi_value": 5.25e7},
    ]
    n2 = store_records_dedup(records_v2, store)
    assert n2 == 1


def test_rate_limit_429_triggers_backoff_then_returns_empty():
    with patch("services.flow_open_interest.requests.get") as mock_get, \
         patch("services.flow_open_interest.time.sleep") as mock_sleep:
        resp = MagicMock()
        resp.status_code = 429
        mock_get.return_value = resp
        result = fetch_open_interest("BTCUSDT", period="5m", limit=30)
        assert result == []
        assert mock_get.call_count >= 5


def test_fetch_handles_request_exception_gracefully():
    import requests as _r
    with patch("services.flow_open_interest.requests.get", side_effect=_r.RequestException("simulated")):
        assert fetch_open_interest("BTCUSDT") == []


def test_fetch_handles_non_200_returns_empty():
    with patch("services.flow_open_interest.requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp
        assert fetch_open_interest("BTCUSDT") == []


def test_poll_and_store_uses_fetch_func(tmp_path):
    store = tmp_path / "flow_oi.jsonl"
    mock_fetch = MagicMock(return_value=[
        {"ts_ms": 100, "symbol": "BTCUSDT", "sum_oi": 1000.0, "sum_oi_value": 5e7}
    ])
    n = poll_and_store("BTCUSDT", store_path=store, period="5m", limit=30, fetch_func=mock_fetch)
    assert n == 1
    mock_fetch.assert_called_once()
