"""P6.1 — Tests flow_top_ls.py (Binance topLongShortPositionRatio).

Vérifie:
  1. Parsing payload Binance → records normalisés
  2. Backfill itère pages jusqu'à 30j ou page vide
  3. Persistence jsonl avec dedup par ts_ms (pas de duplicats)
  4. Rate limit 429 déclenche backoff (mocked)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_top_ls import (
    _parse_records,
    backfill_30d,
    fetch_top_ls_ratio,
    poll_and_store,
    store_records_dedup,
)


# ---------------------------------------------------------------------------
# Tests obligatoires
# ---------------------------------------------------------------------------


def test_parse_payload_correct_format():
    """Payload Binance bien formé → records normalisés ts_ms/symbol/ratios floats."""
    payload = [
        {
            "symbol": "BTCUSDT",
            "longShortRatio": "1.234",
            "longAccount": "0.55",
            "shortAccount": "0.45",
            "timestamp": 1700000000000,
        },
        {
            "symbol": "BTCUSDT",
            "longShortRatio": "0.85",
            "longAccount": "0.46",
            "shortAccount": "0.54",
            "timestamp": 1700000300000,  # +5min
        },
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 2
    assert records[0]["ts_ms"] == 1700000000000
    assert records[0]["long_short_ratio"] == pytest.approx(1.234)
    assert records[0]["long_account"] == pytest.approx(0.55)
    assert records[1]["ts_ms"] == 1700000300000


def test_parse_payload_skips_malformed_entries():
    """Entrées sans clé requise ou valeurs non-numeric → skip silencieux."""
    payload = [
        {"timestamp": 100, "longShortRatio": "1.0", "longAccount": "0.5", "shortAccount": "0.5", "symbol": "BTCUSDT"},
        {"timestamp": 200, "longShortRatio": "BADVAL", "longAccount": "0.5", "shortAccount": "0.5"},  # NaN-like
        {"timestamp": 300},  # missing fields
    ]
    records = _parse_records(payload, symbol="BTCUSDT")
    assert len(records) == 1
    assert records[0]["ts_ms"] == 100


def test_backfill_30d_writes_records(tmp_path):
    """backfill_30d() pagine et écrit les records jusqu'à atteindre 30j ou page vide."""
    from datetime import datetime, timezone
    store = tmp_path / "flow.jsonl"

    # Use timestamps récents (now - 1h, now - 2h, now - 3h) → tous dans la fenêtre 30j
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    one_hour = 3600 * 1000

    page1 = [
        {"ts_ms": now_ms - 1 * one_hour, "symbol": "BTCUSDT", "long_short_ratio": 1.0, "long_account": 0.5, "short_account": 0.5},
        {"ts_ms": now_ms - 2 * one_hour, "symbol": "BTCUSDT", "long_short_ratio": 1.1, "long_account": 0.5, "short_account": 0.5},
    ]
    page2 = [
        {"ts_ms": now_ms - 3 * one_hour, "symbol": "BTCUSDT", "long_short_ratio": 0.9, "long_account": 0.5, "short_account": 0.5},
    ]
    pages = [page1, page2, []]

    def mock_fetch(symbol, period, limit, end_time_ms=None):
        return pages.pop(0) if pages else []

    n = backfill_30d("BTCUSDT", store_path=store, period="5m", fetch_func=mock_fetch)
    assert n == 3

    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    ts_set = {p["ts_ms"] for p in parsed}
    assert ts_set == {now_ms - 1 * one_hour, now_ms - 2 * one_hour, now_ms - 3 * one_hour}


def test_persistence_dedup(tmp_path):
    """store_records_dedup ne ré-écrit pas un ts_ms déjà présent."""
    store = tmp_path / "flow.jsonl"
    records_v1 = [
        {"ts_ms": 100, "symbol": "BTCUSDT", "long_short_ratio": 1.0, "long_account": 0.5, "short_account": 0.5},
        {"ts_ms": 200, "symbol": "BTCUSDT", "long_short_ratio": 1.1, "long_account": 0.5, "short_account": 0.5},
    ]
    n1 = store_records_dedup(records_v1, store)
    assert n1 == 2

    # Re-injection avec doublons + 1 nouveau
    records_v2 = records_v1 + [
        {"ts_ms": 300, "symbol": "BTCUSDT", "long_short_ratio": 1.2, "long_account": 0.5, "short_account": 0.5},
    ]
    n2 = store_records_dedup(records_v2, store)
    assert n2 == 1  # seul le 300 est nouveau

    # Verify file content
    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3


def test_rate_limit_429_triggers_backoff_then_returns_empty(tmp_path):
    """HTTP 429 → backoff exponentiel; après _BACKOFF_MAX_SEC → empty list (no infinite loop)."""
    # Mock requests.get pour retourner 429 systématiquement
    with patch("services.flow_top_ls.requests.get") as mock_get, \
         patch("services.flow_top_ls.time.sleep") as mock_sleep:  # éviter waits réels
        resp = MagicMock()
        resp.status_code = 429
        mock_get.return_value = resp

        result = fetch_top_ls_ratio("BTCUSDT", period="5m", limit=30)
        # backoff doit s'arrêter à _BACKOFF_MAX_SEC=60s : 1+2+4+8+16+32 = 63 → ~6 retries
        assert result == []
        assert mock_get.call_count >= 5  # au moins quelques retries
        assert mock_sleep.call_count >= 5


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_fetch_handles_request_exception_gracefully(tmp_path):
    """RequestException (timeout, DNS, etc.) → empty list, pas de crash."""
    import requests as _r
    with patch("services.flow_top_ls.requests.get", side_effect=_r.RequestException("simulated")):
        result = fetch_top_ls_ratio("BTCUSDT", period="5m")
        assert result == []


def test_fetch_handles_non_200_returns_empty():
    """HTTP 500 ou 404 → empty list (no retry, no crash)."""
    with patch("services.flow_top_ls.requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp
        assert fetch_top_ls_ratio("BTCUSDT") == []


def test_poll_and_store_uses_fetch_func(tmp_path):
    """poll_and_store accepte fetch_func injectable pour tests."""
    store = tmp_path / "flow.jsonl"
    mock_fetch = MagicMock(return_value=[
        {"ts_ms": 100, "symbol": "BTCUSDT", "long_short_ratio": 1.0, "long_account": 0.5, "short_account": 0.5}
    ])
    n = poll_and_store("BTCUSDT", store_path=store, period="5m", limit=30, fetch_func=mock_fetch)
    assert n == 1
    mock_fetch.assert_called_once()
