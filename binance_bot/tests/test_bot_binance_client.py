"""Tests pour BinanceClient (B9 rate limit + B11 time sync).

Aucun appel réseau réel — on mocke `requests.request` et `requests.get`.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.binance_client import (
    BinanceClient,
    WEIGHT_THROTTLE_THRESHOLD,
    TIME_RESYNC_INTERVAL_SEC,
)


def _resp(status=200, json_data=None, headers=None):
    """Construit un mock de réponse requests."""
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = json_data if json_data is not None else {}
    return r


# ============================================================
# B11 — Time sync
# ============================================================

def test_sync_time_computes_offset():
    """_sync_time doit calculer offset = serverTime - localTimeMid."""
    server_time = int(time.time() * 1000) + 1234  # avance de 1234 ms
    fake = _resp(json_data={"serverTime": server_time})
    with patch("bot.binance_client.requests.get", return_value=fake):
        client = BinanceClient(api_key="K", api_secret="S", auto_sync=True)
    # L'offset doit être proche de 1234 ms (à quelques ms près)
    assert abs(client.time_offset_ms() - 1234) < 50


def test_sync_time_warning_when_offset_exceeds_1000ms(caplog):
    """Si |offset| > 1000ms, on log un WARNING."""
    server_time = int(time.time() * 1000) + 5000  # 5s d'écart
    fake = _resp(json_data={"serverTime": server_time})
    import logging
    with caplog.at_level(logging.WARNING, logger="bot.binance_client"):
        with patch("bot.binance_client.requests.get", return_value=fake):
            client = BinanceClient(api_key="K", api_secret="S", auto_sync=True)
    assert any("Clock skew" in rec.message for rec in caplog.records)
    assert client.time_offset_ms() > 1000


def test_signed_request_uses_offset_in_timestamp():
    """Les requêtes signées doivent inclure timestamp = now*1000 + offset."""
    server_time = int(time.time() * 1000) + 7000  # offset ~7000ms
    sync_resp = _resp(json_data={"serverTime": server_time})

    captured_url = {}

    def _fake_request(method, url, headers=None, timeout=None):
        captured_url["url"] = url
        captured_url["headers"] = headers
        return _resp(status=200, json_data={"ok": True})

    with patch("bot.binance_client.requests.get", return_value=sync_resp):
        client = BinanceClient(api_key="K", api_secret="S", auto_sync=True)

    # Empêche resync au moment du request
    client._last_time_sync = time.time()

    with patch("bot.binance_client.requests.request", side_effect=_fake_request):
        client.signed_post("/fapi/v1/order", params={"symbol": "BTCUSDT"})

    url = captured_url["url"]
    assert "timestamp=" in url
    assert "signature=" in url
    assert captured_url["headers"]["X-MBX-APIKEY"] == "K"
    # Extraire timestamp et vérifier qu'il est ≈ now + offset
    import urllib.parse as up
    qs = up.parse_qs(url.split("?", 1)[1])
    ts = int(qs["timestamp"][0])
    expected = int(time.time() * 1000) + client.time_offset_ms()
    assert abs(ts - expected) < 500


# ============================================================
# B9 — Rate limit headers parsing
# ============================================================

def test_rate_limit_headers_parsed():
    """X-MBX-USED-WEIGHT-1m et X-MBX-ORDER-COUNT-1m sont parsés."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    fake = _resp(
        status=200,
        json_data={"ok": True},
        headers={"X-MBX-USED-WEIGHT-1m": "150", "X-MBX-ORDER-COUNT-1m": "7"},
    )
    with patch("bot.binance_client.requests.request", return_value=fake):
        client.public_get("/fapi/v1/ping")
    assert client.used_weight_1m() == 150
    assert client.order_count_1m() == 7


def test_throttle_when_used_weight_above_threshold():
    """Si used_weight > WEIGHT_THROTTLE_THRESHOLD (1920), on sleep."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    client._used_weight_1m = WEIGHT_THROTTLE_THRESHOLD + 100  # 2020
    sleeps = []
    fake = _resp(status=200, json_data={"ok": True})
    with patch("bot.binance_client.requests.request", return_value=fake), \
         patch("bot.binance_client.time.sleep", side_effect=lambda x: sleeps.append(x)):
        client.public_get("/fapi/v1/ping")
    # Au moins un sleep doit être appelé (par _maybe_throttle)
    assert len(sleeps) >= 1
    assert sleeps[0] >= 0  # entre 0 et 60s


# ============================================================
# B9 — Backoff exponentiel sur 5xx
# ============================================================

def test_backoff_exponential_on_500():
    """Sur HTTP 500, on retente avec backoff exponentiel."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    sleeps = []
    # 4 échecs 500 puis succès
    responses = [
        _resp(status=500, json_data={}),
        _resp(status=500, json_data={}),
        _resp(status=500, json_data={}),
        _resp(status=200, json_data={"ok": True}),
    ]
    with patch("bot.binance_client.requests.request", side_effect=responses), \
         patch("bot.binance_client.time.sleep", side_effect=lambda x: sleeps.append(x)):
        result = client.public_get("/fapi/v1/ping")
    assert result == {"ok": True}
    # 3 retries (3 sleeps de backoff sur 500), chacun bornés à 60s
    assert len(sleeps) >= 3
    for s in sleeps:
        assert s <= 60.0


# ============================================================
# B9 — 429 respecte Retry-After
# ============================================================

def test_429_respects_retry_after():
    """Sur HTTP 429, on attend `Retry-After` secondes."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    sleeps = []
    responses = [
        _resp(status=429, json_data={}, headers={"Retry-After": "3"}),
        _resp(status=200, json_data={"ok": True}),
    ]
    with patch("bot.binance_client.requests.request", side_effect=responses), \
         patch("bot.binance_client.time.sleep", side_effect=lambda x: sleeps.append(x)):
        result = client.public_get("/fapi/v1/ping")
    assert result == {"ok": True}
    # Le sleep de 3 secondes doit avoir eu lieu
    assert 3 in sleeps


# ============================================================
# B9 — 418 = ban auto
# ============================================================

def test_418_triggers_ban():
    """Sur HTTP 418, on set banned_until et is_banned() = True."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    fake = _resp(status=418, json_data={}, headers={"Retry-After": "120"})
    with patch("bot.binance_client.requests.request", return_value=fake):
        result = client.public_get("/fapi/v1/ping")
    assert result.get("error") == "banned"
    assert client.is_banned() is True
    # banned_until devrait être ~now + 120s
    assert client._banned_until > time.time() + 100


def test_request_refused_when_banned():
    """Si is_banned() est True, on ne fait pas la requête."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    client._banned_until = time.time() + 60  # banni pour 60s
    with patch("bot.binance_client.requests.request") as mock_req:
        result = client.public_get("/fapi/v1/ping")
    assert result.get("error") == "banned"
    mock_req.assert_not_called()


# ============================================================
# Signature : timestamp + signature présents
# ============================================================

def test_signed_post_has_signature_and_timestamp():
    """Une requête signée POST doit inclure timestamp et signature."""
    client = BinanceClient(api_key="K", api_secret="S", auto_sync=False)
    client._time_offset_ms = 500  # offset connu
    captured = {}

    def _fake(method, url, headers=None, timeout=None):
        captured["url"] = url
        captured["method"] = method
        return _resp(status=200, json_data={"algoId": 1})

    with patch("bot.binance_client.requests.request", side_effect=_fake):
        client.signed_post("/fapi/v1/algoOrder", params={"symbol": "BTCUSDT"})

    assert captured["method"] == "POST"
    assert "/fapi/v1/algoOrder?" in captured["url"]
    qs = captured["url"].split("?", 1)[1]
    import urllib.parse as up
    parsed = dict(up.parse_qsl(qs))
    assert parsed["symbol"] == "BTCUSDT"
    assert "timestamp" in parsed
    assert "recvWindow" in parsed
    assert "signature" in parsed and len(parsed["signature"]) == 64
    # Le timestamp doit refléter l'offset
    expected = int(time.time() * 1000) + 500
    assert abs(int(parsed["timestamp"]) - expected) < 500


def test_periodic_resync_after_interval():
    """Après TIME_RESYNC_INTERVAL_SEC, _maybe_resync déclenche un nouveau sync."""
    sync_resp = _resp(json_data={"serverTime": int(time.time() * 1000)})
    sync_count = {"n": 0}

    def _count_sync(*a, **kw):
        sync_count["n"] += 1
        return sync_resp

    with patch("bot.binance_client.requests.get", side_effect=_count_sync):
        client = BinanceClient(api_key="K", api_secret="S", auto_sync=True)

    assert sync_count["n"] == 1
    # Simule que la dernière sync date d'il y a > intervalle
    client._last_time_sync = time.time() - TIME_RESYNC_INTERVAL_SEC - 1

    with patch("bot.binance_client.requests.get", side_effect=_count_sync), \
         patch("bot.binance_client.requests.request",
               return_value=_resp(status=200, json_data={"ok": True})):
        client.public_get("/fapi/v1/ping")

    assert sync_count["n"] == 2  # un resync a eu lieu
