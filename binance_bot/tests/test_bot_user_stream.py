"""Tests pour UserStreamListener (BUG-B13).

Aucune connexion réelle — on mocke `requests.request` et `websocket.WebSocketApp`.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.user_stream import UserStreamListener, KEEPALIVE_INTERVAL_SEC


# ------------------------------------------------------------------ helpers

def _resp(status=200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _make_listener(callback=None):
    """UserStreamListener avec clés bidons pour tests."""
    return UserStreamListener(
        api_key="K_TEST",
        api_secret="S_TEST",
        base_rest="https://testnet.binancefuture.com",
        base_ws="wss://stream.binancefuture.com/ws",
        on_event_callback=callback,
    )


# ============================================================ listenKey REST

def test_create_listen_key_calls_post_signed():
    """create_listen_key fait un POST signé avec X-MBX-APIKEY."""
    captured = {}

    def _fake(method, url, headers=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        return _resp(json_data={"listenKey": "abc123def456"})

    listener = _make_listener()
    with patch("bot.user_stream.requests.request", side_effect=_fake):
        lk = listener.create_listen_key()

    assert lk == "abc123def456"
    assert captured["method"] == "POST"
    assert "/fapi/v1/listenKey" in captured["url"]
    assert "timestamp=" in captured["url"]
    assert "signature=" in captured["url"]
    assert captured["headers"]["X-MBX-APIKEY"] == "K_TEST"
    assert listener._listen_key == "abc123def456"


def test_create_listen_key_returns_none_on_error():
    listener = _make_listener()
    with patch("bot.user_stream.requests.request",
               return_value=_resp(status=401, json_data={"code": -2014, "msg": "bad key"})):
        lk = listener.create_listen_key()
    assert lk is None


def test_keepalive_listen_key_sends_put():
    """keepalive_listen_key fait un PUT avec listenKey en query."""
    captured = {}

    def _fake(method, url, headers=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        return _resp(json_data={})  # success: empty dict

    listener = _make_listener()
    listener._listen_key = "lk_xyz"
    with patch("bot.user_stream.requests.request", side_effect=_fake):
        ok = listener.keepalive_listen_key()

    assert ok is True
    assert captured["method"] == "PUT"
    assert "listenKey=lk_xyz" in captured["url"]


def test_close_listen_key_sends_delete_and_clears_state():
    captured = {}

    def _fake(method, url, headers=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        return _resp(json_data={})

    listener = _make_listener()
    listener._listen_key = "lk_xyz"
    with patch("bot.user_stream.requests.request", side_effect=_fake):
        ok = listener.close_listen_key()

    assert ok is True
    assert captured["method"] == "DELETE"
    assert listener._listen_key is None


# ============================================================ on_message dispatch

def test_on_message_dispatches_order_trade_update_to_callback():
    received = []
    listener = _make_listener(callback=lambda ev: received.append(ev))

    payload = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1234567890,
        "o": {"s": "BTCUSDT", "S": "BUY", "i": 99, "x": "TRADE", "X": "FILLED"},
    }
    fake_ws = MagicMock()
    listener._on_message(fake_ws, json.dumps(payload))

    assert len(received) == 1
    assert received[0]["e"] == "ORDER_TRADE_UPDATE"
    assert received[0]["o"]["s"] == "BTCUSDT"
    assert listener.events_received == 1


def test_on_message_dispatches_account_update():
    received = []
    listener = _make_listener(callback=lambda ev: received.append(ev))

    payload = {
        "e": "ACCOUNT_UPDATE",
        "E": 1234567890,
        "a": {"B": [{"a": "USDT", "wb": "1000.0"}], "P": []},
    }
    fake_ws = MagicMock()
    listener._on_message(fake_ws, json.dumps(payload))
    assert received[0]["e"] == "ACCOUNT_UPDATE"


def test_on_message_handles_invalid_json():
    received = []
    listener = _make_listener(callback=lambda ev: received.append(ev))
    fake_ws = MagicMock()
    listener._on_message(fake_ws, "not-json")
    assert received == []


def test_on_message_callback_exception_does_not_propagate():
    def bad_cb(ev):
        raise RuntimeError("boom")

    listener = _make_listener(callback=bad_cb)
    fake_ws = MagicMock()
    # Ne doit PAS lever
    listener._on_message(fake_ws, json.dumps({"e": "ACCOUNT_UPDATE"}))


# ============================================================ listenKeyExpired → reconnect

def test_listen_key_expired_triggers_reconnect_flag_and_close():
    listener = _make_listener()
    fake_ws = MagicMock()
    listener._on_message(fake_ws, json.dumps({"e": "listenKeyExpired"}))
    assert listener._reconnect_requested.is_set()
    fake_ws.close.assert_called_once()


def test_ws_loop_recreates_listen_key_after_expiration():
    """Quand _reconnect_requested est set, _ws_loop appelle create_listen_key."""
    listener = _make_listener()
    listener._listen_key = "OLD_KEY"
    listener._reconnect_requested.set()

    create_calls = {"n": 0}

    def fake_create():
        create_calls["n"] += 1
        listener._listen_key = f"NEW_KEY_{create_calls['n']}"
        return listener._listen_key

    # WebSocketApp mock dont run_forever rend la main immédiatement
    fake_ws = MagicMock()
    fake_ws.run_forever = MagicMock(side_effect=lambda **kw: listener._stop_event.set())

    with patch.object(listener, "create_listen_key", side_effect=fake_create), \
         patch("bot.user_stream.websocket.WebSocketApp", return_value=fake_ws):
        listener._ws_loop()

    # Au moins un create suite à reconnect_requested
    assert create_calls["n"] >= 1
    assert listener._listen_key.startswith("NEW_KEY_")


# ============================================================ keepalive timing

def test_keepalive_loop_calls_keepalive_after_interval():
    """Le keepalive_loop appelle keepalive_listen_key périodiquement."""
    listener = _make_listener()
    listener._listen_key = "lk_test"

    call_count = {"n": 0}

    def fake_keepalive():
        call_count["n"] += 1
        if call_count["n"] >= 2:
            listener._stop_event.set()
        return True

    # Patch le wait pour qu'il rende la main immédiatement (simule l'écoulement
    # du temps), et patch keepalive_listen_key pour compter.
    real_wait = listener._stop_event.wait

    def fast_wait(timeout=None):
        # Si on demande KEEPALIVE_INTERVAL_SEC → on retourne False immédiatement
        # (simule timeout écoulé) sauf si stop_event est set.
        if listener._stop_event.is_set():
            return True
        return False  # comme un timeout écoulé sans stop

    with patch.object(listener, "keepalive_listen_key", side_effect=fake_keepalive), \
         patch.object(listener._stop_event, "wait", side_effect=fast_wait):
        listener._keepalive_loop()

    assert call_count["n"] >= 2


def test_keepalive_constant_is_30_minutes():
    assert KEEPALIVE_INTERVAL_SEC == 30 * 60


# ============================================================ start / stop lifecycle

def test_start_then_stop_closes_ws_and_listen_key():
    listener = _make_listener()

    # Mock REST: create + keepalive + delete
    def _fake_request(method, url, headers=None, timeout=None):
        if method == "POST":
            return _resp(json_data={"listenKey": "lk_started"})
        return _resp(json_data={})

    # Mock WebSocketApp.run_forever pour bloquer jusqu'à close()
    closed_event = threading.Event()
    fake_ws = MagicMock()

    def fake_run_forever(**kw):
        closed_event.wait(timeout=5)

    def fake_close():
        closed_event.set()

    fake_ws.run_forever = fake_run_forever
    fake_ws.close = fake_close

    with patch("bot.user_stream.requests.request", side_effect=_fake_request), \
         patch("bot.user_stream.websocket.WebSocketApp", return_value=fake_ws):
        listener.start()
        # Laisse un peu de temps pour que le thread WS démarre
        time.sleep(0.2)
        listener.stop(join_timeout=3.0)

    # listenKey doit être effacé (DELETE appelé)
    assert listener._listen_key is None
    # Threads stoppés
    assert listener._ws_thread is None
    assert listener._keepalive_thread is None


def test_start_is_idempotent():
    listener = _make_listener()
    closed_event = threading.Event()
    fake_ws = MagicMock()
    fake_ws.run_forever = lambda **kw: closed_event.wait(timeout=5)
    fake_ws.close = lambda: closed_event.set()

    def _fake_request(method, url, headers=None, timeout=None):
        if method == "POST":
            return _resp(json_data={"listenKey": "lk_a"})
        return _resp(json_data={})

    with patch("bot.user_stream.requests.request", side_effect=_fake_request), \
         patch("bot.user_stream.websocket.WebSocketApp", return_value=fake_ws):
        listener.start()
        first_thread = listener._ws_thread
        listener.start()  # second call ignoré
        assert listener._ws_thread is first_thread
        listener.stop(join_timeout=3.0)


def test_stop_without_start_does_not_raise():
    listener = _make_listener()
    # Si DELETE est appelé sans listenKey → simplement skip
    listener.stop(join_timeout=1.0)


# ============================================================ backoff bounded

def test_backoff_delay_bounded():
    """Le backoff est borné à RECONNECT_BACKOFF_MAX."""
    from bot.user_stream import RECONNECT_BACKOFF_MAX
    for attempt in [0, 1, 2, 5, 10, 100]:
        d = UserStreamListener._backoff_delay(attempt)
        assert 0 < d <= RECONNECT_BACKOFF_MAX
