"""R3 — Tests fumée des nouveaux runners flow REST + WS daemon."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routines import flow_rest_collector
from routines import flow_liquidations_daemon


# ---------------------------------------------------------------------------
# REST collector
# ---------------------------------------------------------------------------


def test_rest_collector_safe_poll_swallows_exceptions():
    """_safe_poll catch toute exception et retourne 0 sans crash."""
    def boom(symbol, store_path):
        raise ConnectionError("simulated network failure")
    n = flow_rest_collector._safe_poll("test", boom, "BTCUSDT", Path("/tmp/x.jsonl"))
    assert n == 0


def test_rest_collector_safe_poll_returns_count_on_success():
    def ok(symbol, store_path):
        return 7
    n = flow_rest_collector._safe_poll("test", ok, "BTCUSDT", Path("/tmp/x.jsonl"))
    assert n == 7


def test_rest_collector_main_iterates_3_symbols_x_3_services(tmp_path, monkeypatch):
    """main() doit invoquer poll_top_ls, poll_taker, poll_oi pour BTC/ETH/SOL."""
    monkeypatch.setattr(flow_rest_collector, "_data_path", lambda name: tmp_path / name)

    calls = {"top_ls": [], "taker": [], "oi": []}

    def fake_top_ls(symbol, store_path):
        calls["top_ls"].append(symbol)
        return 0

    def fake_taker(symbol, store_path):
        calls["taker"].append(symbol)
        return 0

    def fake_oi(symbol, store_path):
        calls["oi"].append(symbol)
        return 0

    monkeypatch.setattr(flow_rest_collector, "poll_top_ls", fake_top_ls)
    monkeypatch.setattr(flow_rest_collector, "poll_taker", fake_taker)
    monkeypatch.setattr(flow_rest_collector, "poll_oi", fake_oi)

    rc = flow_rest_collector.main()
    assert rc == 0
    assert calls["top_ls"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert calls["taker"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert calls["oi"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_rest_collector_continues_on_partial_failure(tmp_path, monkeypatch):
    """Si 1 service raise, les autres continuent et main() return 0."""
    monkeypatch.setattr(flow_rest_collector, "_data_path", lambda name: tmp_path / name)

    def fake_top_ls(symbol, store_path):
        if symbol == "ETHUSDT":
            raise RuntimeError("simulated 429")
        return 1

    monkeypatch.setattr(flow_rest_collector, "poll_top_ls", fake_top_ls)
    monkeypatch.setattr(flow_rest_collector, "poll_taker", lambda symbol, store_path: 0)
    monkeypatch.setattr(flow_rest_collector, "poll_oi", lambda symbol, store_path: 0)

    rc = flow_rest_collector.main()
    assert rc == 0  # toujours 0 même avec partial failure


# ---------------------------------------------------------------------------
# WS daemon
# ---------------------------------------------------------------------------


def test_ws_daemon_main_starts_and_stops_cleanly(monkeypatch):
    """main() de daemon : start manager, attend signal, stop drain."""
    started = {"value": False}
    stopped = {"value": False}
    fake_manager = MagicMock()
    fake_manager.events_received = 0
    fake_manager.buckets_flushed = 0
    fake_manager.connection_errors = 0

    def fake_start():
        started["value"] = True

    def fake_stop(drain_to_jsonl=True):
        stopped["value"] = True

    fake_manager.start = fake_start
    fake_manager.stop = fake_stop

    monkeypatch.setattr(flow_liquidations_daemon, "LiquidationWSManager",
                        MagicMock(return_value=fake_manager))
    monkeypatch.setattr(flow_liquidations_daemon, "LiquidationAggregator",
                        MagicMock())

    # Mock time.sleep pour exit immédiat — on simule SIGTERM tout de suite
    sleep_calls = {"count": 0}

    def fake_sleep(t):
        sleep_calls["count"] += 1
        # Au 2e sleep, on déclenche stop_requested
        if sleep_calls["count"] >= 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(flow_liquidations_daemon.time, "sleep", fake_sleep)

    # main() catch KeyboardInterrupt via signal handler ; on simule via direct
    import signal
    handlers = {}
    monkeypatch.setattr(flow_liquidations_daemon.signal, "signal",
                        lambda sig, fn: handlers.setdefault(sig, fn))

    # Run avec exception au 2e sleep → on quitte le loop via finally
    with pytest.raises(KeyboardInterrupt):
        flow_liquidations_daemon.main()

    assert started["value"] is True
    assert stopped["value"] is True
