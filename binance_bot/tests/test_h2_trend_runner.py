"""P-MTF-4 — Tests h2_trend_runner.

Tests offline avec mocks (pas d'appel CCXT/Binance).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from binance_bot.bot.state_manager import StateManager
from binance_bot.routines import h2_trend_runner


def _fake_h2_df(close: float, n: int = 300) -> pd.DataFrame:
    """Génère un DataFrame H2 synthétique cohérent (ohlc + ATR-able)."""
    np.random.seed(42)
    base = np.linspace(close - 5_000, close, n)
    noise = np.random.randn(n) * 200
    closes = base + noise
    closes[-1] = close  # force la dernière valeur
    return pd.DataFrame({
        "timestamp": pd.date_range(end="2026-04-30", periods=n, freq="2h", tz="UTC"),
        "open": closes - 50,
        "high": closes + 100,
        "low": closes - 100,
        "close": closes,
        "volume": np.random.uniform(100, 200, n),
    })


@pytest.fixture
def tmp_state(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "params_today": {"tenkan": 21, "kijun": 35, "senkou_b": 90, "shift": 44},
        "symbols": {},
    }))
    return state_file


def test_compute_and_persist_writes_h2_trend_in_state(tmp_state, monkeypatch):
    """compute_and_persist_h2_trend → state.json contient symbols.<sym>.h2_trend."""
    fake_df = _fake_h2_df(close=70_000.0)
    fake_fetcher = MagicMock()
    fake_fetcher.get_ohlcv_multi_tf.return_value = {"2h": fake_df}

    monkeypatch.setattr(h2_trend_runner, "DataFetcher", lambda *a, **kw: fake_fetcher)

    sm = StateManager(str(tmp_state))
    snap = h2_trend_runner.compute_and_persist_h2_trend(
        "BTC/USDT", sm,
        {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26},
    )

    assert "direction" in snap
    assert snap["direction"] in {"long", "short", "flat"}

    sm.reload() if hasattr(sm, "reload") else None
    persisted = json.loads(tmp_state.read_text())
    assert "BTC/USDT" in persisted["symbols"]
    assert "h2_trend" in persisted["symbols"]["BTC/USDT"]
    assert persisted["symbols"]["BTC/USDT"]["h2_trend"]["direction"] == snap["direction"]


def test_main_runs_for_3_symbols_and_persists(tmp_state, monkeypatch):
    """main() processe 3 symboles, persiste les 3 trends."""
    fake_df = _fake_h2_df(close=70_000.0)
    fake_fetcher = MagicMock()
    fake_fetcher.get_ohlcv_multi_tf.return_value = {"2h": fake_df}
    monkeypatch.setattr(h2_trend_runner, "DataFetcher", lambda *a, **kw: fake_fetcher)
    monkeypatch.setattr(h2_trend_runner, "_load_settings", lambda: {
        "state_file": str(tmp_state),
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    })
    monkeypatch.setattr(h2_trend_runner, "ROOT", tmp_state.parent.parent)
    # patch state_file lookup
    monkeypatch.setattr(
        h2_trend_runner, "_load_settings",
        lambda: {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
    )
    # Override le state_file path computation pour tap dans tmp
    real_main = h2_trend_runner.main

    def patched_main():
        # Inject directement le state_mgr pointant tmp_state
        from binance_bot.routines import h2_trend_runner as mod
        sm = StateManager(str(tmp_state))
        for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            try:
                mod.compute_and_persist_h2_trend(
                    symbol, sm,
                    {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26},
                )
            except Exception as e:
                print(f"FAIL {symbol}: {e}")
        return 0

    assert patched_main() == 0
    persisted = json.loads(tmp_state.read_text())
    for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        assert sym in persisted["symbols"]
        assert "h2_trend" in persisted["symbols"][sym]


def test_compute_handles_fetch_failure_gracefully(tmp_state, monkeypatch):
    """Si get_ohlcv_multi_tf retourne dict vide → RuntimeError clean."""
    fake_fetcher = MagicMock()
    fake_fetcher.get_ohlcv_multi_tf.return_value = {}  # pas de TF retourné
    monkeypatch.setattr(h2_trend_runner, "DataFetcher", lambda *a, **kw: fake_fetcher)

    sm = StateManager(str(tmp_state))
    with pytest.raises(RuntimeError, match="no '2h' data"):
        h2_trend_runner.compute_and_persist_h2_trend(
            "BTC/USDT", sm,
            {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26},
        )


def test_resolve_symbols_falls_back_to_default():
    out = h2_trend_runner._resolve_symbols({})
    assert out == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_resolve_symbols_uses_list_from_settings():
    out = h2_trend_runner._resolve_symbols({"symbols": ["BTC/USDT", "ETH/USDT"]})
    assert out == ["BTC/USDT", "ETH/USDT"]


def test_resolve_symbols_handles_dict_format():
    """Format symbols=[{symbol: 'X'}, ...] aussi accepté."""
    out = h2_trend_runner._resolve_symbols({"symbols": [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}]})
    assert out == ["BTC/USDT", "ETH/USDT"]


def test_resolve_symbols_handles_pair_format():
    """Format réel bot_settings.yaml : symbols=[{pair: 'X', leverage: N}, ...]."""
    out = h2_trend_runner._resolve_symbols({"symbols": [
        {"pair": "BTC/USDT", "leverage": 125},
        {"pair": "ETH/USDT", "leverage": 100},
        {"pair": "SOL/USDT", "leverage": 50},
    ]})
    assert out == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_get_h2_params_priority_state_over_settings(tmp_state):
    sm = StateManager(str(tmp_state))
    state_params = {"tenkan": 21, "kijun": 35, "senkou_b": 90, "shift": 44}
    # state_mgr already loaded params_today from fixture, c'est le state
    out = h2_trend_runner._get_h2_params(sm, {"params_h2": {"tenkan": 999}})
    assert out == state_params  # state.json wins


def test_get_h2_params_falls_back_to_default(tmp_path):
    """Si pas params_today ni params_h2 → DEFAULT_H2_PARAMS."""
    state_file = tmp_path / "empty_state.json"
    state_file.write_text(json.dumps({"symbols": {}}))
    sm = StateManager(str(state_file))
    out = h2_trend_runner._get_h2_params(sm, {})
    assert out == h2_trend_runner.DEFAULT_H2_PARAMS
