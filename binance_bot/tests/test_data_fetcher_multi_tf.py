"""P-MTF-1 — Tests du fetcher multi-TF avec cache TTL.

Mocke le client CCXT pour ne pas dépendre du réseau / de Binance.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from binance_bot.services.data_fetcher import DataFetcher


def _fake_candles(start_ms: int, n: int, tf_ms: int) -> list[list]:
    """Génère N bougies fictives pour un mock CCXT."""
    return [
        [start_ms + i * tf_ms, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000.0 + i]
        for i in range(n)
    ]


@pytest.fixture
def fetcher_mocked(monkeypatch):
    """DataFetcher avec exchange CCXT mocké (pas d'appel réseau)."""
    DataFetcher._clear_cache()
    monkeypatch.setenv("BINANCE_API_KEY", "fake")
    monkeypatch.setenv("BINANCE_API_SECRET", "fake")
    f = DataFetcher("BTC/USDT", "2h")
    f.exchange = MagicMock()
    return f


def test_get_ohlcv_multi_tf_returns_dict_per_tf(fetcher_mocked):
    """Retourne un dict avec une entrée DataFrame par TF demandé."""
    now_ms = int(time.time() * 1000)
    # Bougies dans le passé (closed) pour ne pas être filtrées
    fetcher_mocked.exchange.fetch_ohlcv.side_effect = lambda symbol, timeframe, limit: (
        _fake_candles(now_ms - 300 * 7_200_000, 100, 7_200_000)  # 2h
        if timeframe == "2h"
        else _fake_candles(now_ms - 300 * 900_000, 100, 900_000)  # 15m
    )
    result = fetcher_mocked.get_ohlcv_multi_tf(["2h", "15m"], limit=100)
    assert set(result.keys()) == {"2h", "15m"}
    assert all(isinstance(df, pd.DataFrame) for df in result.values())
    assert all(len(df) > 0 for df in result.values())
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 2


def test_cache_hit_avoids_refetch(fetcher_mocked):
    """2e appel dans le TTL → pas de nouvel appel CCXT."""
    now_ms = int(time.time() * 1000)
    fetcher_mocked.exchange.fetch_ohlcv.return_value = _fake_candles(
        now_ms - 300 * 900_000, 100, 900_000
    )
    fetcher_mocked.get_ohlcv_multi_tf(["15m"], limit=100)
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 1
    # 2e appel rapide = cache hit (TTL 15m = 60s par défaut)
    fetcher_mocked.get_ohlcv_multi_tf(["15m"], limit=100)
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 1


def test_cache_miss_after_ttl(fetcher_mocked):
    """Appel après expiration du TTL → re-fetch."""
    now_ms = int(time.time() * 1000)
    fetcher_mocked.exchange.fetch_ohlcv.return_value = _fake_candles(
        now_ms - 300 * 900_000, 100, 900_000
    )
    # TTL court pour test
    fetcher_mocked.get_ohlcv_multi_tf(
        ["15m"], limit=100, cache_ttl_seconds={"15m": 1}
    )
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 1
    time.sleep(1.2)
    fetcher_mocked.get_ohlcv_multi_tf(
        ["15m"], limit=100, cache_ttl_seconds={"15m": 1}
    )
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 2


def test_use_cache_false_forces_refetch(fetcher_mocked):
    """use_cache=False → bypass cache même si TTL non expiré."""
    now_ms = int(time.time() * 1000)
    fetcher_mocked.exchange.fetch_ohlcv.return_value = _fake_candles(
        now_ms - 300 * 900_000, 100, 900_000
    )
    fetcher_mocked.get_ohlcv_multi_tf(["15m"], limit=100)
    fetcher_mocked.get_ohlcv_multi_tf(["15m"], limit=100, use_cache=False)
    assert fetcher_mocked.exchange.fetch_ohlcv.call_count == 2


def test_fetch_failure_skips_tf_does_not_crash(fetcher_mocked):
    """Si un TF crash côté CCXT, on continue sans ce TF (pas d'exception)."""
    now_ms = int(time.time() * 1000)

    def side_effect(symbol, timeframe, limit):
        if timeframe == "15m":
            raise Exception("Binance hiccup")
        return _fake_candles(now_ms - 300 * 7_200_000, 100, 7_200_000)

    fetcher_mocked.exchange.fetch_ohlcv.side_effect = side_effect
    result = fetcher_mocked.get_ohlcv_multi_tf(["2h", "15m"], limit=100)
    assert "2h" in result
    assert "15m" not in result  # skip silencieux, pas dans le dict


def test_backward_compat_get_ohlcv_unchanged(fetcher_mocked):
    """L'API existante get_ohlcv() n'est pas affectée."""
    now_ms = int(time.time() * 1000)
    fetcher_mocked.exchange.fetch_ohlcv.return_value = _fake_candles(
        now_ms - 300 * 7_200_000, 100, 7_200_000
    )
    df = fetcher_mocked.get_ohlcv(limit=100)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert len(df) > 0
