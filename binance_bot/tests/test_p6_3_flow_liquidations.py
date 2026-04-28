"""P6.3 — Tests flow_liquidations.py (Binance !forceOrder@arr WS + buckets 1min).

Vérifie:
  1. Parsing event Binance forceOrder → record normalisé (long_liq/short_liq)
  2. Bucket aggregator : sum notional par side, count, key floor 1min
  3. Flush automatique quand event d'un bucket plus récent arrive
  4. drain() final pour les buckets en mémoire à la fermeture
  5. Append jsonl avec 1 ligne par bucket
  6. Reconnect simulation (mock WSManager start/stop sans WS réel)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_liquidations import (
    LiquidationAggregator,
    LiquidationBucket,
    LiquidationWSManager,
    append_buckets_jsonl,
    bucket_key_ms,
    parse_force_order_event,
)


# ---------------------------------------------------------------------------
# parse_force_order_event
# ---------------------------------------------------------------------------


def test_parse_event_sell_side_translates_to_long_liq():
    """side=SELL Binance = liquidation forcée d'un LONG → side=long_liq normalisé."""
    msg = {
        "e": "forceOrder",
        "E": 1700000000000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "q": "0.5",
            "p": "50000",
            "ap": "50000",
            "T": 1700000000123,
        },
    }
    rec = parse_force_order_event(msg)
    assert rec is not None
    assert rec["symbol"] == "BTCUSDT"
    assert rec["side"] == "long_liq"
    assert rec["qty"] == pytest.approx(0.5)
    assert rec["price"] == pytest.approx(50000)
    assert rec["notional"] == pytest.approx(25000)
    assert rec["ts_ms"] == 1700000000123


def test_parse_event_buy_side_translates_to_short_liq():
    """side=BUY Binance = liquidation d'un SHORT → side=short_liq."""
    msg = {
        "e": "forceOrder",
        "E": 1700000000000,
        "o": {
            "s": "ETHUSDT",
            "S": "BUY",
            "q": "10",
            "p": "2000",
            "ap": "2000",
            "T": 1700000000456,
        },
    }
    rec = parse_force_order_event(msg)
    assert rec["side"] == "short_liq"
    assert rec["notional"] == pytest.approx(20000)


def test_parse_event_skips_non_force_order():
    """Event sans 'e'='forceOrder' → None."""
    msg = {"e": "trade", "o": {"s": "BTCUSDT", "S": "BUY", "q": "1", "p": "100", "T": 1}}
    assert parse_force_order_event(msg) is None


def test_parse_event_handles_string_input():
    """Accepte msg en string JSON."""
    msg_str = json.dumps({
        "e": "forceOrder", "E": 100,
        "o": {"s": "BTCUSDT", "S": "SELL", "q": "1", "p": "100", "ap": "100", "T": 100},
    })
    rec = parse_force_order_event(msg_str)
    assert rec is not None
    assert rec["notional"] == 100


def test_parse_event_returns_none_on_malformed():
    """JSON invalide ou champs manquants → None sans crash."""
    assert parse_force_order_event("not json") is None
    assert parse_force_order_event({"e": "forceOrder", "o": {}}) is None
    # qty négative
    assert parse_force_order_event({
        "e": "forceOrder", "E": 100,
        "o": {"s": "X", "S": "SELL", "q": "-1", "p": "100", "T": 100}
    }) is None


# ---------------------------------------------------------------------------
# bucket_key_ms / LiquidationBucket
# ---------------------------------------------------------------------------


def test_bucket_key_floors_to_minute():
    """ts=1700000123456 → floor à minute (1700000100000 = 1700000123456 - 23456)."""
    ts = 1700000123456
    key = bucket_key_ms(ts, bucket_size_ms=60_000)
    assert key == ts - (ts % 60_000)
    assert key % 60_000 == 0


def test_liquidation_bucket_aggregates_correctly():
    """add_event() somme notional par side et incrémente count."""
    b = LiquidationBucket(bucket_start_ms=1700000000000, symbol="BTCUSDT")
    b.add_event({"side": "long_liq", "notional": 1000.0})
    b.add_event({"side": "long_liq", "notional": 500.0})
    b.add_event({"side": "short_liq", "notional": 200.0})
    assert b.long_liq_notional == 1500.0
    assert b.long_liq_count == 2
    assert b.short_liq_notional == 200.0
    assert b.short_liq_count == 1


# ---------------------------------------------------------------------------
# LiquidationAggregator
# ---------------------------------------------------------------------------


def test_aggregator_flushes_on_newer_bucket():
    """Quand un event d'un bucket plus récent arrive, le précédent est flushé.

    On utilise ts simples qui tombent pile sur les bucket boundaries 60000ms:
    - rec1 ts=60000 → bucket=60000
    - rec2 ts=119999 → bucket=60000 (même bucket)
    - rec3 ts=120000 → bucket=120000 (nouveau bucket → flush)
    """
    agg = LiquidationAggregator(symbols={"BTCUSDT"})
    rec1 = {"ts_ms": 60000, "symbol": "BTCUSDT", "side": "long_liq", "notional": 1000.0}
    rec2 = {"ts_ms": 119999, "symbol": "BTCUSDT", "side": "long_liq", "notional": 500.0}
    rec3 = {"ts_ms": 120000, "symbol": "BTCUSDT", "side": "short_liq", "notional": 200.0}

    assert agg.ingest(rec1) == []  # 1er event, rien à flush
    assert agg.ingest(rec2) == []  # même bucket 60000, accumulation silencieuse
    flushed = agg.ingest(rec3)     # nouveau bucket 120000 → flush l'ancien
    assert len(flushed) == 1
    b = flushed[0]
    assert b["bucket_start_ms"] == 60000
    assert b["long_liq_notional"] == 1500.0
    assert b["long_liq_count"] == 2


def test_aggregator_filters_symbols():
    """symbols={...} ignore les events d'autres symboles."""
    agg = LiquidationAggregator(symbols={"BTCUSDT"})
    rec_eth = {"ts_ms": 1700000000000, "symbol": "ETHUSDT", "side": "long_liq", "notional": 999.0}
    assert agg.ingest(rec_eth) == []
    # drain() ne retourne rien non plus
    assert agg.drain() == []


def test_aggregator_drain_returns_remaining():
    """drain() retourne les buckets en mémoire (pour shutdown propre)."""
    agg = LiquidationAggregator()
    rec = {"ts_ms": 1700000000000, "symbol": "BTCUSDT", "side": "long_liq", "notional": 1000.0}
    agg.ingest(rec)
    remaining = agg.drain()
    assert len(remaining) == 1
    assert remaining[0]["long_liq_notional"] == 1000.0
    # Après drain, plus rien
    assert agg.drain() == []


# ---------------------------------------------------------------------------
# append_buckets_jsonl
# ---------------------------------------------------------------------------


def test_append_buckets_jsonl_writes_one_line_per_bucket(tmp_path):
    store = tmp_path / "flow_liq.jsonl"
    buckets = [
        {"bucket_start_ms": 100, "symbol": "BTCUSDT", "long_liq_notional": 1000.0,
         "short_liq_notional": 0, "long_liq_count": 1, "short_liq_count": 0},
        {"bucket_start_ms": 160, "symbol": "BTCUSDT", "long_liq_notional": 500.0,
         "short_liq_notional": 200.0, "long_liq_count": 1, "short_liq_count": 1},
    ]
    n = append_buckets_jsonl(buckets, store)
    assert n == 2
    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["bucket_start_ms"] == 100


def test_append_buckets_jsonl_empty_returns_zero(tmp_path):
    """Liste vide → no write, retourne 0."""
    store = tmp_path / "flow_liq.jsonl"
    assert append_buckets_jsonl([], store) == 0
    assert not store.exists()  # pas créé si rien à écrire


# ---------------------------------------------------------------------------
# WSManager (avec mock websocket.WebSocketApp)
# ---------------------------------------------------------------------------


def test_wsmanager_on_message_appends_to_aggregator(tmp_path):
    """on_message() parse l'event, ingest dans aggregator, flush si bucket complet."""
    store = tmp_path / "flow_liq.jsonl"
    agg = LiquidationAggregator(symbols={"BTCUSDT"})
    mgr = LiquidationWSManager(aggregator=agg, store_path=store)

    msg1 = json.dumps({
        "e": "forceOrder", "E": 1700000000000,
        "o": {"s": "BTCUSDT", "S": "SELL", "q": "1", "p": "100", "ap": "100", "T": 1700000000000},
    })
    mgr._on_message(None, msg1)  # bucket 1700000000000
    assert mgr.events_received == 1

    msg2 = json.dumps({
        "e": "forceOrder", "E": 1700000060000,
        "o": {"s": "BTCUSDT", "S": "BUY", "q": "2", "p": "100", "ap": "100", "T": 1700000060000},
    })
    mgr._on_message(None, msg2)  # bucket suivant → flush du précédent
    assert mgr.events_received == 2
    assert mgr.buckets_flushed == 1
    # 1 ligne dans le jsonl
    lines = store.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["long_liq_notional"] == 100.0


def test_wsmanager_drain_on_stop(tmp_path):
    """stop(drain_to_jsonl=True) flush les buckets restants."""
    store = tmp_path / "flow_liq.jsonl"
    agg = LiquidationAggregator()
    mgr = LiquidationWSManager(aggregator=agg, store_path=store)

    msg = json.dumps({
        "e": "forceOrder", "E": 100,
        "o": {"s": "BTCUSDT", "S": "SELL", "q": "1", "p": "100", "ap": "100", "T": 100},
    })
    mgr._on_message(None, msg)
    mgr.stop(drain_to_jsonl=True)  # n'a pas démarré le thread mais drain quand même
    assert mgr.buckets_flushed == 1
