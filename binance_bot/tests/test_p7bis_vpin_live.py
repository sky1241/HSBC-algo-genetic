"""P7-bis — Tests collecteur VPIN live (VPINLiveBuilder + OBIBuilder + WS)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from services.vpin_live_collector import (
    OBIBuilder,
    VPINLiveBuilder,
    VPINLiveCollectorWS,
    append_vpin_record,
    parse_book_ticker_event,
    parse_trade_event,
)


# ---------------------------------------------------------------------------
# VPINLiveBuilder
# ---------------------------------------------------------------------------


def test_vpin_live_builder_returns_zero_with_no_trades():
    """Buffer vide → VPIN = 0.0."""
    b = VPINLiveBuilder("BTCUSDT", bucket_size_v=1000.0)
    assert b.current_vpin() == 0.0
    assert b.n_trades() == 0


def test_vpin_live_builder_skips_invalid_trades():
    """Trades avec price≤0, qty≤0, types invalides → ignorés."""
    b = VPINLiveBuilder("BTCUSDT", bucket_size_v=100.0)
    b.add_trade(0, 1, 1)         # price=0 → skip
    b.add_trade(50000, 0, 1)     # qty=0 → skip
    b.add_trade(-1, 1, 1)        # price<0 → skip
    b.add_trade("bad", 1, 1)     # type → skip
    assert b.n_trades() == 0


def test_vpin_live_builder_computes_vpin_after_n_trades():
    """100 trades synthétiques → VPIN ∈ [0, 1] et finite."""
    b = VPINLiveBuilder("BTCUSDT", bucket_size_v=10.0, window=5)
    # Prix qui monte régulièrement = direction haussière → VPIN > 0
    for i in range(100):
        price = 50000.0 + i * 5  # monte de $5 chaque trade
        b.add_trade(price, 1.0, 1700000000000 + i * 100)
    vpin = b.current_vpin()
    assert 0.0 <= vpin <= 1.0
    assert b.n_trades() == 100
    assert b.n_buckets() > 0  # buckets formés (volume_total=100, bucket=10)


def test_vpin_live_builder_balanced_flow_low_vpin():
    """Returns flat (price constant) → BVC p_buy ≈ 0.5 → VPIN proche 0."""
    b = VPINLiveBuilder("BTCUSDT", bucket_size_v=10.0, window=5)
    for i in range(100):
        b.add_trade(50000.0, 1.0, 1700000000000 + i * 100)
    vpin = b.current_vpin()
    # prix constant → VPIN = 0 (pas de toxicité)
    assert vpin == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# OBIBuilder
# ---------------------------------------------------------------------------


def test_obi_builder_balanced_returns_one():
    """bid=10, ask=10 → OBI = 1.0 (parfaitement balanced)."""
    o = OBIBuilder("BTCUSDT")
    o.update(10.0, 10.0)
    assert o.current_obi() == 1.0


def test_obi_builder_extreme_imbalance_returns_zero():
    """bid=100, ask=0 → OBI = 0.0 (un côté domine 100%)."""
    o = OBIBuilder("BTCUSDT")
    o.update(100.0, 0.0)
    assert o.current_obi() == 0.0


def test_obi_builder_handles_zero_total():
    """bid=0, ask=0 (book vide / pas reçu) → 0.5 default safe."""
    o = OBIBuilder("BTCUSDT")
    assert o.current_obi() == 0.5  # state initial
    o.update(0.0, 0.0)
    assert o.current_obi() == 0.5


def test_obi_builder_partial_imbalance():
    """bid=3, ask=1 → OBI = 1 - |3-1|/4 = 1 - 0.5 = 0.5."""
    o = OBIBuilder("BTCUSDT")
    o.update(3.0, 1.0)
    assert o.current_obi() == pytest.approx(0.5, abs=1e-6)


def test_obi_builder_clips_to_unit_interval():
    """Valeurs négatives clamped → OBI ∈ [0, 1]."""
    o = OBIBuilder("BTCUSDT")
    o.update(-5, 10)
    assert 0.0 <= o.current_obi() <= 1.0


# ---------------------------------------------------------------------------
# Parsers (Binance Futures payload)
# ---------------------------------------------------------------------------


def test_trade_payload_parsed_correctly():
    """Payload Binance Futures @trade → record extracted."""
    msg = {
        "e": "trade", "E": 1777391806577, "T": 1777391806577,
        "s": "BTCUSDT", "t": 7611007012, "p": "76017.00", "q": "0.001",
        "X": "MARKET", "m": False,
    }
    rec = parse_trade_event(msg)
    assert rec is not None
    assert rec["symbol"] == "BTCUSDT"
    assert rec["price"] == 76017.0
    assert rec["qty"] == 0.001
    assert rec["ts_ms"] == 1777391806577


def test_trade_payload_string_input_works():
    """Parser accepte string JSON aussi."""
    msg_str = '{"e":"trade","T":1700,"s":"ETHUSDT","p":"3000","q":"0.5"}'
    rec = parse_trade_event(msg_str)
    assert rec is not None
    assert rec["symbol"] == "ETHUSDT"


def test_trade_payload_wrong_type_returns_none():
    """e≠'trade' → None."""
    assert parse_trade_event({"e": "aggTrade", "s": "X"}) is None
    assert parse_trade_event({"foo": "bar"}) is None
    assert parse_trade_event("not json") is None
    assert parse_trade_event(None) is None


def test_book_ticker_payload_parsed_correctly():
    """Payload Binance Futures @bookTicker → record extracted."""
    msg = {
        "e": "bookTicker", "u": 10428547052692, "s": "BTCUSDT",
        "b": "76036.80", "B": "10.849", "a": "76036.90", "A": "3.268",
        "T": 1777391838331, "E": 1777391838331,
    }
    rec = parse_book_ticker_event(msg)
    assert rec is not None
    assert rec["symbol"] == "BTCUSDT"
    assert rec["bid_qty"] == 10.849
    assert rec["ask_qty"] == 3.268


def test_book_ticker_legacy_format_no_e_field():
    """Format legacy sans 'e' field accepté."""
    msg = {"u": 1, "s": "BTCUSDT", "b": "1", "B": "10", "a": "1", "A": "5"}
    rec = parse_book_ticker_event(msg)
    assert rec is not None
    assert rec["bid_qty"] == 10.0


def test_book_ticker_missing_qty_returns_none():
    """Manque B ou A → None."""
    assert parse_book_ticker_event({"e": "bookTicker", "s": "X", "b": "1"}) is None


# ---------------------------------------------------------------------------
# append_vpin_record
# ---------------------------------------------------------------------------


def test_append_vpin_record_writes_jsonl(tmp_path):
    path = tmp_path / "sub" / "vpin_live.jsonl"
    rc = append_vpin_record(path, {"ts_ms": 1, "symbol": "BTCUSDT",
                                    "vpin": 0.5, "obi": 0.7})
    assert rc == 1
    assert path.exists()
    rec = json.loads(path.read_text().strip())
    assert rec["vpin"] == 0.5


def test_append_vpin_record_appends_multiple(tmp_path):
    path = tmp_path / "vpin.jsonl"
    append_vpin_record(path, {"vpin": 0.1})
    append_vpin_record(path, {"vpin": 0.2})
    append_vpin_record(path, {"vpin": 0.3})
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# VPINLiveCollectorWS lifecycle (mocked WS)
# ---------------------------------------------------------------------------


def test_collector_constructor_initializes_per_symbol_state():
    coll = VPINLiveCollectorWS(
        symbols=["BTCUSDT", "ETHUSDT"],
        bucket_sizes={"BTCUSDT": 1000.0, "ETHUSDT": 500.0},
        store_path=Path("/tmp/test_vpin.jsonl"),
    )
    assert "BTCUSDT" in coll._states
    assert "ETHUSDT" in coll._states
    assert coll._states["BTCUSDT"].vpin_builder.bucket_size_v == 1000.0
    assert coll._states["ETHUSDT"].vpin_builder.bucket_size_v == 500.0


def test_collector_on_trade_message_feeds_builder(tmp_path):
    coll = VPINLiveCollectorWS(
        symbols=["BTCUSDT"],
        bucket_sizes={"BTCUSDT": 100.0},
        store_path=tmp_path / "vpin.jsonl",
    )
    # Simule 10 trades sur le callback _on_trade_message
    for i in range(10):
        msg = json.dumps({
            "e": "trade", "T": 1700 + i, "s": "BTCUSDT",
            "p": str(50000 + i), "q": "1.0",
        })
        coll._on_trade_message("BTCUSDT", msg)
    assert coll._states["BTCUSDT"].events_received == 10
    assert coll._states["BTCUSDT"].vpin_builder.n_trades() == 10


def test_collector_on_book_message_updates_obi(tmp_path):
    coll = VPINLiveCollectorWS(
        symbols=["BTCUSDT"], bucket_sizes={"BTCUSDT": 100.0},
        store_path=tmp_path / "vpin.jsonl",
    )
    msg = json.dumps({
        "e": "bookTicker", "u": 1, "s": "BTCUSDT",
        "b": "1", "B": "5", "a": "1", "A": "5",
    })
    coll._on_book_message("BTCUSDT", msg)
    assert coll._states["BTCUSDT"].obi_builder.current_obi() == 1.0


def test_collector_flush_writes_jsonl_when_enough_trades(tmp_path):
    """Flush écrit dans jsonl si n_trades >= 5."""
    coll = VPINLiveCollectorWS(
        symbols=["BTCUSDT"], bucket_sizes={"BTCUSDT": 5.0},
        store_path=tmp_path / "vpin.jsonl",
    )
    # Inject 10 trades + 1 book update
    for i in range(10):
        coll._on_trade_message("BTCUSDT", json.dumps({
            "e": "trade", "T": 1700 + i, "s": "BTCUSDT",
            "p": str(50000 + i), "q": "1.0",
        }))
    coll._on_book_message("BTCUSDT", json.dumps({
        "e": "bookTicker", "u": 1, "s": "BTCUSDT",
        "b": "1", "B": "10", "a": "1", "A": "10",
    }))
    coll._flush_symbol("BTCUSDT")
    rec = json.loads((tmp_path / "vpin.jsonl").read_text().strip().split("\n")[-1])
    assert rec["symbol"] == "BTCUSDT"
    assert 0.0 <= rec["vpin"] <= 1.0
    assert 0.0 <= rec["obi"] <= 1.0
    assert rec["n_trades"] == 10


def test_collector_flush_skips_when_not_enough_trades(tmp_path):
    """< 5 trades → pas de flush."""
    coll = VPINLiveCollectorWS(
        symbols=["BTCUSDT"], bucket_sizes={"BTCUSDT": 5.0},
        store_path=tmp_path / "vpin.jsonl",
    )
    for i in range(3):
        coll._on_trade_message("BTCUSDT", json.dumps({
            "e": "trade", "T": 1700 + i, "s": "BTCUSDT",
            "p": "50000", "q": "1.0",
        }))
    coll._flush_symbol("BTCUSDT")
    assert not (tmp_path / "vpin.jsonl").exists()
