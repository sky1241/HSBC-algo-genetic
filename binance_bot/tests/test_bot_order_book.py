"""Tests pour order_book (OBI + spread + depth)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from services.order_book import BookSnapshot, is_safe_to_trade, parse_order_book


def _book(bids, asks):
    return {"bids": bids, "asks": asks}


def test_parse_empty_returns_none():
    assert parse_order_book({"bids": [], "asks": []}) is None
    assert parse_order_book({"bids": [[100, 1]], "asks": []}) is None


def test_parse_basic():
    book = _book(
        bids=[[78000, 0.5], [77995, 1.0], [77990, 2.0]],
        asks=[[78010, 0.3], [78015, 0.8], [78020, 1.5]],
    )
    snap = parse_order_book(book)
    assert snap is not None
    assert snap.bid_top_price == 78000
    assert snap.ask_top_price == 78010
    assert snap.mid == pytest.approx(78005, abs=0.1)
    # spread = 10/78005 * 10000 ≈ 1.28 bps
    assert snap.spread_bps == pytest.approx(1.28, abs=0.05)


def test_obi_balanced_when_equal_sizes():
    snap = parse_order_book(_book(
        bids=[[100, 1.0]], asks=[[101, 1.0]],
    ))
    assert snap.obi == pytest.approx(0.0)


def test_obi_positive_when_bid_heavy():
    snap = parse_order_book(_book(
        bids=[[100, 5.0]], asks=[[101, 1.0]],
    ))
    assert snap.obi > 0
    assert snap.obi == pytest.approx(0.667, abs=0.01)


def test_obi_negative_when_ask_heavy():
    snap = parse_order_book(_book(
        bids=[[100, 1.0]], asks=[[101, 5.0]],
    ))
    assert snap.obi < 0


def test_depth_50bps_aggregates_correctly():
    """À mid=100, 50 bps = 0.5. Bids ≥ 99.5, asks ≤ 100.5."""
    book = _book(
        bids=[[100, 1.0], [99.6, 2.0], [99.0, 5.0]],   # 3eme hors 50bps
        asks=[[100.2, 0.5], [100.4, 0.5], [100.7, 3.0]],  # 3eme hors
    )
    snap = parse_order_book(book, depth_bps_threshold=50.0)
    # bid_depth = 1.0 + 2.0 = 3.0 (99.0 exclu car < 99.5)
    assert snap.bid_depth_50bps == pytest.approx(3.0)
    # ask_depth = 0.5 + 0.5 = 1.0 (100.7 exclu car > 100.5)
    assert snap.ask_depth_50bps == pytest.approx(1.0)


def test_safe_to_trade_ok_with_normal_book():
    snap = BookSnapshot(
        bid_top_price=78000, ask_top_price=78005, bid_top_size=1, ask_top_size=1,
        mid=78002.5, spread_bps=0.6, obi=0.0,
        bid_depth_50bps=10.0, ask_depth_50bps=10.0,
    )
    ok, reason = is_safe_to_trade(snap, max_spread_bps=10, min_depth_usdt_per_side=5000)
    assert ok is True


def test_safe_to_trade_blocks_wide_spread():
    snap = BookSnapshot(
        bid_top_price=78000, ask_top_price=78500, bid_top_size=1, ask_top_size=1,
        mid=78250, spread_bps=64.0, obi=0.0,
        bid_depth_50bps=10.0, ask_depth_50bps=10.0,
    )
    ok, reason = is_safe_to_trade(snap, max_spread_bps=10)
    assert ok is False
    assert "spread" in reason


def test_safe_to_trade_blocks_thin_depth():
    snap = BookSnapshot(
        bid_top_price=78000, ask_top_price=78005, bid_top_size=1, ask_top_size=1,
        mid=78002.5, spread_bps=0.6, obi=0.0,
        bid_depth_50bps=0.001, ask_depth_50bps=0.001,
    )
    ok, reason = is_safe_to_trade(snap, min_depth_usdt_per_side=5000)
    assert ok is False
    assert "depth" in reason


def test_safe_to_trade_none_book():
    ok, reason = is_safe_to_trade(None)
    assert ok is False
