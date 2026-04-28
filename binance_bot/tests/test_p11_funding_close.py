"""P11 — Tests funding_close (settlement-aware position closing)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.funding_close import (
    DEFAULT_ADVANCE_MINUTES,
    DEFAULT_THRESHOLD_LONG_BPS,
    DEFAULT_THRESHOLD_SHORT_BPS,
    SETTLEMENT_HOURS_UTC,
    evaluate_close_signals,
    is_settlement_window,
    minutes_to_settlement,
    next_settlement,
    should_close_position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(year, month, day, hour, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Settlement timing
# ---------------------------------------------------------------------------


def test_settlement_hours_match_spec():
    assert SETTLEMENT_HOURS_UTC == (0, 8, 16)


def test_next_settlement_picks_first_of_today():
    """à 03:00 UTC → next = 08:00 UTC le même jour."""
    now = _utc(2026, 5, 1, 3, 0)
    assert next_settlement(now) == _utc(2026, 5, 1, 8, 0)


def test_next_settlement_wraps_to_next_day():
    """à 18:30 UTC → next = 00:00 UTC du lendemain."""
    now = _utc(2026, 5, 1, 18, 30)
    assert next_settlement(now) == _utc(2026, 5, 2, 0, 0)


def test_next_settlement_at_exact_boundary():
    """à 08:00:00 UTC pile → next = 08:00 même (>=)."""
    now = _utc(2026, 5, 1, 8, 0, 0)
    assert next_settlement(now) == _utc(2026, 5, 1, 8, 0, 0)


def test_minutes_to_settlement_basic():
    assert minutes_to_settlement(_utc(2026, 5, 1, 7, 55)) == 5
    assert minutes_to_settlement(_utc(2026, 5, 1, 7, 30)) == 30
    assert minutes_to_settlement(_utc(2026, 5, 1, 0, 0)) == 0  # exactement settlement


# ---------------------------------------------------------------------------
# is_settlement_window
# ---------------------------------------------------------------------------


def test_is_settlement_window_within_5min():
    """À 07:56 (4 min avant 08:00) → fenêtre active."""
    assert is_settlement_window(_utc(2026, 5, 1, 7, 56), advance_minutes=5) is True
    # À 07:55 (5 min avant) → encore dans la fenêtre [0, 5]
    assert is_settlement_window(_utc(2026, 5, 1, 7, 55), advance_minutes=5) is True


def test_is_settlement_window_outside():
    """À 07:30 (30 min avant) → hors fenêtre."""
    assert is_settlement_window(_utc(2026, 5, 1, 7, 30), advance_minutes=5) is False
    # Pile sur le settlement → mins=0, donc hors fenêtre stricte (0 < mins).
    assert is_settlement_window(_utc(2026, 5, 1, 8, 0), advance_minutes=5) is False


def test_is_settlement_window_advance_minutes_configurable():
    """advance=10 → la fenêtre 7:50→7:59 active."""
    assert is_settlement_window(_utc(2026, 5, 1, 7, 50), advance_minutes=10) is True
    assert is_settlement_window(_utc(2026, 5, 1, 7, 49), advance_minutes=10) is False


def test_is_settlement_window_zero_advance():
    """advance=0 → jamais dans la fenêtre (sécurité)."""
    assert is_settlement_window(_utc(2026, 5, 1, 7, 59), advance_minutes=0) is False


# ---------------------------------------------------------------------------
# should_close_position
# ---------------------------------------------------------------------------


def test_long_high_funding_should_close():
    """LONG + funding > +5 bps → close."""
    close, reason = should_close_position("long", funding_rate_bps=7.5)
    assert close is True
    assert "longs paient" in reason.lower() or "long" in reason.lower()


def test_long_negative_funding_held():
    """LONG + funding < 0 → hold (les shorts nous paient)."""
    close, reason = should_close_position("long", funding_rate_bps=-3.0)
    assert close is False
    assert reason == ""


def test_long_funding_below_threshold_held():
    """LONG + funding ∈ (0, +5] → hold (pas extrême)."""
    close, _ = should_close_position("long", funding_rate_bps=3.0)
    assert close is False
    close, _ = should_close_position("long", funding_rate_bps=5.0)  # exactement seuil
    assert close is False


def test_short_high_funding_should_close():
    """SHORT + funding < -5 bps → close (shorts paient)."""
    close, reason = should_close_position("short", funding_rate_bps=-7.5)
    assert close is True
    assert "short" in reason.lower() and "paient" in reason.lower()


def test_short_positive_funding_held():
    """SHORT + funding > 0 → hold (les longs nous paient)."""
    close, _ = should_close_position("short", funding_rate_bps=3.0)
    assert close is False


def test_threshold_configurable():
    """Threshold custom à 10 bps : funding=7 ne ferme plus le LONG."""
    close, _ = should_close_position("long", 7.0, threshold_long_bps=10.0)
    assert close is False
    # En revanche à 12 bps, ça ferme
    close, _ = should_close_position("long", 12.0, threshold_long_bps=10.0)
    assert close is True


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        should_close_position("flat", 7.0)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        should_close_position("long", 7.0, threshold_long_bps=-1.0)


# ---------------------------------------------------------------------------
# evaluate_close_signals (intégration)
# ---------------------------------------------------------------------------


def test_no_action_outside_settlement_window():
    """Loin du settlement → aucun signal généré, même funding extrême."""
    positions = [{"id": 1, "side": "long", "symbol": "BTCUSDT"}]
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 4, 0),  # 4h avant 08:00
        positions=positions,
        funding_rate_bps=50.0,  # extrême
    )
    assert signals == []


def test_settlement_window_long_high_funding_emits_close():
    """Dans fenêtre + LONG + funding élevé → close émis."""
    positions = [
        {"id": 42, "side": "long", "symbol": "BTCUSDT"},
        {"id": 43, "side": "short", "symbol": "ETHUSDT"},  # short avec funding+, on hold
    ]
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 7, 56),  # 4 min avant 08:00
        positions=positions,
        funding_rate_bps=8.0,
    )
    assert len(signals) == 1
    sig = signals[0]
    assert sig["action"] == "funding_close"
    assert sig["side"] == "long"
    assert sig["pos_id"] == 42
    assert sig["symbol"] == "BTCUSDT"
    assert sig["funding_rate_bps"] == pytest.approx(8.0)
    assert sig["minutes_to_settlement"] == 4


def test_settlement_window_short_negative_funding_emits_close():
    positions = [{"id": 1, "side": "short", "symbol": "SOLUSDT"}]
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 15, 57),  # 3 min avant 16:00
        positions=positions,
        funding_rate_bps=-10.0,
    )
    assert len(signals) == 1
    assert signals[0]["side"] == "short"
    assert signals[0]["funding_rate_bps"] == pytest.approx(-10.0)


def test_settlement_window_held_positions_no_close():
    """LONG + funding négatif → on reste, aucun signal même dans la fenêtre."""
    positions = [{"id": 1, "side": "long", "symbol": "BTCUSDT"}]
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 7, 56),
        positions=positions,
        funding_rate_bps=-5.0,
    )
    assert signals == []


def test_threshold_passed_through_evaluate():
    """Threshold custom propagé jusqu'à should_close_position."""
    positions = [{"id": 1, "side": "long", "symbol": "BTCUSDT"}]
    # avec threshold=10, funding=7 ne déclenche pas
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 7, 56),
        positions=positions,
        funding_rate_bps=7.0,
        threshold_long_bps=10.0,
    )
    assert signals == []


def test_default_constants_match_spec():
    assert DEFAULT_THRESHOLD_LONG_BPS == 5.0
    assert DEFAULT_THRESHOLD_SHORT_BPS == 5.0
    assert DEFAULT_ADVANCE_MINUTES == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_naive_datetime_treated_as_utc():
    """datetime sans tzinfo → traité comme UTC (ne raise pas)."""
    naive = datetime(2026, 5, 1, 7, 56)
    assert is_settlement_window(naive) is True


def test_evaluate_skips_invalid_side_in_position():
    """Position avec side invalide → ignorée (pas de raise)."""
    positions = [
        {"id": 1, "side": "weird", "symbol": "X"},
        {"id": 2, "side": "long", "symbol": "BTCUSDT"},
    ]
    signals = evaluate_close_signals(
        now=_utc(2026, 5, 1, 7, 56),
        positions=positions,
        funding_rate_bps=8.0,
    )
    assert len(signals) == 1
    assert signals[0]["pos_id"] == 2
