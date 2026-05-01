"""P-MTF-3 — Tests d'intégration trend_gate_fn dans SignalEngine.

On teste uniquement le helper `_trend_gate_blocks()` car le pipeline complet
detect_signals() est testé séparément via les tests d'intégration multi-TF
(P-MTF-10) qui montent un vrai DataFrame Ichimoku 15m.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from binance_bot.services.signal_engine import SignalEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine_no_gate():
    """SignalEngine sans trend_gate_fn (mode legacy mono-H2)."""
    return SignalEngine(max_positions=3)


def make_engine(direction: str | None, raise_exc: bool = False):
    """SignalEngine avec un trend_gate_fn paramétrable."""
    if raise_exc:
        gate = MagicMock(side_effect=RuntimeError("h2 unavailable"))
    elif direction is None:
        gate = MagicMock(return_value=None)
    else:
        gate = MagicMock(return_value={"direction": direction, "reason": "test"})
    return SignalEngine(max_positions=3, trend_gate_fn=gate)


# ---------------------------------------------------------------------------
# Tests _trend_gate_blocks()
# ---------------------------------------------------------------------------


def test_no_gate_does_not_block(engine_no_gate):
    """trend_gate_fn=None → pas de blocage (backward compat)."""
    blocks, reason = engine_no_gate._trend_gate_blocks("long")
    assert blocks is False
    assert reason == ""
    blocks, reason = engine_no_gate._trend_gate_blocks("short")
    assert blocks is False


def test_h2_long_allows_signal_long():
    eng = make_engine("long")
    blocks, reason = eng._trend_gate_blocks("long")
    assert blocks is False
    assert reason == ""


def test_h2_long_blocks_signal_short():
    eng = make_engine("long")
    blocks, reason = eng._trend_gate_blocks("short")
    assert blocks is True
    assert "h2_direction=long" in reason


def test_h2_short_allows_signal_short():
    eng = make_engine("short")
    blocks, reason = eng._trend_gate_blocks("short")
    assert blocks is False


def test_h2_short_blocks_signal_long():
    eng = make_engine("short")
    blocks, reason = eng._trend_gate_blocks("long")
    assert blocks is True
    assert "h2_direction=short" in reason


def test_h2_flat_blocks_both_sides():
    eng = make_engine("flat")
    blocks_long, reason_long = eng._trend_gate_blocks("long")
    blocks_short, reason_short = eng._trend_gate_blocks("short")
    assert blocks_long is True
    assert blocks_short is True
    assert "h2_flat" in reason_long
    assert "h2_flat" in reason_short


def test_h2_none_treated_as_flat():
    """trend_gate_fn() retourne None → traité comme flat → blocage."""
    eng = make_engine(None)
    blocks, reason = eng._trend_gate_blocks("long")
    assert blocks is True
    assert "h2_flat" in reason


def test_h2_exception_blocks_safe_fail():
    """Exception dans le callback → BLOQUE (safe-fail spécifique trend gate)."""
    eng = make_engine(None, raise_exc=True)
    blocks, reason = eng._trend_gate_blocks("long")
    assert blocks is True
    assert "trend_gate_exception" in reason


def test_h2_unknown_direction_treated_as_flat():
    """Direction non reconnue → bloque (sécurité)."""
    eng = make_engine("sideways")  # invalid value
    blocks_long, _ = eng._trend_gate_blocks("long")
    blocks_short, _ = eng._trend_gate_blocks("short")
    # "sideways" != "long" et != "short" → bloque les 2
    assert blocks_long is True
    assert blocks_short is True


def test_h2_gate_called_per_invocation():
    """Le callback est appelé à chaque _trend_gate_blocks (pas caché)."""
    gate = MagicMock(return_value={"direction": "long"})
    eng = SignalEngine(max_positions=3, trend_gate_fn=gate)
    eng._trend_gate_blocks("long")
    eng._trend_gate_blocks("short")
    eng._trend_gate_blocks("long")
    assert gate.call_count == 3
