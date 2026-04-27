"""Tests pour reconcile_positions — BUG-006."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.reconciler import reconcile_positions


class FakeStateManager:
    """Stub léger pour StateManager."""

    def __init__(self, positions_long=None, positions_short=None):
        self.state = {
            'positions_long': list(positions_long or []),
            'positions_short': list(positions_short or []),
        }
        self.saved = 0

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value

    def save(self):
        self.saved += 1


def _exchange_with_positions(positions):
    ex = MagicMock()
    ex.fetch_positions.return_value = positions
    return ex


def test_reconcile_no_divergence_does_not_save():
    """State et Binance d'accord (rien d'ouvert) → aucun save, (0,0)."""
    sm = FakeStateManager()
    ex = _exchange_with_positions([])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert (added, removed) == (0, 0)
    assert sm.saved == 0


def test_reconcile_adds_missing_long_position():
    """Binance a une position long, state n'a rien → ajoute au state."""
    sm = FakeStateManager()
    ex = _exchange_with_positions([
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.05},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert added == 1
    assert removed == 0
    assert len(sm.get('positions_long')) == 1
    assert sm.get('positions_long')[0]['size'] == 0.05
    assert sm.get('positions_long')[0]['reconciled'] is True
    assert sm.saved == 1


def test_reconcile_adds_missing_short_position():
    sm = FakeStateManager()
    ex = _exchange_with_positions([
        {'symbol': 'BTC/USDT', 'side': 'short', 'contracts': 0.03},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert added == 1
    assert sm.get('positions_short')[0]['size'] == 0.03


def test_reconcile_removes_phantom_long_position():
    """State liste une position long, Binance n'en a pas → la retire du state."""
    sm = FakeStateManager(
        positions_long=[{'id': 'long_1', 'size': 0.01, 'entry': 50000, 'stop': 49000, 'tp': 60000}]
    )
    ex = _exchange_with_positions([])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert added == 0
    assert removed == 1
    assert sm.get('positions_long') == []
    assert sm.saved == 1


def test_reconcile_removes_phantom_short_position():
    sm = FakeStateManager(
        positions_short=[{'id': 'short_1', 'size': 0.02, 'entry': 50000, 'stop': 51000, 'tp': 40000}]
    )
    ex = _exchange_with_positions([])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert removed == 1
    assert sm.get('positions_short') == []


def test_reconcile_filters_by_symbol():
    """Une position ETH/USDT ne doit pas affecter BTC/USDT."""
    sm = FakeStateManager()
    ex = _exchange_with_positions([
        {'symbol': 'ETH/USDT', 'side': 'long', 'contracts': 1.0},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert (added, removed) == (0, 0)


def test_reconcile_ignores_zero_contracts_positions():
    """Binance retourne parfois des positions avec contracts=0 — on les ignore."""
    sm = FakeStateManager()
    ex = _exchange_with_positions([
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert (added, removed) == (0, 0)


def test_reconcile_handles_fetch_positions_exception():
    """Si fetch_positions lève, on retourne (0,0) sans modifier le state."""
    sm = FakeStateManager(
        positions_long=[{'id': 'long_1', 'size': 0.01, 'entry': 50000, 'stop': 49000, 'tp': 60000}]
    )
    ex = MagicMock()
    ex.fetch_positions.side_effect = Exception("network down")
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert (added, removed) == (0, 0)
    assert len(sm.get('positions_long')) == 1  # state inchangé
    assert sm.saved == 0


def test_reconcile_state_already_matches_binance():
    """State a une long position et Binance aussi — pas de modification."""
    sm = FakeStateManager(
        positions_long=[{'id': 'long_1', 'size': 0.05, 'entry': 50000, 'stop': 49000, 'tp': 60000}]
    )
    ex = _exchange_with_positions([
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.05},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    # state non vide ET binance non vide → no-op
    assert (added, removed) == (0, 0)
    assert sm.saved == 0


def test_reconcile_both_sides_simultaneously():
    """State a un short fantôme, Binance a un long manquant — gère les deux."""
    sm = FakeStateManager(
        positions_short=[{'id': 'short_1', 'size': 0.02, 'entry': 50000, 'stop': 51000, 'tp': 40000}]
    )
    ex = _exchange_with_positions([
        {'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.04},
    ])
    added, removed = reconcile_positions(sm, ex, "BTC/USDT")
    assert added == 1
    assert removed == 1
    assert len(sm.get('positions_long')) == 1
    assert sm.get('positions_short') == []
