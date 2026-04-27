"""Tests pour kill_switch — B7."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.kill_switch import is_killed, read_kill_reason, trigger_kill


@pytest.fixture
def flag(tmp_path):
    return tmp_path / "data" / ".killed"


def test_is_killed_false_when_no_flag(flag):
    assert is_killed(flag) is False


def test_is_killed_true_when_flag_exists(flag):
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("test")
    assert is_killed(flag) is True


def test_read_kill_reason_returns_empty_when_no_flag(flag):
    assert read_kill_reason(flag) == ""


def test_trigger_kill_creates_flag_with_reason(flag):
    actions = trigger_kill(flag, reason="test_global_stop")
    assert flag.exists()
    body = flag.read_text()
    assert "KILLED at" in body
    assert "test_global_stop" in body
    assert f"rm {flag}" in body
    assert actions == []  # rien à fermer si pas de trade_mgr


def test_trigger_kill_skips_close_in_simulation(flag):
    state_mgr = MagicMock()
    state_mgr.get.side_effect = lambda k, default=None: {
        "positions_long": [{"id": "long_1", "size": 0.01}],
        "positions_short": [],
    }.get(k, default)

    trade_mgr = MagicMock()
    trade_mgr.mode = "simulation"

    actions = trigger_kill(flag, "test", state_mgr=state_mgr, trade_mgr=trade_mgr,
                           current_price=78000, capital_usdt=1000)

    assert flag.exists()
    trade_mgr.execute_signal.assert_not_called()
    trade_mgr.exchange.cancel_all_orders.assert_not_called()
    assert actions == []


def test_trigger_kill_closes_positions_in_live(flag):
    state_mgr = MagicMock()
    state_mgr.get.side_effect = lambda k, default=None: {
        "positions_long": [{"id": "long_1", "size": 0.01}],
        "positions_short": [{"id": "short_1", "size": 0.02}],
    }.get(k, default)

    trade_mgr = MagicMock()
    trade_mgr.mode = "live"
    trade_mgr.symbol = "BTC/USDT"
    trade_mgr.execute_signal.return_value = "mkt_order_id"

    actions = trigger_kill(flag, "global_stop", state_mgr=state_mgr, trade_mgr=trade_mgr,
                           current_price=78000, capital_usdt=2500)

    # 1 close_long + 1 close_short + 1 cancel_all_orders
    assert trade_mgr.execute_signal.call_count == 2
    trade_mgr.exchange.cancel_all_orders.assert_called_once_with("BTC/USDT")

    actions_str = " | ".join(actions)
    assert "close_long" in actions_str
    assert "close_short" in actions_str
    assert "cancel_all_orders" in actions_str
    assert flag.exists()


def test_trigger_kill_continues_even_if_close_fails(flag):
    state_mgr = MagicMock()
    state_mgr.get.side_effect = lambda k, default=None: {
        "positions_long": [{"id": "long_1", "size": 0.01}],
        "positions_short": [],
    }.get(k, default)

    trade_mgr = MagicMock()
    trade_mgr.mode = "live"
    trade_mgr.symbol = "BTC/USDT"
    trade_mgr.execute_signal.side_effect = Exception("network error")

    actions = trigger_kill(flag, "global_stop", state_mgr=state_mgr, trade_mgr=trade_mgr,
                           current_price=78000, capital_usdt=1000)

    # Le flag est créé même si la fermeture échoue
    assert flag.exists()
    assert any("FAILED" in a for a in actions)
