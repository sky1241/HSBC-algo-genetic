"""Tests pour RiskManager — focus BUG-005 (sizing avec levier)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.risk_manager import RiskManager


def test_position_size_default_no_leverage():
    rm = RiskManager(initial_capital=1000, position_size_pct=0.01, max_leverage=10)
    qty = rm.calculate_position_size(current_equity_usdt=1000, price=50000)
    # 1000 * 0.01 * 1 / 50000 = 0.0002 → round 3 = 0.000
    assert qty == 0.0


def test_position_size_with_leverage():
    rm = RiskManager(initial_capital=1000, position_size_pct=0.01, max_leverage=10)
    qty = rm.calculate_position_size(current_equity_usdt=1000, price=50000, leverage=10.0)
    # 1000 * 0.01 * 10 / 50000 = 0.002
    assert qty == pytest.approx(0.002)


def test_position_size_leverage_clamped_to_max():
    """Si leverage demandé > max_leverage, on clamp."""
    rm = RiskManager(initial_capital=1000, position_size_pct=0.01, max_leverage=5)
    qty = rm.calculate_position_size(current_equity_usdt=1000, price=50000, leverage=20.0)
    # leverage clampé à 5: 1000 * 0.01 * 5 / 50000 = 0.001
    assert qty == pytest.approx(0.001)


def test_position_size_leverage_floored_to_one():
    """Levier < 1 doit être ramené à 1."""
    rm = RiskManager(initial_capital=1000, position_size_pct=0.01, max_leverage=10)
    qty = rm.calculate_position_size(current_equity_usdt=1000, price=50000, leverage=0.5)
    # leverage clampé à 1: 1000 * 0.01 * 1 / 50000 = 0.0002 → round 3 = 0.000
    assert qty == 0.0


def test_global_stop_triggers_below_threshold():
    rm = RiskManager(initial_capital=1000, stop_global_pct=0.50)
    assert rm.check_global_stop(400) is True
    assert rm.check_global_stop(500) is True  # pile au seuil
    assert rm.check_global_stop(501) is False
