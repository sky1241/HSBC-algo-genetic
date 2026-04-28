"""R5 — Tests des factories intraday_runner pour le VPIN gate."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from routines.intraday_runner import (
    _make_vpin_data_fn,
    _make_vpin_gate_config,
)


def test_vpin_gate_config_from_settings():
    settings = {
        "vpin_mode": "gate",
        "vpin_block_threshold": 0.65,
        "vpin_kill_threshold": 0.90,
        "vpin_kill_obi_threshold": 0.25,
        "vpin_reset_threshold": 0.45,
        "vpin_block_duration_minutes": 20,
    }
    cfg = _make_vpin_gate_config(settings)
    assert cfg.mode == "gate"
    assert cfg.block_threshold == pytest.approx(0.65)
    assert cfg.kill_threshold == pytest.approx(0.90)
    assert cfg.kill_obi_threshold == pytest.approx(0.25)
    assert cfg.reset_threshold == pytest.approx(0.45)
    assert cfg.block_duration_minutes == 20


def test_vpin_gate_config_defaults_when_missing():
    """Settings vide → config avec valeurs par défaut spec."""
    cfg = _make_vpin_gate_config({})
    assert cfg.mode == "log_only"  # default safe
    assert cfg.block_threshold == 0.70
    assert cfg.kill_threshold == 0.85
    assert cfg.kill_obi_threshold == 0.30
    assert cfg.reset_threshold == 0.50
    assert cfg.block_duration_minutes == 15


def test_vpin_data_fn_placeholder_returns_none():
    """Tant que le collecteur live n'existe pas, retourne None (gate skip)."""
    fn = _make_vpin_data_fn("BTCUSDT")
    assert fn() is None
