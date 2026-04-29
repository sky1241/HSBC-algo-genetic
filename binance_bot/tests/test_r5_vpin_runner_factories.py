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


def test_vpin_data_fn_returns_none_when_jsonl_absent(monkeypatch, tmp_path):
    """P7-bis : si data/vpin_live.jsonl absent → None (gate skip safe).

    Pré-P7-bis ce test asserait `None` car la fn était placeholder
    inconditionnel. Post-P7-bis (mergé f5f22d7), la fn lit le jsonl
    et l'absence du fichier reste la condition None safe.
    """
    from routines import intraday_runner
    fake_root = tmp_path / "fakeroot"
    (fake_root / "data").mkdir(parents=True)
    monkeypatch.setattr(intraday_runner, "ROOT", fake_root)
    fn = _make_vpin_data_fn("BTCUSDT")
    assert fn() is None  # file doesn't exist


def test_vpin_data_fn_returns_tuple_when_jsonl_fresh(monkeypatch, tmp_path):
    """P7-bis : avec record frais dans jsonl → (vpin, obi)."""
    from routines import intraday_runner
    import json
    import time
    fake_root = tmp_path / "fakeroot"
    (fake_root / "data").mkdir(parents=True)
    jsonl = fake_root / "data" / "vpin_live.jsonl"
    now_ms = int(time.time() * 1000)
    jsonl.write_text(json.dumps({
        "ts_ms": now_ms,
        "symbol": "BTCUSDT",
        "vpin": 0.42,
        "obi": 0.78,
        "n_buckets": 5,
        "n_trades": 120,
    }) + "\n")
    monkeypatch.setattr(intraday_runner, "ROOT", fake_root)
    fn = _make_vpin_data_fn("BTCUSDT")
    result = fn()
    assert result is not None
    vpin, obi = result
    assert vpin == pytest.approx(0.42)
    assert obi == pytest.approx(0.78)


def test_vpin_data_fn_returns_none_when_jsonl_stale(monkeypatch, tmp_path):
    """P7-bis : record > 10min stale → None (daemon probablement down)."""
    from routines import intraday_runner
    import json
    import time
    fake_root = tmp_path / "fakeroot"
    (fake_root / "data").mkdir(parents=True)
    jsonl = fake_root / "data" / "vpin_live.jsonl"
    stale_ms = int(time.time() * 1000) - (15 * 60 * 1000)  # 15min ago
    jsonl.write_text(json.dumps({
        "ts_ms": stale_ms,
        "symbol": "BTCUSDT",
        "vpin": 0.42,
        "obi": 0.78,
        "n_buckets": 5,
        "n_trades": 120,
    }) + "\n")
    monkeypatch.setattr(intraday_runner, "ROOT", fake_root)
    fn = _make_vpin_data_fn("BTCUSDT")
    assert fn() is None  # stale
