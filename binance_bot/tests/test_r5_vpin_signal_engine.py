"""R5 — Tests d'intégration VPIN gate dans signal_engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from services.signal_engine import SignalEngine
from src.vpin_gate import VPINGateConfig, VPINState  # type: ignore


def _make_df(close=50000.0, atr=100.0, signal_long=False, signal_short=False):
    return pd.DataFrame({
        "close": [close],
        "ATR": [atr],
        "signal_long": [signal_long],
        "signal_short": [signal_short],
    })


# ---------------------------------------------------------------------------
# Mode log_only — never blocks (default safe)
# ---------------------------------------------------------------------------


def test_signal_engine_vpin_log_only_does_not_block(tmp_path):
    """Mode log_only : même VPIN=0.95, signal d'entrée passe."""
    cfg = VPINGateConfig(mode="log_only")
    log_path = tmp_path / "vpin_events.jsonl"

    eng = SignalEngine(
        vpin_data_fn=lambda: (0.95, 0.10),
        vpin_gate_config=cfg,
        vpin_event_log_path=log_path,
    )
    df = _make_df(signal_long=True)
    signals = eng.detect_signals(
        df, params={"atr_mult": 10.0, "tp_mult": 20.0},
        current_price=50000.0,
    )
    # Mode log_only → action allow → signal d'entrée généré
    assert any(s["action"] == "open_long" for s in signals)
    # Event loggé
    assert log_path.exists()
    rec = json.loads(log_path.read_text().strip())
    assert rec["vpin"] == pytest.approx(0.95)
    assert rec["action"] == "allow"
    assert rec["mode"] == "log_only"


# ---------------------------------------------------------------------------
# Mode gate — block_new_entries
# ---------------------------------------------------------------------------


def test_signal_engine_vpin_gate_blocks_long_entry(tmp_path):
    """Mode gate + VPIN=0.75 → signal long bloqué."""
    cfg = VPINGateConfig(mode="gate", block_threshold=0.70)

    eng = SignalEngine(
        vpin_data_fn=lambda: (0.75, 0.5),
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
    )
    df = _make_df(signal_long=True)
    signals = eng.detect_signals(
        df, params={"atr_mult": 10.0, "tp_mult": 20.0},
        current_price=50000.0,
    )
    assert not any(s["action"] == "open_long" for s in signals)


def test_signal_engine_vpin_gate_blocks_short_entry(tmp_path):
    """Symétrique pour short."""
    cfg = VPINGateConfig(mode="gate")
    eng = SignalEngine(
        vpin_data_fn=lambda: (0.80, 0.5),
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
    )
    df = _make_df(signal_short=True)
    signals = eng.detect_signals(df, {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    assert not any(s["action"] == "open_short" for s in signals)


# ---------------------------------------------------------------------------
# Mode gate — kill_and_block
# ---------------------------------------------------------------------------


def test_signal_engine_vpin_kill_flats_open_positions(tmp_path):
    """Mode gate + VPIN=0.95 + OBI=0.10 → kill, toutes positions flattées."""
    cfg = VPINGateConfig(mode="gate", kill_threshold=0.85, kill_obi_threshold=0.30)
    eng = SignalEngine(
        vpin_data_fn=lambda: (0.95, 0.10),
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
    )
    eng.load_state(
        positions_long=[{"id": 1, "entry": 50000.0, "stop": 48000.0, "tp": 52000.0,
                         "size": 0.01}],
        positions_short=[{"id": 2, "entry": 50000.0, "stop": 52000.0, "tp": 48000.0,
                          "size": 0.01}],
        daily_loss=0.0,
    )
    df = _make_df()
    signals = eng.detect_signals(df, {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    # Au moins 2 signaux close (long + short flattés)
    close_actions = [s for s in signals if "close" in s.get("action", "")]
    assert len(close_actions) >= 2
    # Reasons taggées vpin_cascade
    assert any("vpin_cascade" in s.get("reason", "") for s in close_actions)


def test_signal_engine_vpin_kill_sends_warn_notification(tmp_path):
    """Kill action → notifier.warn appelé."""
    cfg = VPINGateConfig(mode="gate")
    notifier = MagicMock()
    eng = SignalEngine(
        vpin_data_fn=lambda: (0.95, 0.10),
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
        notifier=notifier,
    )
    df = _make_df()
    eng.detect_signals(df, {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    notifier.warn.assert_called_once()
    msg = notifier.warn.call_args.args[0]
    assert "VPIN" in msg and "kill_and_block" in msg


# ---------------------------------------------------------------------------
# Robustesse data
# ---------------------------------------------------------------------------


def test_signal_engine_vpin_data_fn_returns_none_no_block(tmp_path):
    """vpin_data_fn() retourne None → allow (data unavailable)."""
    cfg = VPINGateConfig(mode="gate")
    eng = SignalEngine(
        vpin_data_fn=lambda: None,
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
    )
    df = _make_df(signal_long=True)
    signals = eng.detect_signals(df, {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    assert any(s["action"] == "open_long" for s in signals)


def test_signal_engine_vpin_data_fn_raises_safe_fallback(tmp_path):
    """vpin_data_fn raise → fail-open (allow)."""
    def boom():
        raise RuntimeError("vpin computation crashed")
    cfg = VPINGateConfig(mode="gate")
    eng = SignalEngine(
        vpin_data_fn=boom,
        vpin_gate_config=cfg,
    )
    df = _make_df(signal_long=True)
    signals = eng.detect_signals(df, {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    assert any(s["action"] == "open_long" for s in signals)


def test_signal_engine_vpin_disabled_when_no_data_fn():
    """Pas de vpin_data_fn → totalement skippé (pas de log)."""
    eng = SignalEngine()  # tout default
    action, reason = eng._evaluate_vpin_gate()
    assert action == "allow"
    assert reason == "vpin_disabled"


def test_signal_engine_vpin_data_out_of_range_safe(tmp_path):
    """VPIN ou OBI hors [0,1] → safe allow."""
    cfg = VPINGateConfig(mode="gate")
    eng = SignalEngine(
        vpin_data_fn=lambda: (1.5, 0.5),
        vpin_gate_config=cfg,
    )
    action, reason = eng._evaluate_vpin_gate()
    assert action == "allow"
    assert "out_of_range" in reason


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_signal_engine_vpin_state_persists_across_cycles(tmp_path):
    """Le state currently_blocked doit survivre via vpin_state_dict roundtrip."""
    cfg = VPINGateConfig(mode="gate", block_duration_minutes=15)
    # Cycle 1 : trigger
    eng1 = SignalEngine(
        vpin_data_fn=lambda: (0.80, 0.5),
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
    )
    eng1.detect_signals(_make_df(), {"atr_mult": 10.0, "tp_mult": 20.0}, 50000.0)
    state_dict = eng1.get_vpin_state_dict()
    assert state_dict is not None
    assert state_dict["currently_blocked"] is True
    # Cycle 2 : on instancie un nouveau SignalEngine avec le state précédent
    eng2 = SignalEngine(
        vpin_data_fn=lambda: (0.55, 0.5),  # > reset, reste blocked
        vpin_gate_config=cfg,
        vpin_event_log_path=tmp_path / "v.jsonl",
        vpin_state_dict=state_dict,
    )
    action, reason = eng2._evaluate_vpin_gate()
    assert action == "block_new_entries"
    assert reason == "state_maintained"
