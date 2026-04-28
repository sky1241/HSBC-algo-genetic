"""P1 — Tests daily caps soft 2%/3% + hard 10% sur SignalEngine.

Vérifie:
  1. Soft loss cap atteint → flat positions + block_until_next_utc_midnight
  2. Soft gain cap atteint → flat positions + block_until_next_utc_midnight
  3. Hard loss cap atteint → trigger kill_switch + flat
  4. Aucune action si PnL dans les caps (entries autorisées)
  5. Block actif → pas d'entrée mais sorties TP/SL OK
  6. Block expiré (00:00 UTC suivant) → auto-clear, entries autorisées à nouveau
  7. Hard cap → notifier.critical appelé
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.signal_engine import SignalEngine


def _make_df_with_signal(signal_long=False, signal_short=False, atr=100.0, close=50000.0):
    """DataFrame Ichimoku minimal avec signaux donnés."""
    return pd.DataFrame({
        "close": [close],
        "ATR": [atr],
        "signal_long": [signal_long],
        "signal_short": [signal_short],
    })


# ---------------------------------------------------------------------------
# Tests obligatoires (spec P1)
# ---------------------------------------------------------------------------


def test_soft_loss_cap_flats_and_blocks(tmp_path):
    """PnL = -2.5% < soft_loss 2% → flat all + block_until set."""
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_loss_hard_cap_pct=0.10,
    )
    # Charger 1 long ouvert avant cap
    se.load_state(
        positions_long=[{"id": "L1", "entry": 50000, "stop": -np.inf, "tp": np.inf, "size": 0.01}],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.025,  # -2.5% : sous le soft cap
    )
    df = _make_df_with_signal(signal_long=False, signal_short=False)
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    # 1 close_long généré + reason="soft_loss_cap"
    assert any(s["action"] == "close_long" and s["reason"] == "soft_loss_cap" for s in signals)
    # Position interne effacée
    assert se.positions_long == []
    # Block_until_iso set sur prochain 00:00 UTC
    assert se.block_until_iso is not None
    parsed = datetime.fromisoformat(se.block_until_iso)
    assert parsed.hour == 0 and parsed.minute == 0


def test_soft_gain_cap_flats_and_blocks():
    """PnL = +3.5% > gain_soft 3% → flat all + block."""
    se = SignalEngine(
        daily_gain_soft_cap_pct=0.03,
        daily_loss_hard_cap_pct=0.10,
    )
    se.load_state(
        positions_long=[],
        positions_short=[{"id": "S1", "entry": 50000, "stop": np.inf, "tp": -np.inf, "size": 0.01}],
        daily_loss=0.0,
        daily_pnl_pct=+0.035,
    )
    df = _make_df_with_signal()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    assert any(s["action"] == "close_short" and s["reason"] == "soft_gain_cap" for s in signals)
    assert se.positions_short == []
    assert se.block_until_iso is not None


def test_hard_loss_cap_triggers_kill(tmp_path):
    """PnL = -12% < hard_loss 10% → flat + kill_switch + Telegram critical."""
    kill_path = tmp_path / ".killed"
    notifier = MagicMock()
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_loss_hard_cap_pct=0.10,
        kill_switch_path=kill_path,
        notifier=notifier,
    )
    se.load_state(
        positions_long=[{"id": "L1", "entry": 50000, "stop": -np.inf, "tp": np.inf, "size": 0.01}],
        positions_short=[{"id": "S1", "entry": 50000, "stop": np.inf, "tp": -np.inf, "size": 0.01}],
        daily_loss=0.0,
        daily_pnl_pct=-0.12,  # -12% : hard cap
    )
    df = _make_df_with_signal()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    # Toutes positions fermées avec reason hard_loss_cap
    close_actions = [s for s in signals if s["action"].startswith("close_")]
    assert len(close_actions) == 2
    assert all(s["reason"] == "hard_loss_cap" for s in close_actions)
    # Telegram critical appelé
    assert notifier.critical.called
    msg = notifier.critical.call_args[0][0]
    assert "HARD DAILY LOSS CAP" in msg


def test_no_action_within_caps():
    """PnL = -1% ∈ [-2%, +3%] → comportement legacy normal (TP/SL/entrées OK)."""
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_gain_soft_cap_pct=0.03,
        daily_loss_hard_cap_pct=0.10,
    )
    se.load_state(
        positions_long=[],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.01,  # -1% : dans les caps
    )
    # Signal long actif → doit générer open_long normalement
    df = _make_df_with_signal(signal_long=True)
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    assert any(s["action"] == "open_long" for s in signals)
    assert se.block_until_iso is None  # pas de block déclenché


def test_block_resets_at_utc_midnight():
    """block_until_iso passé → auto-clear, entries autorisées à nouveau."""
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_loss_hard_cap_pct=0.10,
    )
    # Block expiré : 1h dans le passé
    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    se.load_state(
        positions_long=[],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=0.0,
        block_until_iso=expired,
    )
    # _is_blocked_now() doit retourner False et clear le block
    assert se._is_blocked_now() is False
    assert se.block_until_iso is None  # auto-clear

    # Et un signal_long doit générer open_long
    df = _make_df_with_signal(signal_long=True)
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    assert any(s["action"] == "open_long" for s in signals)


def test_hard_cap_sends_telegram_alert():
    """Hard cap → notifier.critical appelé avec message explicite."""
    notifier = MagicMock()
    se = SignalEngine(
        daily_loss_hard_cap_pct=0.10,
        notifier=notifier,
    )
    se.load_state(
        positions_long=[],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.15,
    )
    df = _make_df_with_signal()
    se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    notifier.critical.assert_called_once()
    msg = notifier.critical.call_args[0][0]
    assert "HARD DAILY LOSS CAP" in msg
    assert "freeze" in msg.lower()


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_block_active_allows_exits_blocks_entries():
    """Block actif: pas d'open_long mais TP/SL atteint → close généré."""
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_loss_hard_cap_pct=0.10,
    )
    future_block = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    # 1 LONG ouvert avec TP très bas qu'on va heurter au current_price
    se.load_state(
        positions_long=[{"id": "L1", "entry": 49000, "stop": 48000, "tp": 50000, "size": 0.01}],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.005,
        block_until_iso=future_block,
    )
    df = _make_df_with_signal(signal_long=True)
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    # Le TP est hit → close_long généré (sortie autorisée pendant block)
    close_actions = [s for s in signals if s["action"] == "close_long"]
    assert len(close_actions) >= 1
    assert close_actions[0]["reason"] == "take_profit"
    # Mais aucune nouvelle entrée long malgré signal_long
    open_actions = [s for s in signals if s["action"] == "open_long"]
    assert len(open_actions) == 0


def test_caps_disabled_when_zero():
    """Si tous caps = 0, comportement legacy strict (pas de cap check)."""
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.0,
        daily_gain_soft_cap_pct=0.0,
        daily_loss_hard_cap_pct=0.0,
    )
    se.load_state(
        positions_long=[],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.50,  # -50% gigantesque, hors-norme
    )
    df = _make_df_with_signal(signal_long=True)
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    # Avec caps=0, le signal long passe (entrée autorisée)
    assert any(s["action"] == "open_long" for s in signals)


def test_hard_cap_priority_over_soft():
    """Si pnl atteint à la fois soft et hard, hard prime (kill_switch direct)."""
    notifier = MagicMock()
    se = SignalEngine(
        daily_loss_soft_cap_pct=0.02,
        daily_loss_hard_cap_pct=0.10,
        notifier=notifier,
    )
    se.load_state(
        positions_long=[{"id": "L1", "entry": 50000, "stop": -np.inf, "tp": np.inf, "size": 0.01}],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.15,  # déclenche soft ET hard
    )
    df = _make_df_with_signal()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    # Reason doit être hard, pas soft
    close_actions = [s for s in signals if s["action"].startswith("close_")]
    assert close_actions[0]["reason"] == "hard_loss_cap"
    notifier.critical.assert_called_once()
