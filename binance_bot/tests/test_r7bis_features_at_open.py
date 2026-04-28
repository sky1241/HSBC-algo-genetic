"""R7-bis — Tests capture features_snapshot à open + lecture au close.

Résout L-002 : 6 features pre_trade meta_context étaient None par
défaut. R7-bis capture les valeurs au moment de l'open dans
state_manager.add_position(features_snapshot=...) et les passe à
_build_meta_context au close via entry_features=...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from bot.state_manager import StateManager
from bot.trade_meta import build_meta_label
from routines.intraday_runner import (
    _build_features_snapshot_at_open,
    _build_meta_context,
)


# ---------------------------------------------------------------------------
# state_manager.add_position(features_snapshot=...)
# ---------------------------------------------------------------------------


def test_state_manager_add_position_accepts_features_snapshot(tmp_path):
    """add_position(features_snapshot={...}) → snapshot stocké dans la position."""
    state_path = tmp_path / "state.json"
    sm = StateManager(state_file=str(state_path))
    sm.add_position(
        side="long", entry=50000.0, stop=48000.0, tp=52000.0, size=0.01,
        symbol="BTC/USDT",
        features_snapshot={"atr_at_entry": 800.0, "regime_har": "mid"},
    )
    pos = sm.state["symbols"]["BTC/USDT"]["positions_long"][0]
    assert "features_snapshot" in pos
    assert pos["features_snapshot"]["atr_at_entry"] == 800.0
    assert pos["features_snapshot"]["regime_har"] == "mid"


def test_state_manager_add_position_backward_compat_no_snapshot(tmp_path):
    """Sans features_snapshot → backward-compat. Pas de clé."""
    state_path = tmp_path / "state.json"
    sm = StateManager(state_file=str(state_path))
    sm.add_position(
        side="long", entry=50000.0, stop=48000.0, tp=52000.0, size=0.01,
        symbol="BTC/USDT",
    )
    pos = sm.state["symbols"]["BTC/USDT"]["positions_long"][0]
    assert "features_snapshot" not in pos


def test_state_manager_add_position_invalid_snapshot_safe_fallback(tmp_path):
    """features_snapshot non-dict-castable → safe fallback (pas de clé)."""
    state_path = tmp_path / "state.json"
    sm = StateManager(state_file=str(state_path))
    # Object qui ne peut pas être dict()-cast: int, str, etc.
    sm.add_position(
        side="long", entry=50000.0, stop=48000.0, tp=52000.0, size=0.01,
        symbol="BTC/USDT",
        features_snapshot=42,  # type: ignore  # provoquera TypeError
    )
    pos = sm.state["symbols"]["BTC/USDT"]["positions_long"][0]
    # Pas de crash, et pas de clé features_snapshot
    assert "features_snapshot" not in pos


# ---------------------------------------------------------------------------
# _build_meta_context(entry_features=...)
# ---------------------------------------------------------------------------


def test_build_meta_context_uses_entry_features_when_provided():
    """entry_features fourni → ses valeurs prennent priorité."""
    sig = {"action": "close_long", "reason": "TP"}
    ef = {
        "atr_at_entry": 800.0,
        "rv_predicted_har": 0.025,
        "regime_har": "high",
        "vpin_at_entry": 0.42,
        "obi_at_entry": 0.65,
        "composite_signal": 0.31,
        "cloud_breakout_size_atr_units": 1.4,
        "volume_relative_30d": 1.15,
        "funding_rate_at_entry_bps": 1.2,
        "btc_dominance": 0.52,
    }
    ctx = _build_meta_context(
        signal_id="t1", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=1,
        # Compute-at-close valeurs (devraient être OVERRIDEÉES par ef)
        atr_at_entry=999.0, regime_har="low", composite_score=0.0,
        entry_features=ef,
    )
    # entry_features.atr=800 doit prendre priorité sur arg=999
    assert ctx["pre_trade"]["atr_at_entry"] == 800.0
    assert ctx["pre_trade"]["rv_predicted_har"] == 0.025
    assert ctx["pre_trade"]["regime_har"] == "high"  # ef override "low"
    assert ctx["pre_trade"]["vpin_at_entry"] == 0.42
    assert ctx["pre_trade"]["obi_at_entry"] == 0.65
    assert ctx["pre_trade"]["composite_signal"] == 0.31  # ef override 0.0
    assert ctx["pre_trade"]["cloud_breakout_size_atr_units"] == 1.4
    assert ctx["pre_trade"]["volume_relative_30d"] == 1.15
    assert ctx["pre_trade"]["funding_rate_at_entry_bps"] == 1.2
    assert ctx["context"]["btc_dominance"] == 0.52


def test_build_meta_context_falls_back_to_close_time_when_no_snapshot():
    """entry_features=None → comportement R7 inchangé (compute-at-close)."""
    sig = {"action": "close_long", "reason": "TP"}
    ctx = _build_meta_context(
        signal_id="t2", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=1,
        atr_at_entry=120.0, regime_har="mid", composite_score=0.15,
        entry_features=None,
    )
    # Fallback : valeurs compute-at-close utilisées
    assert ctx["pre_trade"]["atr_at_entry"] == 120.0
    assert ctx["pre_trade"]["regime_har"] == "mid"
    assert ctx["pre_trade"]["composite_signal"] == 0.15
    # Les features non capturées restent None
    assert ctx["pre_trade"]["rv_predicted_har"] is None
    assert ctx["pre_trade"]["vpin_at_entry"] is None


def test_build_meta_context_partial_entry_features_uses_fallback_for_missing():
    """entry_features partiel → keys présentes priorité, manquantes fallback."""
    sig = {"action": "close_long", "reason": "TP"}
    ef = {"atr_at_entry": 500.0}  # only atr
    ctx = _build_meta_context(
        signal_id="t3", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=0,
        atr_at_entry=999.0, regime_har="low", composite_score=0.5,
        entry_features=ef,
    )
    assert ctx["pre_trade"]["atr_at_entry"] == 500.0  # ef priorité
    assert ctx["pre_trade"]["regime_har"] == "low"    # fallback (ef.regime_har absent)
    assert ctx["pre_trade"]["composite_signal"] == 0.5  # fallback


# ---------------------------------------------------------------------------
# _build_features_snapshot_at_open
# ---------------------------------------------------------------------------


def test_features_snapshot_with_all_sources_available():
    """Toutes les sources mockées → snapshot rempli."""
    df = pd.DataFrame({
        "ATR": [120.0],
        "close": [50100.0],
        "senkou_a": [49800.0],
        "senkou_b": [49500.0],
    })
    snap = _build_features_snapshot_at_open(
        symbol="BTCUSDT",
        df_ichimoku=df,
        returns_1h_series=None,  # rv_predicted_har laissé None
        regime_gate_fn=lambda: "mid",
        composite_log_fn=lambda: {"score": 0.42, "components": {}},
        vpin_data_fn=lambda: (0.55, 0.78),
    )
    assert snap["atr_at_entry"] == 120.0
    assert snap["regime_har"] == "mid"
    assert snap["composite_signal"] == pytest.approx(0.42)
    assert snap["vpin_at_entry"] == pytest.approx(0.55)
    assert snap["obi_at_entry"] == pytest.approx(0.78)
    # cloud_breakout : close=50100 > cloud_top=49800 → (50100-49800)/120 = 2.5
    assert snap["cloud_breakout_size_atr_units"] == pytest.approx(2.5)
    # rv_predicted_har None car returns_1h absent
    assert snap["rv_predicted_har"] is None


def test_features_snapshot_with_no_sources_returns_none_values():
    """Sources absentes → snap clés présentes mais None (pas crash)."""
    df = pd.DataFrame({"ATR": [0.0]})  # ATR=0 → atr=None
    snap = _build_features_snapshot_at_open(
        symbol="BTCUSDT",
        df_ichimoku=df,
        returns_1h_series=None,
        regime_gate_fn=None,
        composite_log_fn=None,
        vpin_data_fn=None,
    )
    assert snap["atr_at_entry"] is None  # ATR=0 → None
    assert snap["regime_har"] is None
    assert snap["composite_signal"] is None
    assert snap["vpin_at_entry"] is None
    assert snap["obi_at_entry"] is None


def test_features_snapshot_handles_callbacks_raising():
    """Si callbacks raise → snap clés None safe (pas crash)."""
    def boom_regime():
        raise RuntimeError("regime fail")
    def boom_composite():
        raise RuntimeError("composite fail")
    def boom_vpin():
        raise RuntimeError("vpin fail")
    df = pd.DataFrame({"ATR": [100.0], "close": [50000], "senkou_a": [49000], "senkou_b": [48500]})
    snap = _build_features_snapshot_at_open(
        symbol="BTCUSDT", df_ichimoku=df, returns_1h_series=None,
        regime_gate_fn=boom_regime,
        composite_log_fn=boom_composite,
        vpin_data_fn=boom_vpin,
    )
    # Tous les callbacks crashent mais snap reste cohérent
    assert snap["regime_har"] is None
    assert snap["composite_signal"] is None
    assert snap["vpin_at_entry"] is None
    # ATR toujours capturé (pas de callback)
    assert snap["atr_at_entry"] == 100.0


def test_features_snapshot_cloud_breakout_inside_cloud_returns_zero():
    """Si close ∈ [cloud_bot, cloud_top] → breakout_size = 0."""
    df = pd.DataFrame({
        "ATR": [100.0], "close": [49000], "senkou_a": [48800], "senkou_b": [49200],
    })
    snap = _build_features_snapshot_at_open(
        symbol="BTCUSDT", df_ichimoku=df, returns_1h_series=None,
        regime_gate_fn=None, composite_log_fn=None, vpin_data_fn=None,
    )
    assert snap["cloud_breakout_size_atr_units"] == 0.0


# ---------------------------------------------------------------------------
# Integration : meta_label via build_meta_label valide après merge entry_features
# ---------------------------------------------------------------------------


def test_e2e_meta_label_with_entry_features_passes_validation():
    """Meta_context produit avec entry_features → build_meta_label OK."""
    sig = {"action": "close_long", "reason": "take_profit"}
    ef = {
        "atr_at_entry": 800.0, "rv_predicted_har": 0.025,
        "regime_har": "mid", "vpin_at_entry": 0.42, "obi_at_entry": 0.7,
        "composite_signal": 0.31, "cloud_breakout_size_atr_units": 1.4,
        "volume_relative_30d": 1.15, "funding_rate_at_entry_bps": 1.2,
        "btc_dominance": 0.52,
    }
    ctx = _build_meta_context(
        signal_id="trade_001", symbol="BTC/USDT", sig=sig,
        opened_at_iso="2026-04-28T08:00:00Z",
        phase_K3=1, atr_at_entry=999.0,
        regime_har="low", composite_score=0.0,
        entry_features=ef,
    )
    payload = build_meta_label(
        trade_id=ctx["trade_id"],
        timestamp_open=ctx["timestamp_open"],
        timestamp_close=ctx["timestamp_close"],
        symbol=ctx["symbol"],
        side="LONG",
        entry_price=42000.0, exit_price=42500.0, qty=0.01,
        pnl_gross_usd=5.0, fees_paid_usd=0.34,
        funding_paid_usd=0.05, pnl_net_usd=4.61,
        context=ctx["context"], pre_trade=ctx["pre_trade"],
        execution=ctx["execution"], exit_info=ctx["exit"],
    )
    assert payload["trade_id"] == "trade_001"
    # Verify entry_features valeurs persistent (priorité respectée)
    assert payload["pre_trade"]["atr_at_entry"] == 800.0
    assert payload["pre_trade"]["regime_har"] == "mid"
    assert payload["pre_trade"]["vpin_at_entry"] == 0.42
    assert payload["pre_trade"]["obi_at_entry"] == 0.7
    assert payload["pre_trade"]["funding_rate_at_entry_bps"] == 1.2
