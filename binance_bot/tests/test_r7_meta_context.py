"""R7 — Tests _build_meta_context (intraday_runner)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from routines.intraday_runner import _build_meta_context
from bot.trade_meta import MetaLabelLogger, build_meta_label
from bot.paper_trader import PaperTrader


# ---------------------------------------------------------------------------
# _build_meta_context
# ---------------------------------------------------------------------------


def test_meta_context_has_all_required_blocks():
    sig = {"action": "close_long", "reason": "take_profit", "exit": 50000.0}
    ctx = _build_meta_context(
        signal_id="sig_42", symbol="BTC/USDT", sig=sig,
        opened_at_iso="2026-04-28T08:00:00+00:00",
        phase_K3=2, atr_at_entry=120.5,
        regime_har="mid", composite_score=0.15,
    )
    # Top-level requis
    for k in ("trade_id", "timestamp_open", "timestamp_close", "symbol",
              "context", "pre_trade", "execution", "exit"):
        assert k in ctx
    # Sub-blocs requis
    for k in ("phase_K3", "days_since_halving", "day_of_week", "hour_of_day",
              "btc_dominance"):
        assert k in ctx["context"]
    for k in ("atr_at_entry", "rv_predicted_har", "regime_har",
              "vpin_at_entry", "composite_signal",
              "cloud_breakout_size_atr_units", "volume_relative_30d",
              "funding_rate_at_entry_bps"):
        assert k in ctx["pre_trade"]
    for k in ("tf_signal_origin", "n_tf_confirming", "slippage_bps"):
        assert k in ctx["execution"]
    assert "reason" in ctx["exit"]


def test_meta_context_normalizes_symbol_slash():
    sig = {"action": "close_long", "reason": "TP"}
    ctx = _build_meta_context(
        signal_id="x", symbol="ETH/USDT", sig=sig,
        opened_at_iso=None, phase_K3=None,
        atr_at_entry=None, regime_har=None, composite_score=None,
    )
    assert ctx["symbol"] == "ETHUSDT"


def test_meta_context_maps_take_profit_to_TP():
    """log_close émet reason='take_profit' qu'il faut mapper vers 'TP' (enum spec)."""
    sig = {"action": "close_long", "reason": "take_profit"}
    ctx = _build_meta_context(
        signal_id="x", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=0,
        atr_at_entry=None, regime_har=None, composite_score=None,
    )
    assert ctx["exit"]["reason"] == "TP"


def test_meta_context_maps_trailing_stop_to_trailing():
    sig = {"action": "close_short", "reason": "trailing_stop"}
    ctx = _build_meta_context(
        signal_id="x", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=0,
        atr_at_entry=None, regime_har=None, composite_score=None,
    )
    assert ctx["exit"]["reason"] == "trailing"


def test_meta_context_unknown_reason_falls_back_to_manual():
    sig = {"action": "close_long", "reason": "weird_reason_xyz"}
    ctx = _build_meta_context(
        signal_id="x", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=0,
        atr_at_entry=None, regime_har=None, composite_score=None,
    )
    assert ctx["exit"]["reason"] == "manual"


def test_meta_context_passes_build_meta_label_validation():
    """Le ctx produit doit être valide pour build_meta_label() sans raise."""
    sig = {"action": "close_long", "reason": "take_profit"}
    ctx = _build_meta_context(
        signal_id="trade_001", symbol="BTC/USDT", sig=sig,
        opened_at_iso="2026-04-28T08:00:00Z",
        phase_K3=1, atr_at_entry=800.0,
        regime_har="mid", composite_score=0.3,
    )
    # On simule les champs paper_trader-injectés
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
    assert payload["context"]["phase_K3"] == 1


def test_meta_context_handles_none_phase_as_zero():
    sig = {"action": "close_long", "reason": "TP"}
    ctx = _build_meta_context(
        signal_id="x", symbol="BTCUSDT", sig=sig,
        opened_at_iso=None, phase_K3=None,
        atr_at_entry=None, regime_har=None, composite_score=None,
    )
    assert ctx["context"]["phase_K3"] == 0


# ---------------------------------------------------------------------------
# Branchement E2E PaperTrader + meta_logger
# ---------------------------------------------------------------------------


def test_paper_trader_with_meta_logger_writes_jsonl_on_close(tmp_path):
    """Smoke E2E : log_close avec meta_context écrit dans trades_meta.jsonl."""
    csv_path = tmp_path / "paper.csv"
    meta_path = tmp_path / "trades_meta.jsonl"
    meta_logger = MetaLabelLogger(path=meta_path)
    trader = PaperTrader(log_path=csv_path, meta_logger=meta_logger)

    sig = {"action": "close_long", "reason": "take_profit"}
    ctx = _build_meta_context(
        signal_id="sig_R7_smoke", symbol="BTC/USDT", sig=sig,
        opened_at_iso="2026-04-28T08:00:00+00:00",
        phase_K3=2, atr_at_entry=120.0,
        regime_har="mid", composite_score=0.0,
    )
    trader.log_close(
        action="close_long", qty=0.01, exit_price=42500.0, entry_price=42000.0,
        live_order_id="live_99", signal_id="sig_R7_smoke",
        held_seconds=3600.0, meta_context=ctx,
    )

    entries = list(meta_logger.iter_entries())
    assert len(entries) == 1
    rec = entries[0]
    assert rec["payload"]["trade_id"] == "sig_R7_smoke"
    assert rec["payload"]["symbol"] == "BTCUSDT"
    assert rec["payload"]["exit"]["reason"] == "TP"
    # Hash chain doit être valide
    status = meta_logger.verify()
    assert status.valid is True
