"""P-MTF-5 — Tests execution_runner_15m.

Tests offline avec mocks pour les factories d'intraday_runner et CCXT.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from binance_bot.bot.state_manager import StateManager
from binance_bot.routines import execution_runner_15m as runner


def _fake_15m_df(close: float, n: int = 300) -> pd.DataFrame:
    np.random.seed(7)
    base = np.linspace(close - 1_000, close, n)
    closes = base + np.random.randn(n) * 50
    closes[-1] = close
    return pd.DataFrame({
        "timestamp": pd.date_range(end="2026-04-30", periods=n, freq="15min", tz="UTC"),
        "open": closes - 10,
        "high": closes + 50,
        "low": closes - 50,
        "close": closes,
        "volume": np.random.uniform(100, 200, n),
    })


@pytest.fixture
def tmp_state_with_h2(tmp_path):
    """state.json avec h2_trend frais pour BTC/USDT (long)."""
    state_file = tmp_path / "state.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    state_file.write_text(json.dumps({
        "initial_capital_usdt": 100.0,
        "rolling_equity_high_usdt": 100.0,
        "params_today": {"tenkan": 21, "kijun": 35, "senkou_b": 90, "shift": 44},
        "symbols": {
            "BTC/USDT": {
                "positions_long": [],
                "positions_short": [],
                "h2_trend": {
                    "direction": "long",
                    "last_close": 70_000.0,
                    "cloud_top": 65_000.0,
                    "cloud_bottom": 60_000.0,
                    "computed_at_iso": now_iso,
                    "reason": "above_cloud",
                },
            },
        },
    }))
    return state_file


# ---------------------------------------------------------------------------
# _make_trend_gate_fn
# ---------------------------------------------------------------------------


def test_trend_gate_returns_h2_when_fresh(tmp_state_with_h2):
    sm = StateManager(str(tmp_state_with_h2))
    gate = runner._make_trend_gate_fn(sm, "BTC/USDT", max_age_hours=3.0)
    h2 = gate()
    assert h2["direction"] == "long"
    assert h2["last_close"] == 70_000.0


def test_trend_gate_returns_flat_when_stale(tmp_path):
    state_file = tmp_path / "state.json"
    old_iso = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    state_file.write_text(json.dumps({
        "symbols": {"BTC/USDT": {"h2_trend": {"direction": "long", "computed_at_iso": old_iso}}},
    }))
    sm = StateManager(str(state_file))
    gate = runner._make_trend_gate_fn(sm, "BTC/USDT", max_age_hours=3.0)
    h2 = gate()
    assert h2["direction"] == "flat"
    assert h2["reason"] == "h2_stale"


def test_trend_gate_returns_flat_when_missing(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"symbols": {}}))
    sm = StateManager(str(state_file))
    gate = runner._make_trend_gate_fn(sm, "BTC/USDT", max_age_hours=3.0)
    h2 = gate()
    assert h2["direction"] == "flat"
    assert h2["reason"] in ("h2_stale", "h2_missing")


def test_trend_gate_closure_captures_symbol_correctly(tmp_path):
    """Test du default-arg trick : closures pour différents symboles ne se mélangent pas."""
    state_file = tmp_path / "state.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    state_file.write_text(json.dumps({
        "symbols": {
            "BTC/USDT": {"h2_trend": {"direction": "long", "computed_at_iso": now_iso}},
            "ETH/USDT": {"h2_trend": {"direction": "short", "computed_at_iso": now_iso}},
        },
    }))
    sm = StateManager(str(state_file))
    gates = []
    for sym in ["BTC/USDT", "ETH/USDT"]:
        gates.append(runner._make_trend_gate_fn(sm, sym, max_age_hours=3.0))
    assert gates[0]()["direction"] == "long"
    assert gates[1]()["direction"] == "short"


# ---------------------------------------------------------------------------
# _maybe_flat_eod (P-MTF-9)
# ---------------------------------------------------------------------------


def test_flat_eod_disabled_returns_empty(tmp_state_with_h2):
    sm = StateManager(str(tmp_state_with_h2))
    closed = runner._maybe_flat_eod({"flat_eod_enabled": False}, sm, ["BTC/USDT"])
    assert closed == []


def test_flat_eod_does_not_trigger_before_threshold(tmp_state_with_h2):
    sm = StateManager(str(tmp_state_with_h2))
    # 22:30 UTC : pas encore l'heure
    now = datetime(2026, 5, 1, 22, 30, tzinfo=timezone.utc)
    closed = runner._maybe_flat_eod(
        {"flat_eod_enabled": True, "flat_eod_minute_utc": 45},
        sm, ["BTC/USDT"], now_utc=now,
    )
    assert closed == []


def test_flat_eod_triggers_at_threshold_and_closes_positions(tmp_path):
    """À 23:45 UTC, ferme toutes les positions."""
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "symbols": {
            "BTC/USDT": {
                "positions_long": [
                    {"id": "long_1", "entry": 70_000.0, "stop": 65_000.0, "tp": 75_000.0, "size": 0.01},
                ],
                "positions_short": [
                    {"id": "short_1", "entry": 70_000.0, "stop": 75_000.0, "tp": 65_000.0, "size": 0.01},
                ],
            },
        },
    }))
    sm = StateManager(str(state_file))
    now = datetime(2026, 5, 1, 23, 45, tzinfo=timezone.utc)
    closed = runner._maybe_flat_eod(
        {"flat_eod_enabled": True, "flat_eod_minute_utc": 45},
        sm, ["BTC/USDT"], now_utc=now,
    )
    assert len(closed) == 2
    assert {c["side"] for c in closed} == {"long", "short"}
    assert all(c["reason"] == "flat_eod" for c in closed)
    # Vérifier que state.json a bien été mis à jour
    persisted = json.loads(state_file.read_text())
    assert persisted["symbols"]["BTC/USDT"]["positions_long"] == []
    assert persisted["symbols"]["BTC/USDT"]["positions_short"] == []


def test_flat_eod_no_positions_returns_empty(tmp_state_with_h2):
    sm = StateManager(str(tmp_state_with_h2))
    now = datetime(2026, 5, 1, 23, 50, tzinfo=timezone.utc)
    closed = runner._maybe_flat_eod(
        {"flat_eod_enabled": True, "flat_eod_minute_utc": 45},
        sm, ["BTC/USDT"], now_utc=now,
    )
    # Pas de positions ouvertes dans la fixture
    assert closed == []


# ---------------------------------------------------------------------------
# main() — test E2E mocké
# ---------------------------------------------------------------------------


def test_main_skips_when_multi_tf_disabled(tmp_state_with_h2, monkeypatch):
    monkeypatch.setattr(runner, "_load_settings", lambda: {"multi_tf_enabled": False})
    monkeypatch.setattr(runner, "ROOT", tmp_state_with_h2.parent.parent)
    rc = runner.main()
    assert rc == 0  # exit clean, pas d'erreur


def test_main_skips_when_kill_switch_present(tmp_path, monkeypatch):
    """Si data/.killed existe → exit 0 sans rien faire."""
    binance_bot_dir = tmp_path / "binance_bot"
    data_dir = binance_bot_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / ".killed").write_text("killed for test")
    state_file = data_dir / "state.json"
    state_file.write_text(json.dumps({"symbols": {}}))

    monkeypatch.setattr(runner, "_load_settings", lambda: {
        "multi_tf_enabled": True,
        "state_file": "data/state.json",
    })
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    rc = runner.main()
    assert rc == 0
