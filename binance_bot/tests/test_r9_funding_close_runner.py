"""R9 — Tests d'intégration funding_close_runner (P11 live wire)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from routines import funding_close_runner


# ---------------------------------------------------------------------------
# main() outside settlement window
# ---------------------------------------------------------------------------


def test_main_skips_outside_settlement_window(tmp_path, monkeypatch):
    """Hors fenêtre [HH:55, HH:00) → return 0, pas de fetch / pas d'audit."""
    # Force ROOT pour state path absent
    monkeypatch.setattr(funding_close_runner, "ROOT", tmp_path)
    # Outside window (10:00 UTC = pas dans [00:55-59, 08:55-59, 16:55-59])
    fake_now = datetime(2026, 4, 28, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(funding_close_runner, "datetime", _DateTimeStub(fake_now))

    rc = funding_close_runner.main()
    assert rc == 0


# ---------------------------------------------------------------------------
# main() inside window with no positions
# ---------------------------------------------------------------------------


def test_main_skips_when_no_open_positions(tmp_path, monkeypatch):
    """Dans fenêtre + pas de positions ouvertes → return 0 sans fetch."""
    state_path = tmp_path / "data" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"symbols": {}}))
    monkeypatch.setattr(funding_close_runner, "ROOT", tmp_path)
    fake_now = datetime(2026, 4, 28, 7, 56, tzinfo=timezone.utc)  # 4 min avant 08:00
    monkeypatch.setattr(funding_close_runner, "datetime", _DateTimeStub(fake_now))

    rc = funding_close_runner.main()
    assert rc == 0


# ---------------------------------------------------------------------------
# main() with positions, dry-run mode (default safety)
# ---------------------------------------------------------------------------


def test_main_dry_run_logs_decisions_no_orders(tmp_path, monkeypatch):
    """Dans fenêtre + positions + DRY-RUN → log décisions, aucun ordre exécuté."""
    state_path = tmp_path / "data" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "symbols": {
            "BTC/USDT": {
                "positions_long": [{"id": "long_1", "entry": 50000.0, "size": 0.01}],
            }
        }
    }))
    cfg_path = tmp_path / "configs" / "bot_settings.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("trade_mode: simulation\nfunding_close_threshold_long_bps: 5.0\n")

    monkeypatch.setattr(funding_close_runner, "ROOT", tmp_path)
    fake_now = datetime(2026, 4, 28, 7, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(funding_close_runner, "datetime", _DateTimeStub(fake_now))
    # Mock fetch funding → high (déclenche close)
    monkeypatch.setattr(funding_close_runner, "_fetch_funding_rate_bps", lambda s: 8.0)
    # Pas de live env var → DRY-RUN
    monkeypatch.delenv("HSBC_FUNDING_CLOSE_LIVE", raising=False)
    # Mock _execute_close pour vérifier qu'il n'est PAS appelé
    exec_called = {"flag": False}
    def _no_exec(*a, **k):
        exec_called["flag"] = True
        return None
    monkeypatch.setattr(funding_close_runner, "_execute_close", _no_exec)

    rc = funding_close_runner.main()
    assert rc == 0
    assert exec_called["flag"] is False  # DRY-RUN → pas d'exec


def test_main_live_mode_requires_both_env_and_trade_mode(tmp_path, monkeypatch):
    """LIVE mode actif uniquement si HSBC_FUNDING_CLOSE_LIVE=1 ET trade_mode=live."""
    state_path = tmp_path / "data" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "symbols": {
            "BTC/USDT": {
                "positions_long": [{"id": "long_1", "entry": 50000.0, "size": 0.01}],
            }
        }
    }))
    cfg_path = tmp_path / "configs" / "bot_settings.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # trade_mode=simulation → pas live même avec env=1
    cfg_path.write_text("trade_mode: simulation\nfunding_close_threshold_long_bps: 5.0\n")

    monkeypatch.setattr(funding_close_runner, "ROOT", tmp_path)
    fake_now = datetime(2026, 4, 28, 7, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(funding_close_runner, "datetime", _DateTimeStub(fake_now))
    monkeypatch.setattr(funding_close_runner, "_fetch_funding_rate_bps", lambda s: 8.0)
    monkeypatch.setenv("HSBC_FUNDING_CLOSE_LIVE", "1")
    exec_called = {"flag": False}
    monkeypatch.setattr(funding_close_runner, "_execute_close",
                        lambda *a, **k: (exec_called.update({"flag": True}) or None))

    rc = funding_close_runner.main()
    assert rc == 0
    # trade_mode=simulation domine → pas d'exec live
    assert exec_called["flag"] is False


def test_main_live_mode_executes_when_both_set(tmp_path, monkeypatch):
    """LIVE = env=1 ET trade_mode=live → _execute_close appelé."""
    state_path = tmp_path / "data" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "symbols": {
            "BTC/USDT": {
                "positions_long": [{"id": "long_1", "entry": 50000.0, "size": 0.01}],
            }
        }
    }))
    cfg_path = tmp_path / "configs" / "bot_settings.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("trade_mode: live\nfunding_close_threshold_long_bps: 5.0\n")

    monkeypatch.setattr(funding_close_runner, "ROOT", tmp_path)
    fake_now = datetime(2026, 4, 28, 7, 56, tzinfo=timezone.utc)
    monkeypatch.setattr(funding_close_runner, "datetime", _DateTimeStub(fake_now))
    monkeypatch.setattr(funding_close_runner, "_fetch_funding_rate_bps", lambda s: 8.0)
    monkeypatch.setenv("HSBC_FUNDING_CLOSE_LIVE", "1")
    exec_calls = []
    monkeypatch.setattr(funding_close_runner, "_execute_close",
                        lambda sig, pos, *a, **k: exec_calls.append(sig) or "fake_order_42")

    rc = funding_close_runner.main()
    assert rc == 0
    assert len(exec_calls) == 1
    assert exec_calls[0]["side"] == "long"


# ---------------------------------------------------------------------------
# _flatten_positions
# ---------------------------------------------------------------------------


def test_flatten_positions_extracts_long_and_short():
    state = {
        "symbols": {
            "BTC/USDT": {
                "positions_long": [{"id": "L1", "entry": 50000, "size": 0.01}],
                "positions_short": [{"id": "S1", "entry": 49000, "size": 0.02}],
            },
            "ETH/USDT": {
                "positions_long": [{"id": "L2", "entry": 2500, "size": 0.05}],
                "positions_short": [],
            },
        }
    }
    out = funding_close_runner._flatten_positions(state)
    assert len(out) == 3
    sides = {(p["symbol"], p["side"]) for p in out}
    assert ("BTC/USDT", "long") in sides
    assert ("BTC/USDT", "short") in sides
    assert ("ETH/USDT", "long") in sides


def test_flatten_positions_empty_state():
    assert funding_close_runner._flatten_positions({}) == []
    assert funding_close_runner._flatten_positions({"symbols": {}}) == []


# ---------------------------------------------------------------------------
# Helper datetime stub
# ---------------------------------------------------------------------------


class _DateTimeStub:
    """Stub pour datetime.now() dans le module testé."""
    def __init__(self, fake_now: datetime):
        self._fake = fake_now
        # Délègue les attributs comme datetime, timezone, fromisoformat...
        from datetime import datetime as _real_dt, timezone as _real_tz, timedelta as _real_td
        self._real_dt = _real_dt
        self.timezone = _real_tz
        self.timedelta = _real_td

    @property
    def datetime(self):
        return self._real_dt

    def now(self, tz=None):
        if tz is not None:
            return self._fake.astimezone(tz)
        return self._fake.replace(tzinfo=None)

    def fromisoformat(self, s):
        return self._real_dt.fromisoformat(s)
