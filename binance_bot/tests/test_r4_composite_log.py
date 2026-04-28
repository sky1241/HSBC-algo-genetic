"""R4 — Tests P6.5 composite signal log-only branchement signal_engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.signal_engine import SignalEngine


# ---------------------------------------------------------------------------
# _log_composite_signal direct
# ---------------------------------------------------------------------------


def test_composite_log_no_fn_no_op():
    """Pas de callback → no-op silencieux."""
    eng = SignalEngine()
    eng._log_composite_signal()  # ne raise pas
    assert eng.composite_log_fn is None


def test_composite_log_fn_called_and_persisted_to_jsonl(tmp_path):
    """Callback appelé + JSONL append + ts_ms ajouté."""
    log_path = tmp_path / "composite_log.jsonl"
    fn = MagicMock(return_value={
        "score": 0.42,
        "raw_score": 0.42,
        "components": {"top_ls": 1.5, "taker": -0.3, "liq": 0.5, "oi": 0.1},
        "symbol": "BTCUSDT",
        "n_obs": {"top_ls": 100, "taker": 100, "liq": 100, "oi": 100},
    })
    eng = SignalEngine(composite_log_fn=fn, composite_log_path=log_path)
    eng._log_composite_signal()
    fn.assert_called_once()
    assert log_path.exists()
    line = log_path.read_text().strip()
    rec = json.loads(line)
    assert rec["score"] == pytest.approx(0.42)
    assert rec["symbol"] == "BTCUSDT"
    assert "ts_ms" in rec  # ajouté par SignalEngine


def test_composite_log_appends_multiple_lines(tmp_path):
    """Calls multiples → JSONL append (pas écrasement)."""
    log_path = tmp_path / "composite_log.jsonl"
    fn = MagicMock(return_value={"score": 0.1, "symbol": "ETHUSDT"})
    eng = SignalEngine(composite_log_fn=fn, composite_log_path=log_path)
    eng._log_composite_signal()
    eng._log_composite_signal()
    eng._log_composite_signal()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3


def test_composite_log_callback_exception_safe_silent(tmp_path):
    """Si callback raise → silent, pas d'écriture, pas de crash bot."""
    log_path = tmp_path / "composite_log.jsonl"
    def boom():
        raise RuntimeError("flow files unreadable")
    eng = SignalEngine(composite_log_fn=boom, composite_log_path=log_path)
    eng._log_composite_signal()  # ne raise pas
    assert not log_path.exists()  # pas d'écriture


def test_composite_log_invalid_return_silent(tmp_path):
    """Callback retourne non-dict → silent."""
    log_path = tmp_path / "composite_log.jsonl"
    eng = SignalEngine(composite_log_fn=lambda: "not a dict",
                       composite_log_path=log_path)
    eng._log_composite_signal()
    assert not log_path.exists()


def test_composite_log_no_path_does_not_persist(tmp_path):
    """composite_log_path=None → callback called mais pas de JSONL."""
    fn = MagicMock(return_value={"score": 0.1, "symbol": "BTCUSDT"})
    eng = SignalEngine(composite_log_fn=fn, composite_log_path=None)
    eng._log_composite_signal()
    fn.assert_called_once()
    # Pas de fichier créé (path=None) — rien à vérifier sauf no-crash.


def test_composite_log_notifier_info_called(tmp_path):
    """Si notifier set → notifier.info appelé avec score formaté."""
    notifier = MagicMock()
    fn = lambda: {
        "score": 0.5,
        "components": {"top_ls": 1.2, "taker": -0.5, "liq": 0.3, "oi": 0.1},
        "symbol": "BTCUSDT",
    }
    eng = SignalEngine(composite_log_fn=fn, composite_log_path=None,
                       notifier=notifier)
    eng._log_composite_signal()
    notifier.info.assert_called_once()
    msg = notifier.info.call_args.args[0]
    assert "BTCUSDT" in msg and "+0.500" in msg


# ---------------------------------------------------------------------------
# Hook dans detect_signals
# ---------------------------------------------------------------------------


def test_detect_signals_invokes_composite_log(tmp_path):
    """detect_signals appelle _log_composite_signal au début du cycle."""
    import pandas as pd
    log_path = tmp_path / "composite_log.jsonl"
    fn = MagicMock(return_value={"score": 0.0, "symbol": "BTCUSDT"})
    eng = SignalEngine(composite_log_fn=fn, composite_log_path=log_path)
    # df Ichimoku minimal mais non vide
    df = pd.DataFrame({
        "ATR": [100.0],
        "signal_long": [False],
        "signal_short": [False],
    })
    eng.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0},
                       current_price=50000.0)
    fn.assert_called_once()
    assert log_path.exists()


def test_detect_signals_empty_df_no_log_call():
    """df vide → return [] tôt, pas d'appel composite_log_fn."""
    import pandas as pd
    fn = MagicMock(return_value={"score": 0.0, "symbol": "BTC"})
    eng = SignalEngine(composite_log_fn=fn)
    eng.detect_signals(pd.DataFrame(), params={}, current_price=0.0)
    fn.assert_not_called()  # short-circuit avant le log


# ---------------------------------------------------------------------------
# Factory _make_composite_log_fn (intraday_runner)
# ---------------------------------------------------------------------------


def test_factory_produces_callable_returning_symbol():
    """_make_composite_log_fn(symbol, data_dir) → fn() retourne dict avec symbol."""
    from routines.intraday_runner import _make_composite_log_fn
    fn = _make_composite_log_fn("BTC/USDT", "/tmp/nonexistent_data_dir_xyz")
    result = fn()
    # Pas de fichiers data → compute_composite_score retourne tous z=0
    assert isinstance(result, dict)
    assert result.get("symbol") == "BTCUSDT"  # slash stripped + upper
