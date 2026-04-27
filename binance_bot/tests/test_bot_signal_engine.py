"""Tests pour SignalEngine — focus BUG-003 (trailing stop ATR)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from services.signal_engine import SignalEngine


def _make_df(close: float, atr: float, signal_long: bool = False, signal_short: bool = False):
    """Construit un DataFrame Ichimoku minimal pour la dernière bougie."""
    return pd.DataFrame({
        'close': [close],
        'ATR': [atr],
        'signal_long': [signal_long],
        'signal_short': [signal_short],
    })


# ============================================================
# BUG-003 — trailing stop ATR
# ============================================================

def test_trailing_stop_ratchets_up_long():
    """Quand le prix monte sur un LONG, le stop suit (cliquet)."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "long_1", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01}
    engine.positions_long = [pos]

    # Prix monte à 52000, ATR=200 → nouveau stop = 52000 - 2*200 = 51600
    df = _make_df(close=52000, atr=200)
    engine.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=52000)

    assert pos["stop"] == pytest.approx(51600.0)


def test_trailing_stop_ratchets_down_short():
    """Quand le prix baisse sur un SHORT, le stop suit (cliquet)."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "short_1", "entry": 50000, "stop": 51000, "tp": 40000, "size": 0.01}
    engine.positions_short = [pos]

    # Prix baisse à 48000, ATR=200 → nouveau stop = 48000 + 400 = 48400
    df = _make_df(close=48000, atr=200)
    engine.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=48000)

    assert pos["stop"] == pytest.approx(48400.0)


def test_trailing_stop_never_loosens_long():
    """Si le prix RECULE sur un LONG, le stop ne descend pas."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "long_1", "entry": 50000, "stop": 51500, "tp": 60000, "size": 0.01}
    engine.positions_long = [pos]

    # Prix retombe à 51000, candidat stop = 51000 - 400 = 50600 < 51500 → on garde 51500
    df = _make_df(close=51000, atr=200)
    engine.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=51000)

    assert pos["stop"] == pytest.approx(51500.0)


def test_trailing_stop_never_loosens_short():
    """Si le prix REMONTE sur un SHORT, le stop ne remonte pas."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "short_1", "entry": 50000, "stop": 48500, "tp": 40000, "size": 0.01}
    engine.positions_short = [pos]

    # Prix remonte à 49000, candidat stop = 49400 > 48500 → on garde 48500
    df = _make_df(close=49000, atr=200)
    engine.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=49000)

    assert pos["stop"] == pytest.approx(48500.0)


def test_trailing_stop_no_op_when_atr_zero():
    """Si ATR=0 (donnée manquante), on ne touche à rien."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "long_1", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01}
    engine.positions_long = [pos]

    df = _make_df(close=52000, atr=0)
    engine.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=52000)

    assert pos["stop"] == 49000  # inchangé


def test_trailing_stop_triggers_close_after_ratchet():
    """Après ratchet, si le prix retraverse le nouveau stop, on close."""
    engine = SignalEngine(atr_trailing_mult=2.0)
    pos = {"id": "long_1", "entry": 50000, "stop": 49000, "tp": 60000, "size": 0.01}
    engine.positions_long = [pos]

    # Bougie 1 : prix 52000, ATR 200 → stop monte à 51600
    df1 = _make_df(close=52000, atr=200)
    sigs1 = engine.detect_signals(df1, params={"atr_mult": 10, "tp_mult": 20}, current_price=52000)
    assert len(sigs1) == 0
    assert pos["stop"] == pytest.approx(51600.0)

    # Bougie 2 : prix retombe à 51500 < stop 51600 → close
    df2 = _make_df(close=51500, atr=200)
    sigs2 = engine.detect_signals(df2, params={"atr_mult": 10, "tp_mult": 20}, current_price=51500)
    closes = [s for s in sigs2 if s.get("action") == "close_long"]
    assert len(closes) == 1
    assert closes[0]["reason"] == "trailing_stop"


def test_default_atr_trailing_mult_is_two():
    engine = SignalEngine()
    assert engine.atr_trailing_mult == 2.0
