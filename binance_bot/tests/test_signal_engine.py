#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests unitaires: vérifier parité signal_engine vs backtest."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.signal_engine import SignalEngine


def test_signal_long_detection():
    """Test: détection signal LONG."""
    engine = SignalEngine()
    
    df = pd.DataFrame({
        'close': [50000],
        'ATR': [400],
        'signal_long': [True],
        'signal_short': [False],
    })
    
    params = {"atr_mult": 10.0, "tp_mult": 20.0}
    signals = engine.detect_signals(df, params, current_price=50000)
    
    assert len(signals) == 1
    assert signals[0]['action'] == 'open_long'
    assert signals[0]['entry'] == 50000
    assert signals[0]['stop'] == 50000 - (400 * 10.0 * 2.0)  # Entry - ATR*atr_mult*2
    assert signals[0]['tp'] == 50000 + (400 * 20.0)  # Entry + ATR*tp_mult
    
    print("✅ test_signal_long_detection PASS")


def test_tp_hit():
    """Test: sortie par take profit."""
    engine = SignalEngine()
    
    # Simuler position ouverte
    engine.positions_long = [{
        "id": "test_1",
        "entry": 50000,
        "stop": 48000,
        "tp": 55000,
        "size": 0.01
    }]
    
    df = pd.DataFrame({'close': [56000], 'ATR': [400], 'signal_long': [False], 'signal_short': [False]})
    params = {"atr_mult": 10.0, "tp_mult": 20.0}
    signals = engine.detect_signals(df, params, current_price=56000)
    
    assert len(signals) == 1
    assert signals[0]['action'] == 'close_long'
    assert signals[0]['reason'] == 'take_profit'
    
    print("✅ test_tp_hit PASS")


def test_stop_hit():
    """Test: sortie par trailing stop."""
    engine = SignalEngine()
    
    engine.positions_long = [{
        "id": "test_2",
        "entry": 50000,
        "stop": 48000,
        "tp": 55000,
        "size": 0.01
    }]
    
    df = pd.DataFrame({'close': [47000], 'ATR': [400], 'signal_long': [False], 'signal_short': [False]})
    params = {"atr_mult": 10.0, "tp_mult": 20.0}
    signals = engine.detect_signals(df, params, current_price=47000)
    
    assert len(signals) == 1
    assert signals[0]['action'] == 'close_long'
    assert signals[0]['reason'] == 'trailing_stop'
    
    print("✅ test_stop_hit PASS")


if __name__ == "__main__":
    test_signal_long_detection()
    test_tp_hit()
    test_stop_hit()
    print("\n✅ Tous les tests PASS")

