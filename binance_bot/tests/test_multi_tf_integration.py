"""P-MTF-10 — Tests d'intégration end-to-end multi-TF.

Pipeline complet : trend_filter_h2 → state.json → SignalEngine.trend_gate_fn →
détection signal LONG/SHORT 15m → vérification du blocage selon h2_trend.

Utilise un DataFrame Ichimoku 15m réel (calculé sur données synthétiques)
pour avoir signal_long/signal_short déterministes, puis injecte différents
h2_trend dans le SignalEngine via trend_gate_fn.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from binance_bot.services.ichimoku_engine import calculate_ichimoku
from binance_bot.services.signal_engine import SignalEngine


# ---------------------------------------------------------------------------
# Helpers : génère un DF 15m avec un signal long ou short forcé
# ---------------------------------------------------------------------------


def _make_df_with_signal(side: str, n: int = 200) -> pd.DataFrame:
    """Construit un DF Ichimoku 15m avec signal_long ou signal_short FORCÉ
    sur la dernière bougie (truquage post-calcul pour tests déterministes).

    On commence par un DF Ichimoku réel (calculé), puis on remplace
    last['signal_long'/'signal_short'] explicitement. ATR et cloud_top/bottom
    restent réels pour que sized/stop/tp soient calculables.
    """
    np.random.seed(0)
    base = np.linspace(60_000, 70_000, n)
    noise = np.random.randn(n) * 100
    closes = base + noise
    df = pd.DataFrame({
        "timestamp": pd.date_range(end="2026-04-30", periods=n, freq="15min", tz="UTC"),
        "open": closes - 50,
        "high": closes + 300,
        "low": closes - 300,
        "close": closes,
        "volume": np.full(n, 100.0),
    })
    df = calculate_ichimoku(df, tenkan=9, kijun=26, senkou_b=52, shift=26)
    # Force le signal sur la dernière bougie
    df.loc[df.index[-1], "signal_long"] = (side == "long")
    df.loc[df.index[-1], "signal_short"] = (side == "short")
    # S'assurer que ATR est non-NaN (sinon SignalEngine peut sortir tôt)
    if pd.isna(df.iloc[-1].get("ATR")):
        df.loc[df.index[-1], "ATR"] = 100.0
    # S'assurer que cloud_top/bottom sont non-NaN
    if pd.isna(df.iloc[-1].get("cloud_top")):
        df.loc[df.index[-1], "cloud_top"] = float(df.iloc[-1]["close"]) - 500
        df.loc[df.index[-1], "cloud_bottom"] = float(df.iloc[-1]["close"]) - 1000
    return df


def _h2_snapshot(direction: str) -> dict:
    """Génère un snapshot h2_trend frais."""
    return {
        "direction": direction,
        "last_close": 70_000.0,
        "cloud_top": 65_000.0,
        "cloud_bottom": 60_000.0,
        "computed_at_iso": datetime.now(timezone.utc).isoformat(),
        "reason": f"test_{direction}",
    }


def _engine_with_h2(direction: str, **kwargs) -> SignalEngine:
    """SignalEngine avec trend_gate_fn qui retourne un h2 de direction donnée."""
    gate = MagicMock(return_value=_h2_snapshot(direction))
    return SignalEngine(
        max_positions=3,
        trend_gate_fn=gate,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Scénarios d'intégration (table des 7 du plan P-MTF-10)
# ---------------------------------------------------------------------------


def test_h2_long_plus_signal_long_15m_emits_trade():
    """1. H2 long + signal long 15m → trade émis."""
    df = _make_df_with_signal("long", n=200)
    eng = _engine_with_h2("long")
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    # Verify le DF a bien un signal_long sur la dernière bougie
    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    assert "open_long" in actions, f"Expected open_long, got {actions}"


def test_h2_long_plus_signal_short_15m_blocks():
    """2. H2 long + signal short 15m → bloqué (trend_gate)."""
    df = _make_df_with_signal("short", n=200)
    eng = _engine_with_h2("long")  # H2 dit long, signal dit short → block
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    assert df.iloc[-1]["signal_short"] is True or df.iloc[-1]["signal_short"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    assert "open_short" not in actions, f"Should be blocked, got {actions}"


def test_h2_short_plus_signal_long_15m_blocks():
    """3. H2 short + signal long 15m → bloqué."""
    df = _make_df_with_signal("long", n=200)
    eng = _engine_with_h2("short")
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    assert "open_long" not in actions


def test_h2_flat_blocks_any_signal():
    """4. H2 flat + n'importe quel signal → bloqué."""
    df_long = _make_df_with_signal("long", n=200)
    df_short = _make_df_with_signal("short", n=200)

    for df in (df_long, df_short):
        eng = _engine_with_h2("flat")
        eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)
        signals = eng.detect_signals(df, current_price=70_000.0,
                                      params={"atr_mult": 5.0, "tp_mult": 3.0})
        actions = [s["action"] for s in signals]
        assert "open_long" not in actions
        assert "open_short" not in actions


def test_h2_callback_exception_blocks_safe_fail():
    """5. H2 callback exception → block par sécurité."""
    df = _make_df_with_signal("long", n=200)
    bad_gate = MagicMock(side_effect=RuntimeError("h2 unavailable"))
    eng = SignalEngine(max_positions=3, trend_gate_fn=bad_gate)
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    # Trend gate exception → block long ET short (safe-fail)
    assert "open_long" not in actions
    assert "open_short" not in actions


def test_h2_aligned_but_var_gate_blocks():
    """6. H2 trend OK + VaR gate block → bloqué (gate VaR gagne)."""
    df = _make_df_with_signal("long", n=200)
    var_gate = MagicMock(return_value=(False, "var_too_high"))
    eng = SignalEngine(
        max_positions=3,
        trend_gate_fn=lambda: _h2_snapshot("long"),
        var_gate_fn=var_gate,
    )
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    assert "open_long" not in actions, "VaR gate should win even with H2 aligned"


def test_h2_aligned_but_daily_cap_atteint_flat_block():
    """7. H2 trend OK + daily cap atteint → flat all + block (cap gagne)."""
    df = _make_df_with_signal("long", n=200)
    eng = SignalEngine(
        max_positions=3,
        trend_gate_fn=lambda: _h2_snapshot("long"),
        daily_loss_soft_cap_pct=0.02,  # -2%
    )
    eng.load_state(
        positions_long=[],
        positions_short=[],
        daily_loss=0.0,
        daily_pnl_pct=-0.025,  # déjà -2.5% → cap déclenché
    )

    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    # Soft cap doit bloquer les nouvelles entrées
    assert "open_long" not in actions, f"Daily cap should block, got {actions}"


def test_no_trend_gate_legacy_behavior_unchanged():
    """Backward compat : trend_gate_fn=None → comportement legacy mono-H2."""
    df = _make_df_with_signal("long", n=200)
    eng = SignalEngine(max_positions=3, trend_gate_fn=None)  # explicit None
    eng.load_state(positions_long=[], positions_short=[], daily_loss=0.0)

    assert df.iloc[-1]["signal_long"] is True or df.iloc[-1]["signal_long"] == 1

    signals = eng.detect_signals(df, current_price=70_000.0,
                                  params={"atr_mult": 5.0, "tp_mult": 3.0})
    actions = [s["action"] for s in signals]
    # Pas de gate H2 = signal long passe (modulo autres gates)
    assert "open_long" in actions
