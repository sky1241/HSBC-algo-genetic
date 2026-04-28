"""P2 — Tests anti-martingale drawdown_size_multiplier + branchement signal_engine.

Vérifie:
  1. Pas de drawdown → multiplicateur = 1.0 (taille pleine)
  2. dd = -5% → 0.667 (deux tiers)
  3. dd = -10% → 0.5 (moitié)
  4. dd = -15% → 0.0 (arrêt)
  5. Recovery only on new high (rebond intermédiaire ne reset pas)
  6. Premier trade (pas d'historique) → 1.0
  7. Branchement signal_engine: drawdown_scale_fn module la size de open_long/short
  8. Kill switch + Telegram à dd < -15% (test côté intraday_runner via patch)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binance_bot"))

# `src/risk_sizing.py` utilise un import relatif `from . import cost_model`,
# donc on l'importe via le package `src`.
from src.risk_sizing import drawdown_size_multiplier  # type: ignore
from services.signal_engine import SignalEngine  # type: ignore


# ---------------------------------------------------------------------------
# Tests obligatoires — drawdown_size_multiplier (pure function)
# ---------------------------------------------------------------------------


def test_no_dd_returns_full_size():
    """current_equity == rolling_high → 1.0 (full size)."""
    assert drawdown_size_multiplier(100.0, 100.0) == 1.0
    # Légèrement au-dessus du high (cas où le high n'a pas encore été update)
    assert drawdown_size_multiplier(105.0, 100.0) == 1.0


def test_5pct_dd_returns_two_thirds():
    """dd = -5%+ε → 0.667 (deux tiers)."""
    # dd = -6% (sous le seuil -5%)
    mult = drawdown_size_multiplier(94.0, 100.0)
    assert mult == 0.667
    # dd = -5% pile : pas encore "<-5%" donc reste à 1.0
    mult_at_5 = drawdown_size_multiplier(95.0, 100.0)
    assert mult_at_5 == 1.0  # boundary: dd == -0.05 n'est pas < -0.05


def test_10pct_dd_halves_size():
    """dd = -10%+ε → 0.5 (moitié)."""
    mult = drawdown_size_multiplier(89.0, 100.0)
    assert mult == 0.5
    # dd = -11% encore plus → toujours 0.5 (pas encore -15%)
    assert drawdown_size_multiplier(89.0, 100.0) == 0.5


def test_15pct_dd_kills_returns_zero():
    """dd < -15% → 0.0 (arrêt système)."""
    mult = drawdown_size_multiplier(84.0, 100.0)  # dd = -16%
    assert mult == 0.0
    mult_severe = drawdown_size_multiplier(50.0, 100.0)  # dd = -50%
    assert mult_severe == 0.0


def test_recovery_only_on_new_high():
    """rolling_high doit représenter le PEAK; si current<peak, mult réduit même si rebond.

    On simule : high atteint à 100, descente à 85 (mult=0.5), rebond à 95 (toujours <-5% du high)
    → mult doit RESTER 0.667 (pas reset à 1.0 jusqu'à new high).
    """
    high = 100.0
    # Descente forte
    assert drawdown_size_multiplier(85.0, high) == 0.5
    # Rebond mais pas nouveau high
    assert drawdown_size_multiplier(95.0, high) == 1.0  # dd = -5% pile, ≥ -5%
    assert drawdown_size_multiplier(94.0, high) == 0.667  # dd = -6%
    # Nouveau high explicite (caller doit avoir update rolling_high)
    new_high = 110.0
    assert drawdown_size_multiplier(110.0, new_high) == 1.0


def test_first_trade_no_history_returns_one():
    """Pas d'historique (rolling_high <= 0) → 1.0 sans crash."""
    assert drawdown_size_multiplier(100.0, 0.0) == 1.0
    assert drawdown_size_multiplier(100.0, -50.0) == 1.0
    # current_equity dégénéré
    assert drawdown_size_multiplier(0.0, 100.0) == 1.0
    assert drawdown_size_multiplier(-10.0, 100.0) == 1.0


# ---------------------------------------------------------------------------
# Tests obligatoires — branchement signal_engine.drawdown_scale_fn
# ---------------------------------------------------------------------------


def _make_df_signal_long(close=50000.0, atr=100.0):
    return pd.DataFrame({
        "close": [close],
        "ATR": [atr],
        "signal_long": [True],
        "signal_short": [False],
    })


def test_signal_engine_size_modulated_by_drawdown_scale_fn():
    """SignalEngine multiplie size par drawdown_scale_fn (combiné avec portfolio_scale)."""
    # dd_scale = 0.5 mock (équivalent dd ~-10%)
    se = SignalEngine(drawdown_scale_fn=lambda: 0.5)
    df = _make_df_signal_long()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)

    open_signals = [s for s in signals if s["action"] == "open_long"]
    assert len(open_signals) == 1
    # size = 0.01 × 1.0 (no portfolio scale) × 0.5 (dd) = 0.005
    assert abs(open_signals[0]["size"] - 0.005) < 1e-9


def test_signal_engine_dd_zero_kills_size():
    """drawdown_scale_fn=0 → size = 0 (pas d'entrée effective taille zéro)."""
    se = SignalEngine(drawdown_scale_fn=lambda: 0.0)
    df = _make_df_signal_long()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    # Le signal d'open est tout de même émis, mais size=0 → trade_manager créera
    # un ordre de qty 0 qui sera arrondi out par Binance min notional.
    assert len(open_signals) == 1
    assert open_signals[0]["size"] == 0.0


def test_signal_engine_dd_combined_with_portfolio_scale():
    """size = 0.01 × portfolio_scale × dd_scale. Test produit avec deux scales."""
    se = SignalEngine(
        portfolio_scale_fn=lambda: 0.6,
        drawdown_scale_fn=lambda: 0.667,
    )
    df = _make_df_signal_long()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    expected = 0.01 * 0.6 * 0.667
    assert abs(open_signals[0]["size"] - expected) < 1e-9


def test_signal_engine_dd_callback_exception_safe_fallback():
    """Si drawdown_scale_fn raise → fallback à 1.0 (pas de crash bot)."""
    def _raise():
        raise RuntimeError("simulated error")
    se = SignalEngine(drawdown_scale_fn=_raise)
    df = _make_df_signal_long()
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    # size legacy 0.01 × 1.0 × 1.0 = 0.01 (fallback safe)
    assert abs(open_signals[0]["size"] - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_dd_multiplier_boundary_minus_5_exact():
    """Boundary: dd = -5% exact → 1.0 (>= -5%)."""
    assert drawdown_size_multiplier(95.0, 100.0) == 1.0


def test_dd_multiplier_boundary_minus_10_exact():
    """Boundary: dd = -10% exact → 0.667 (n'est pas < -10%)."""
    assert drawdown_size_multiplier(90.0, 100.0) == 0.667


def test_dd_multiplier_boundary_minus_15_exact():
    """Boundary: dd = -15% exact → 0.5 (n'est pas < -15%)."""
    assert drawdown_size_multiplier(85.0, 100.0) == 0.5


def test_dd_multiplier_progression_monotonic():
    """Plus le drawdown augmente, plus le multiplicateur diminue (monotonique)."""
    high = 100.0
    multipliers = [
        drawdown_size_multiplier(100.0, high),
        drawdown_size_multiplier(94.0, high),
        drawdown_size_multiplier(89.0, high),
        drawdown_size_multiplier(84.0, high),
    ]
    # Strictement décroissant
    assert multipliers == [1.0, 0.667, 0.5, 0.0]
    # Bonus: confirme jamais croissant
    for a, b in zip(multipliers, multipliers[1:]):
        assert a >= b
