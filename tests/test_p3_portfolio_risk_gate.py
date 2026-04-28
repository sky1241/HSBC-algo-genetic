"""P3 — Tests portfolio_var_95 + can_enter_new_position + branchement signal_engine.

Vérifie:
  1. VaR vide = 0 avec portefeuille vide
  2. VaR augmente avec le nombre de positions corrélées
  3. VaR plus basse avec positions anticorrélées (diversification)
  4. Gate bloque au-dessus du threshold
  5. Gate autorise sous le threshold
  6. Gate retourne raison explicite si bloqué
  7. VaR utilise returns_history 30j (1h) en historical simulation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binance_bot"))

from src.portfolio_risk_gate import (
    portfolio_var_95,
    can_enter_new_position,
)
from services.signal_engine import SignalEngine


# Fixtures communes
def _corr_btc_eth(rho: float = 0.85) -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, rho], [rho, 1.0]],
        index=["BTC/USDT", "ETH/USDT"],
        columns=["BTC/USDT", "ETH/USDT"],
    )


def _vols_btc_eth() -> dict:
    return {"BTC/USDT": 0.60, "ETH/USDT": 0.75}


# ---------------------------------------------------------------------------
# Tests obligatoires
# ---------------------------------------------------------------------------


def test_var_zero_with_empty_portfolio():
    """Portefeuille vide → VaR = 0."""
    var = portfolio_var_95(
        positions={},
        rolling_correlations=_corr_btc_eth(),
        rolling_volatilities=_vols_btc_eth(),
    )
    assert var == 0.0


def test_var_increases_with_position_count():
    """2 positions corrélées long > 1 position long (en VaR)."""
    var_1 = portfolio_var_95(
        positions={"BTC/USDT": {"side": "long", "notional_pct": 0.05}},
        rolling_correlations=_corr_btc_eth(),
        rolling_volatilities=_vols_btc_eth(),
    )
    var_2 = portfolio_var_95(
        positions={
            "BTC/USDT": {"side": "long", "notional_pct": 0.05},
            "ETH/USDT": {"side": "long", "notional_pct": 0.05},
        },
        rolling_correlations=_corr_btc_eth(),
        rolling_volatilities=_vols_btc_eth(),
    )
    assert var_2 > var_1, f"var_2={var_2} should be > var_1={var_1} (corrélations amplifient)"


def test_var_lower_with_anticorrelated_assets():
    """Long+Short BTC/ETH avec corr 0.85 → VaR plus basse que long+long."""
    var_long_long = portfolio_var_95(
        positions={
            "BTC/USDT": {"side": "long", "notional_pct": 0.05},
            "ETH/USDT": {"side": "long", "notional_pct": 0.05},
        },
        rolling_correlations=_corr_btc_eth(),
        rolling_volatilities=_vols_btc_eth(),
    )
    var_long_short = portfolio_var_95(
        positions={
            "BTC/USDT": {"side": "long", "notional_pct": 0.05},
            "ETH/USDT": {"side": "short", "notional_pct": 0.05},
        },
        rolling_correlations=_corr_btc_eth(),
        rolling_volatilities=_vols_btc_eth(),
    )
    assert var_long_short < var_long_long


def test_gate_blocks_above_threshold():
    """Position projetée qui pousse VaR > seuil → blocked."""
    # Threshold artificiellement bas pour forcer blocage avec 2 positions BTC+ETH long w=5% each
    state = {
        "open_positions": {"BTC/USDT": {"side": "long", "notional_pct": 0.05}},
        "rolling_correlations": _corr_btc_eth(),
        "rolling_volatilities": _vols_btc_eth(),
    }
    allowed, reason = can_enter_new_position(
        symbol="ETH/USDT",
        side="long",
        notional_pct=0.05,
        current_state=state,
        threshold=0.001,  # 0.1% : très bas → forcera blocage
    )
    assert not allowed
    assert "VaR95" in reason and "threshold" in reason


def test_gate_allows_below_threshold():
    """VaR projetée < seuil → autorisé."""
    state = {
        "open_positions": {},
        "rolling_correlations": _corr_btc_eth(),
        "rolling_volatilities": _vols_btc_eth(),
    }
    allowed, reason = can_enter_new_position(
        symbol="BTC/USDT",
        side="long",
        notional_pct=0.01,  # tiny position
        current_state=state,
        threshold=0.08,  # 8%
    )
    assert allowed
    assert reason == ""


def test_gate_returns_reason_when_blocked():
    """Quand bloqué, le reason contient les chiffres exacts (VaR + threshold)."""
    state = {
        "open_positions": {
            "BTC/USDT": {"side": "long", "notional_pct": 0.10},
            "ETH/USDT": {"side": "long", "notional_pct": 0.10},
        },
        "rolling_correlations": _corr_btc_eth(),
        "rolling_volatilities": _vols_btc_eth(),
    }
    allowed, reason = can_enter_new_position(
        symbol="BTC/USDT",
        side="long",
        notional_pct=0.05,
        current_state=state,
        threshold=0.001,
    )
    assert not allowed
    assert "%" in reason  # contient des pourcentages
    # Format: "VaR95 projected X.XXX% > threshold X.XX%"
    assert "projected" in reason.lower()


def test_var_uses_30d_returns_history():
    """historical_simulation : returns_history multi-symbol → VaR cohérente."""
    np.random.seed(42)
    # 30 jours × 24h = 720 bars 1h
    n_bars = 720
    btc_returns = np.random.normal(0, 0.005, size=n_bars)  # ~0.5% sigma 1h
    eth_returns = np.random.normal(0, 0.006, size=n_bars)
    # Légère corrélation injectée
    eth_returns = 0.5 * btc_returns + 0.5 * eth_returns
    history = pd.DataFrame({"BTC/USDT": btc_returns, "ETH/USDT": eth_returns})

    var = portfolio_var_95(
        positions={
            "BTC/USDT": {"side": "long", "notional_pct": 0.05},
            "ETH/USDT": {"side": "long", "notional_pct": 0.05},
        },
        returns_history=history,
        # Pas besoin de corr/vol fournis car returns_history a priorité
    )
    assert var > 0
    assert np.isfinite(var)
    # Sanity: VaR doit être < 100% (cas dégénéré sinon)
    assert var < 1.0


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_var_zero_with_no_info():
    """Pas d'historique ni corr/vol → VaR = 0 (info insuffisante, safe)."""
    var = portfolio_var_95(
        positions={"BTC/USDT": {"side": "long", "notional_pct": 0.05}},
        returns_history=None,
        rolling_correlations=None,
        rolling_volatilities=None,
    )
    assert var == 0.0


def test_signal_engine_var_gate_blocks_open_long():
    """Branchement: var_gate_fn refuse → pas d'open_long émis."""
    se = SignalEngine(
        var_gate_fn=lambda side, sized: (False, "test block"),
    )
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    assert len(open_signals) == 0  # gate a bloqué


def test_signal_engine_var_gate_allows_open_long():
    """var_gate_fn autorise → open_long émis normalement."""
    se = SignalEngine(
        var_gate_fn=lambda side, sized: (True, ""),
    )
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    assert len(open_signals) == 1


def test_signal_engine_var_gate_exception_safe_fallback():
    """Si var_gate_fn raise → autorisation par défaut (safe fallback)."""
    def _raise(side, sized):
        raise RuntimeError("simulated")
    se = SignalEngine(var_gate_fn=_raise)
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    # Exception → fallback autorise → open_long émis
    assert any(s["action"] == "open_long" for s in signals)
