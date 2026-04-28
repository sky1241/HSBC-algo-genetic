"""F-004 / INSTR-5 — Test d'intégration end-to-end Kelly portfolio_state.

Comble le trou identifié par audit CLAR-2 : `kelly_fraction(portfolio_state=...)`
n'avait aucun test d'intégration via `SignalEngine`. Le scale est calculé par
la factory `_make_portfolio_scale_fn` (intraday_runner) qui construit
`portfolio_state` à la volée puis appelle `compute_aggregated_var` (et non
`kelly_fraction` directement, mais le résultat module finalement le size
émis par `SignalEngine.detect_signals`).

Scénario test :
  1. Setup state_mgr avec 1 position BTC long (notional 6% capital).
  2. Mock corrélations rolling avec corr(BTC,ETH) = 0.85.
  3. Construire portfolio_scale_fn via factory réelle pour symbol="ETH/USDT".
  4. Appeler SignalEngine.detect_signals avec signal_long=True sur ETH.
  5. Assert : taille ETH (avec BTC ouvert) < taille ETH (sans BTC).

Si le test passe → la chaîne intraday_runner factory → SignalEngine → size
est intacte. Si fail → bug silent dans la factory ou la propagation.
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

from binance_bot.services.signal_engine import SignalEngine
from binance_bot.routines.intraday_runner import _make_portfolio_scale_fn


class _MockStateManager:
    """Mock minimal de StateManager pour tests d'intégration."""

    def __init__(self, symbols_state: dict):
        self.state = {"symbols": symbols_state}


def _make_correlations() -> pd.DataFrame:
    """Corrélations BTC/USDT × ETH/USDT × SOL/USDT avec corr BTC-ETH = 0.85."""
    return pd.DataFrame(
        [
            [1.00, 0.85, 0.70],
            [0.85, 1.00, 0.75],
            [0.70, 0.75, 1.00],
        ],
        index=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        columns=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    )


def _make_volatilities() -> dict:
    """Vol annualisée fictive — proxy stress crypto."""
    return {"BTC/USDT": 0.60, "ETH/USDT": 0.70, "SOL/USDT": 0.85}


def _make_eth_signal_df() -> pd.DataFrame:
    return pd.DataFrame({
        "close": [3000.0],
        "ATR": [50.0],
        "signal_long": [True],
        "signal_short": [False],
    })


# ---------------------------------------------------------------------------
# Scénario : factory réelle propage portfolio_state jusqu'à signal_engine
# ---------------------------------------------------------------------------


def test_portfolio_scale_fn_returns_below_one_when_correlated_position_open():
    """Etant donné BTC long avec corr(BTC,ETH)=0.85, le scale ETH doit être <1."""
    # State : BTC position ouverte, notional 6% capital
    state_btc_open = _MockStateManager({
        "BTC/USDT": {
            "positions_long": [{"id": "L1", "size": 0.06,
                                "entry": 50000.0}],
            "positions_short": [],
        },
        "ETH/USDT": {"positions_long": [], "positions_short": []},
    })

    fn = _make_portfolio_scale_fn(
        state_mgr=state_btc_open,
        current_symbol="ETH/USDT",
        total_capital=1000.0,
        corr_df=_make_correlations(),
        vol_dict=_make_volatilities(),
        max_portfolio_risk=0.06,
        current_price=3000.0,
    )
    scale = fn()
    # BTC consomme du risk → scale ETH doit être < 1
    assert 0.0 <= scale < 1.0, f"Expected scale < 1 (BTC corrélé), got {scale}"


def test_portfolio_scale_fn_returns_one_when_no_other_positions():
    """Sans positions ouvertes ailleurs → scale = 1 (full Kelly)."""
    state_empty = _MockStateManager({
        "BTC/USDT": {"positions_long": [], "positions_short": []},
        "ETH/USDT": {"positions_long": [], "positions_short": []},
    })
    fn = _make_portfolio_scale_fn(
        state_mgr=state_empty, current_symbol="ETH/USDT",
        total_capital=1000.0, corr_df=_make_correlations(),
        vol_dict=_make_volatilities(), max_portfolio_risk=0.06,
        current_price=3000.0,
    )
    assert fn() == 1.0


def test_signal_engine_eth_size_smaller_with_btc_open_than_without():
    """Test bout-en-bout : taille ETH proposée par SignalEngine.detect_signals
    est PLUS PETITE quand BTC est ouvert et corrélé que quand BTC est fermé.

    Preuve que portfolio_state est bien propagé via :
        intraday_runner._make_portfolio_scale_fn
        → callback closure capture state_mgr + corr_df + vol_dict
        → SignalEngine.detect_signals invoke portfolio_scale_fn()
        → multiplie le size final
    """
    # --- Cas 1 : BTC fermé ---
    state_empty = _MockStateManager({
        "BTC/USDT": {"positions_long": [], "positions_short": []},
        "ETH/USDT": {"positions_long": [], "positions_short": []},
    })
    fn_empty = _make_portfolio_scale_fn(
        state_mgr=state_empty, current_symbol="ETH/USDT",
        total_capital=1000.0, corr_df=_make_correlations(),
        vol_dict=_make_volatilities(), max_portfolio_risk=0.06,
        current_price=3000.0,
    )
    eng_empty = SignalEngine(portfolio_scale_fn=fn_empty)
    signals_empty = eng_empty.detect_signals(
        _make_eth_signal_df(),
        params={"atr_mult": 10.0, "tp_mult": 20.0},
        current_price=3000.0,
    )

    # --- Cas 2 : BTC long ouvert (notional 6% capital, corr 0.85 avec ETH) ---
    state_btc_open = _MockStateManager({
        "BTC/USDT": {
            "positions_long": [{"id": "L1", "size": 0.06, "entry": 50000.0}],
            "positions_short": [],
        },
        "ETH/USDT": {"positions_long": [], "positions_short": []},
    })
    fn_btc = _make_portfolio_scale_fn(
        state_mgr=state_btc_open, current_symbol="ETH/USDT",
        total_capital=1000.0, corr_df=_make_correlations(),
        vol_dict=_make_volatilities(), max_portfolio_risk=0.06,
        current_price=3000.0,
    )
    eng_btc = SignalEngine(portfolio_scale_fn=fn_btc)
    signals_btc = eng_btc.detect_signals(
        _make_eth_signal_df(),
        params={"atr_mult": 10.0, "tp_mult": 20.0},
        current_price=3000.0,
    )

    # Extract size of open_long signals (case 1 vs case 2)
    size_empty = next(
        (s["size"] for s in signals_empty if s.get("action") == "open_long"),
        None,
    )
    size_btc = next(
        (s["size"] for s in signals_btc if s.get("action") == "open_long"),
        None,
    )

    # Both should have generated a signal (Ichimoku trigger same in both)
    assert size_empty is not None, "Cas 1 (BTC fermé) doit générer signal long"
    assert size_btc is not None, "Cas 2 (BTC ouvert) doit générer signal long"
    # Avec BTC ouvert (corr 0.85), le size ETH doit être strictement plus petit
    assert size_btc < size_empty, (
        f"Portfolio gate broken : size_btc_open={size_btc:.6f} >= "
        f"size_empty={size_empty:.6f} (devrait être <)"
    )


def test_signal_engine_no_factory_returns_default_size():
    """Sans portfolio_scale_fn (None), le size par défaut (0.01) est émis."""
    eng = SignalEngine(portfolio_scale_fn=None)
    signals = eng.detect_signals(
        _make_eth_signal_df(),
        params={"atr_mult": 10.0, "tp_mult": 20.0},
        current_price=3000.0,
    )
    open_long = [s for s in signals if s.get("action") == "open_long"]
    assert len(open_long) == 1
    # Size par défaut = 0.01 (config bot 1% capital)
    assert open_long[0]["size"] == pytest.approx(0.01, rel=0.01)
