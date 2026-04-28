"""P4 — Tests HAR-RV (Corsi 2009) régime de volatilité + branchement signal_engine.

Vérifie:
  1. realized_volatility match formule sqrt(sum(r²)) sur fenêtre fixe
  2. fit_har_rv retourne 4 coefficients finis sur série synthétique
  3. predict_rv strictement positif (jamais log négatif après exp)
  4. classify_regime utilise bien quantiles 30% / 70% (low/mid/high)
  5. Sur returns IID Gaussien, predict_rv converge vers mean RV (sanity Corsi)
  6. Sur returns avec autocorrélation positive en RV, β coefs > 0 (persistance)
  7. Branchement: regime_gate_fn="low" → signal_engine bloque entrées
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

from src.har_rv import (
    realized_volatility,
    fit_har_rv,
    predict_rv,
    classify_regime,
)
from services.signal_engine import SignalEngine


# ---------------------------------------------------------------------------
# Tests obligatoires
# ---------------------------------------------------------------------------


def test_rv_calculation_matches_formula():
    """22 returns connus → RV(22) = sqrt(sum(r²))."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, size=100))
    rv = realized_volatility(returns, window=22)
    # Manuel: dernière valeur RV = sqrt(sum(r[78:100]²))
    last_22 = returns.iloc[78:100].to_numpy()
    expected = float(np.sqrt(np.sum(last_22 ** 2)))
    assert abs(rv.iloc[-1] - expected) < 1e-9


def test_har_fit_returns_finite_params():
    """fit_har_rv retourne 4 coefs finis (β0, β_h, β_d, β_w) sur train assez long."""
    np.random.seed(123)
    returns = pd.Series(np.random.normal(0, 0.005, size=2000))
    params = fit_har_rv(returns)
    assert params["n_obs"] > 100
    for k in ("beta_0", "beta_h", "beta_d", "beta_w"):
        assert k in params
        assert np.isfinite(params[k])


def test_har_predict_strictly_positive():
    """predict_rv retourne valeur > 0 (jamais log négatif après exp)."""
    np.random.seed(7)
    returns = pd.Series(np.random.normal(0, 0.01, size=2000))
    params = fit_har_rv(returns)
    rv_h = float(realized_volatility(returns, 1).iloc[-1])
    rv_d = float(realized_volatility(returns, 5).iloc[-1])
    rv_w = float(realized_volatility(returns, 22).iloc[-1])
    pred = predict_rv(params, rv_h, rv_d, rv_w)
    assert pred > 0
    assert np.isfinite(pred)


def test_classify_regime_quantiles():
    """Vérifie quantiles 30%/70% : valeur < q30 → "low", > q70 → "high"."""
    # historical_rv: distribution uniforme 0.01..1.01 sur 1000 points
    historical = pd.Series(np.linspace(0.01, 1.01, 1000))
    q30 = historical.quantile(0.30)  # ≈ 0.31
    q70 = historical.quantile(0.70)  # ≈ 0.71

    assert classify_regime(0.05, historical) == "low"   # < q30
    assert classify_regime(0.50, historical) == "mid"   # entre
    assert classify_regime(0.99, historical) == "high"  # > q70

    # Boundaries strictes
    assert classify_regime(q30, historical) == "mid"   # == q30 n'est pas < q30
    assert classify_regime(q70, historical) == "mid"   # == q70 n'est pas > q70


def test_har_iid_converges_to_mean():
    """Sur série IID N(0,σ), predict_rv converge vers la mean RV historique.

    Sanity check Corsi 2009 : sur process sans persistance, le HAR collapse
    vers une simple constante = E[RV].
    """
    np.random.seed(2025)
    sigma_per_bar = 0.005
    returns = pd.Series(np.random.normal(0, sigma_per_bar, size=5000))
    params = fit_har_rv(returns)
    rv_h = float(realized_volatility(returns, 1).iloc[-1])
    rv_d = float(realized_volatility(returns, 5).iloc[-1])
    rv_w = float(realized_volatility(returns, 22).iloc[-1])
    pred = predict_rv(params, rv_h, rv_d, rv_w)
    mean_rv_h = float(realized_volatility(returns, 1).dropna().mean())
    # Tolérance large car estimation OLS sur série IID = noise dominant
    relative_error = abs(pred - mean_rv_h) / max(mean_rv_h, 1e-9)
    assert relative_error < 0.50, f"pred={pred} vs mean_rv_h={mean_rv_h}"


def test_har_persistence_captured():
    """Sur série avec autocorrélation positive en variance (volatility clustering),
    au moins UN des betas doit être > 0 (capture de la persistance)."""
    np.random.seed(99)
    n = 3000
    # Volatility clustering: σ_t = base + amp × sin(t/100) puis returns ~ N(0,σ_t)
    t = np.arange(n)
    sigmas = 0.005 + 0.003 * np.abs(np.sin(t / 100.0))  # > 0 toujours
    returns = pd.Series(np.random.normal(0, sigmas))
    params = fit_har_rv(returns)
    # Au moins un coef β positif (capture du clustering)
    positive_coefs = [params[k] > 0 for k in ("beta_h", "beta_d", "beta_w")]
    assert any(positive_coefs), f"persistence not captured: betas={params}"


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_signal_engine_regime_gate_low_blocks_entries():
    """regime_gate_fn="low" → signal_engine ne génère pas de open_long."""
    se = SignalEngine(regime_gate_fn=lambda: "low")
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    open_signals = [s for s in signals if s["action"] == "open_long"]
    assert len(open_signals) == 0


def test_signal_engine_regime_gate_mid_allows_entries():
    """regime_gate_fn="mid" → entries autorisées."""
    se = SignalEngine(regime_gate_fn=lambda: "mid")
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    assert any(s["action"] == "open_long" for s in signals)


def test_signal_engine_regime_gate_high_allows_entries():
    """regime_gate_fn="high" → entries autorisées (vol haute = trends potentiels)."""
    se = SignalEngine(regime_gate_fn=lambda: "high")
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    assert any(s["action"] == "open_long" for s in signals)


def test_signal_engine_regime_callback_exception_safe():
    """Exception dans regime_gate_fn → fallback no block (entries autorisées)."""
    def _raise():
        raise RuntimeError("simulated")
    se = SignalEngine(regime_gate_fn=_raise)
    df = pd.DataFrame({
        "close": [50000.0],
        "ATR": [100.0],
        "signal_long": [True],
        "signal_short": [False],
    })
    signals = se.detect_signals(df, params={"atr_mult": 10.0, "tp_mult": 20.0}, current_price=50000.0)
    # Exception → safe fallback → entry autorisée
    assert any(s["action"] == "open_long" for s in signals)


def test_har_fit_returns_nan_when_train_too_short():
    """Train < 30 obs → params NaN, n_obs=0 (pas de crash)."""
    short_returns = pd.Series(np.random.normal(0, 0.01, size=10))
    params = fit_har_rv(short_returns)
    assert params["n_obs"] == 0
    for k in ("beta_0", "beta_h", "beta_d", "beta_w"):
        assert np.isnan(params[k])
