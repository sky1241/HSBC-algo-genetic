"""R6 — Tests migration `arch` lib + per-symbol assignment + audit log helper."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.garch import (
    DEFAULT_GARCH_MODEL,
    HAS_ARCH,
    SYMBOL_GARCH_MODEL,
    fit_egarch,
    fit_for_symbol,
    fit_tgarch,
    forecast_egarch,
    forecast_tgarch,
    garch_regime_label,
)


# ---------------------------------------------------------------------------
# arch lib disponible
# ---------------------------------------------------------------------------


def test_arch_library_available():
    """`arch` lib doit être installé (Kevin Sheppard, github.com/bashtage/arch)."""
    assert HAS_ARCH is True


# ---------------------------------------------------------------------------
# Per-symbol assignment
# ---------------------------------------------------------------------------


def test_symbol_garch_model_assignment_matches_spec():
    """Spec R6 : BTC=TGARCH, ETH=EGARCH, SOL=TGARCH."""
    assert SYMBOL_GARCH_MODEL["BTCUSDT"] == "TGARCH"
    assert SYMBOL_GARCH_MODEL["ETHUSDT"] == "EGARCH"
    assert SYMBOL_GARCH_MODEL["SOLUSDT"] == "TGARCH"
    assert DEFAULT_GARCH_MODEL == "TGARCH"


def test_fit_for_symbol_picks_tgarch_for_btc():
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 0.01, 1000)
    fit = fit_for_symbol("BTCUSDT", eps)
    assert fit["model"] == "TGARCH"


def test_fit_for_symbol_picks_egarch_for_eth():
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 0.01, 1000)
    fit = fit_for_symbol("ETHUSDT", eps)
    assert fit["model"] == "EGARCH"


def test_fit_for_symbol_handles_slash_format():
    """ETH/USDT devrait normaliser en ETHUSDT et choisir EGARCH."""
    rng = np.random.default_rng(2)
    eps = rng.normal(0, 0.01, 1000)
    fit = fit_for_symbol("ETH/USDT", eps)
    assert fit["model"] == "EGARCH"


def test_fit_for_symbol_unknown_falls_back_to_default():
    rng = np.random.default_rng(3)
    eps = rng.normal(0, 0.01, 1000)
    fit = fit_for_symbol("DOGEUSDT", eps)
    assert fit["model"] == DEFAULT_GARCH_MODEL  # TGARCH


# ---------------------------------------------------------------------------
# Stockage du fit pour forecast
# ---------------------------------------------------------------------------


def test_fit_includes_arch_fit_object():
    """Dict retourné contient `_arch_fit` pour permettre forecast multi-step."""
    rng = np.random.default_rng(7)
    eps = rng.normal(0, 0.01, 1000)
    fit = fit_tgarch(eps)
    assert "_arch_fit" in fit
    assert fit["_arch_fit"] is not None
    assert "_scale" in fit
    assert fit["_scale"] == 100.0


def test_fit_short_series_returns_empty_no_arch_fit():
    """< 50 obs → empty result avec _arch_fit None."""
    fit = fit_tgarch(pd.Series([0.01] * 30))
    assert fit["converged"] is False
    assert fit["_arch_fit"] is None


# ---------------------------------------------------------------------------
# garch_regime_label (branchement vs HAR-RV)
# ---------------------------------------------------------------------------


def test_garch_regime_label_low():
    # ratio = 0.5/1.0 = 0.5 < 0.7
    assert garch_regime_label(0.5, 1.0) == "low"


def test_garch_regime_label_mid():
    assert garch_regime_label(1.0, 1.0) == "mid"
    assert garch_regime_label(1.4, 1.0) == "mid"


def test_garch_regime_label_high():
    assert garch_regime_label(2.0, 1.0) == "high"


def test_garch_regime_label_safe_defaults():
    assert garch_regime_label(0.5, 0.0) == "mid"  # baseline=0 → safe
    assert garch_regime_label(float("nan"), 1.0) == "mid"
