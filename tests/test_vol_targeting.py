"""Tests pour vol targeting + Kelly fractionnaire — B16."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.vol_targeting import kelly_fraction, realized_volatility, vol_targeted_size


# ============================================================
# realized_volatility
# ============================================================

def test_realized_vol_constant_returns_zero():
    """Returns constants ⇒ std=0 ⇒ vol=0 (à epsilon flottant près)."""
    rs = np.array([0.01] * 50)
    assert realized_volatility(rs, periods_per_year=252) == pytest.approx(0.0, abs=1e-10)


def test_realized_vol_normal_returns_matches_theoretical():
    """Returns ~N(0, 0.01²) sur 252 périodes ⇒ vol annualisée ≈ 0.01 * sqrt(252) ≈ 0.158."""
    rng = np.random.default_rng(42)
    rs = rng.normal(0, 0.01, size=2000)
    vol = realized_volatility(rs, periods_per_year=252, window=2000)
    assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_realized_vol_uses_only_window():
    """Le window limite le calcul aux N derniers points."""
    rs = np.concatenate([np.zeros(100), np.random.default_rng(1).normal(0, 0.05, 20)])
    vol_full = realized_volatility(rs, window=120)
    vol_recent = realized_volatility(rs, window=20)
    # vol récente est plus haute que vol full (incluant zeros)
    assert vol_recent > vol_full


def test_realized_vol_handles_nan():
    rs = np.array([0.01, np.nan, 0.02, np.nan, 0.015])
    v = realized_volatility(rs, window=10)
    assert not np.isnan(v)


def test_realized_vol_returns_nan_when_too_short():
    assert np.isnan(realized_volatility(np.array([]), window=10))
    assert np.isnan(realized_volatility(np.array([0.01]), window=10))


# ============================================================
# kelly_fraction
# ============================================================

def test_kelly_zero_when_var_zero():
    assert kelly_fraction(0.001, 0.0) == 0.0


def test_kelly_full_size():
    """f_full = mean / var. fraction=1 ⇒ retourne f_full."""
    f = kelly_fraction(0.001, 0.0001, fraction=1.0)
    assert f == pytest.approx(10.0)


def test_kelly_half():
    f_full = kelly_fraction(0.001, 0.0001, fraction=1.0)
    f_half = kelly_fraction(0.001, 0.0001, fraction=0.5)
    assert f_half == pytest.approx(f_full * 0.5)


def test_kelly_negative_when_mean_negative():
    """Mean négatif ⇒ Kelly négatif (short / pas de position)."""
    assert kelly_fraction(-0.001, 0.0001, fraction=0.5) < 0


# ============================================================
# vol_targeted_size
# ============================================================

def test_vol_targeted_returns_zero_when_inputs_zero():
    assert vol_targeted_size(0, 78000, 0.5, 0.15, 0.20) == 0.0
    assert vol_targeted_size(5000, 0, 0.5, 0.15, 0.20) == 0.0
    assert vol_targeted_size(5000, 78000, 0.5, 0.15, 0) == 0.0


def test_vol_targeted_reduces_when_realized_above_target():
    """Si vol réalisée > target, on réduit le sizing."""
    qty_calm = vol_targeted_size(5000, 78000, kelly_size=1.0,
                                  target_vol_annual=0.15, realized_vol_annual=0.10)
    qty_volatile = vol_targeted_size(5000, 78000, kelly_size=1.0,
                                      target_vol_annual=0.15, realized_vol_annual=0.40)
    assert qty_volatile < qty_calm


def test_vol_targeted_increases_when_realized_below_target():
    """Vol réalisée < target ⇒ on augmente (capé par max_leverage)."""
    qty = vol_targeted_size(5000, 78000, kelly_size=1.0,
                             target_vol_annual=0.15, realized_vol_annual=0.05,
                             max_leverage=3.0)
    # vol_scaling = 0.15/0.05 = 3.0 — exactement le cap
    # fraction = min(1.0, 3.0, 3.0) = 1.0 (kelly limite)
    # qty = 5000 * 1.0 / 78000 ≈ 0.0641
    assert qty == pytest.approx(5000 / 78000, rel=0.01)


def test_vol_targeted_caps_at_max_leverage():
    """Vol scaling énorme ne doit pas exploser au-delà du cap."""
    qty = vol_targeted_size(5000, 78000, kelly_size=10.0,
                             target_vol_annual=0.15, realized_vol_annual=0.01,
                             max_leverage=3.0)
    # vol_scaling = 15, kelly = 10, mais cap = 3
    # qty = 5000 * 3 / 78000 ≈ 0.192
    assert qty == pytest.approx(5000 * 3 / 78000, rel=0.01)


def test_vol_targeted_picks_min_of_kelly_and_vol():
    """Kelly plus prudent que vol_scaling ⇒ kelly l'emporte."""
    qty = vol_targeted_size(5000, 78000, kelly_size=0.1,
                             target_vol_annual=0.15, realized_vol_annual=0.10)
    # vol_scaling = 1.5, kelly = 0.1, min = 0.1
    # qty = 5000 * 0.1 / 78000 ≈ 0.00641
    assert qty == pytest.approx(500 / 78000, rel=0.01)
