"""Tests pour White's Reality Check + Hansen SPA — B4 quant audit."""
from __future__ import annotations

import numpy as np
import pytest

from src.reality_check import (
    hansen_spa_test,
    stationary_bootstrap_indices,
    whites_reality_check,
)


# ============================================================
# Stationary bootstrap (Politis-Romano)
# ============================================================

def test_stationary_bootstrap_shape():
    idx = stationary_bootstrap_indices(n=100, n_bootstrap=50, block_size_mean=5.0, rng=np.random.default_rng(42))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0
    assert idx.max() < 100


def test_stationary_bootstrap_indices_in_range():
    idx = stationary_bootstrap_indices(n=20, n_bootstrap=200, block_size_mean=4.0)
    assert (idx >= 0).all()
    assert (idx < 20).all()


def test_stationary_bootstrap_deterministic_with_seed():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = stationary_bootstrap_indices(50, 10, 5.0, rng1)
    b = stationary_bootstrap_indices(50, 10, 5.0, rng2)
    assert np.array_equal(a, b)


def test_stationary_bootstrap_bad_inputs():
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(n=0, n_bootstrap=10, block_size_mean=5)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(n=10, n_bootstrap=10, block_size_mean=1.0)


# ============================================================
# White's Reality Check
# ============================================================

def test_wrc_no_edge_high_p_value():
    """Pure noise (returns ~ N(0,1)) ⇒ aucune stratégie n'a d'edge ⇒ p large."""
    rng = np.random.default_rng(42)
    excess = rng.normal(0, 0.01, size=(500, 5))
    res = whites_reality_check(excess, n_bootstrap=500, block_size_mean=10, seed=1)
    assert res.p_value > 0.10


def test_wrc_strong_edge_low_p_value():
    """Une stratégie avec drift positif clair ⇒ p faible."""
    rng = np.random.default_rng(7)
    excess = rng.normal(0, 0.01, size=(500, 5))
    excess[:, 2] += 0.005  # ajoute un drift quotidien substantiel à la stratégie 2
    res = whites_reality_check(excess, n_bootstrap=500, block_size_mean=10, seed=2)
    assert res.p_value < 0.05
    assert res.best_strategy == 2


def test_wrc_handles_1d_input():
    rng = np.random.default_rng(0)
    excess = rng.normal(0.001, 0.005, size=300)
    res = whites_reality_check(excess, n_bootstrap=300, block_size_mean=8, seed=0)
    assert 0.0 <= res.p_value <= 1.0
    assert res.best_strategy == 0
    assert res.n_strategies == 1


def test_wrc_data_snooping_detection():
    """Un panel de N stratégies pures noise dont une 'gagne' par chance ⇒ p doit
    être élevée (le RC capture le data snooping)."""
    rng = np.random.default_rng(99)
    K = 100
    excess = rng.normal(0, 0.01, size=(300, K))
    res = whites_reality_check(excess, n_bootstrap=500, block_size_mean=10, seed=99)
    # Avec 100 stratégies pures noise, le best aura un Sharpe non-nul par hasard,
    # mais le RC doit voir ça comme non-significatif.
    assert res.p_value > 0.05


def test_wrc_test_stat_in_range():
    rng = np.random.default_rng(1)
    excess = rng.normal(0.001, 0.01, size=(200, 3))
    res = whites_reality_check(excess, n_bootstrap=200, block_size_mean=10, seed=1)
    assert 0.0 <= res.p_value <= 1.0


# ============================================================
# Hansen SPA
# ============================================================

def test_spa_no_edge_high_p_consistent():
    rng = np.random.default_rng(42)
    excess = rng.normal(0, 0.01, size=(500, 5))
    res = hansen_spa_test(excess, n_bootstrap=500, block_size_mean=10, seed=1)
    assert res.p_value_consistent > 0.10


def test_spa_strong_edge_low_p_consistent():
    rng = np.random.default_rng(7)
    excess = rng.normal(0, 0.01, size=(500, 5))
    excess[:, 1] += 0.005
    res = hansen_spa_test(excess, n_bootstrap=500, block_size_mean=10, seed=2)
    assert res.p_value_consistent < 0.05
    assert res.best_strategy == 1


def test_spa_lower_p_more_conservative_than_consistent():
    """SPA_l (lower) doit être >= SPA_c >= SPA_u par construction."""
    rng = np.random.default_rng(3)
    excess = rng.normal(0, 0.01, size=(400, 8))
    excess[:, 4] += 0.002
    res = hansen_spa_test(excess, n_bootstrap=400, block_size_mean=10, seed=3)
    assert res.p_value_lower >= res.p_value_consistent - 1e-9
    assert res.p_value_consistent >= res.p_value_upper - 1e-9


def test_spa_more_powerful_than_wrc_with_poor_performers():
    """Si le panel inclut des stratégies clairement très inférieures, SPA doit donner
    une p-value <= White (plus puissant)."""
    rng = np.random.default_rng(11)
    n = 500
    K = 20
    excess = rng.normal(0, 0.01, size=(n, K))
    # Une stratégie avec un edge modéré
    excess[:, 0] += 0.0015
    # Plusieurs stratégies clairement médiocres (drift très négatif)
    for k in range(1, 15):
        excess[:, k] -= 0.005
    wrc = whites_reality_check(excess, n_bootstrap=500, block_size_mean=10, seed=11)
    spa = hansen_spa_test(excess, n_bootstrap=500, block_size_mean=10, seed=11)
    # SPA centered devrait donner une p-value <= ou approx égale à White
    # (Hansen est strictement plus puissant en présence de poor performers)
    assert spa.p_value_consistent <= wrc.p_value + 0.05


def test_spa_p_values_in_range():
    rng = np.random.default_rng(5)
    excess = rng.normal(0, 0.01, size=(200, 3))
    res = hansen_spa_test(excess, n_bootstrap=200, block_size_mean=10, seed=5)
    for p in (res.p_value_lower, res.p_value_consistent, res.p_value_upper):
        assert 0.0 <= p <= 1.0
