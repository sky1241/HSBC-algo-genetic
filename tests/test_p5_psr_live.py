"""P5 — Tests Probabilistic Sharpe Ratio (Bailey-LdP 2012, Eq. 9).

Vérifie:
  1. Sur returns Gaussiens (g3=0, g4=3), formule = Φ((SR-SR*)·sqrt(n-1))
  2. Skewness négative réduit PSR (queue gauche → moins d'edge)
  3. Kurtosis élevée (fat tails) réduit PSR
  4. min_track_record_length retourne valeur finie sur cas typique
  5. PSR baisse monotonement quand returns se dégradent
  6. PSR(SR_benchmark = SR_hat) = 0.5 exactement
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.psr_live import (
    classify_psr_alert,
    compute_psr,
    min_track_record_length,
)


# ---------------------------------------------------------------------------
# Tests obligatoires
# ---------------------------------------------------------------------------


def test_psr_normal_returns_matches_simple_formula():
    """Sur returns Gaussiens g3=0, g4=3 → PSR = Φ((SR_hat-SR*)·sqrt(n-1)).

    Tolérance large car skew/kurt empiriques diffèrent légèrement de (0, 3).
    """
    np.random.seed(42)
    n = 1000
    mu, sigma = 0.001, 0.005
    returns = pd.Series(np.random.normal(mu, sigma, size=n))
    sr_hat = float(returns.mean() / returns.std(ddof=1))

    psr_actual = compute_psr(returns, sr_benchmark=0.0)

    # Formule simplifiée Gaussienne (g3=0, g4=3) :
    # PSR = Φ(SR_hat × sqrt(n-1) / sqrt(1 - 0 × SR + (3-1)/4 × SR²))
    #     = Φ(SR_hat × sqrt(n-1) / sqrt(1 + SR²/2))
    var_factor = 1.0 + (sr_hat ** 2) / 2.0
    z_simple = sr_hat * math.sqrt(n - 1) / math.sqrt(var_factor)
    psr_simple = float(stats.norm.cdf(z_simple))

    # Tolérance 5% car skew/kurt empiriques != exact 0/3
    assert abs(psr_actual - psr_simple) < 0.05


def test_psr_negative_skew_reduces_psr():
    """Skewness négative (mêmes returns, biais ajouté) → PSR < cas symétrique."""
    np.random.seed(11)
    n = 500
    base = np.random.normal(0.001, 0.005, size=n)
    # Inject skew négatif : remplacer les top 5% par grosses pertes
    skewed = base.copy()
    top_idx = np.argsort(skewed)[-25:]
    skewed[top_idx] = -np.abs(skewed[top_idx]) * 3  # invert sign + amplify

    psr_normal = compute_psr(pd.Series(base), sr_benchmark=0.0)
    psr_skewed = compute_psr(pd.Series(skewed), sr_benchmark=0.0)
    assert psr_skewed < psr_normal


def test_psr_high_kurtosis_reduces_psr():
    """Kurtosis élevée (fat tails) à mu/sigma équivalent → PSR plus prudente.

    Implementation: returns base puis ajout outliers ±5σ qui boostent kurt
    sans déplacer la mean.
    """
    np.random.seed(22)
    n = 1000
    base = np.random.normal(0.001, 0.005, size=n)
    fat = base.copy()
    # Injection ±5σ symétrique pour ne pas changer mean/std significativement
    fat[::100] = +5 * 0.005
    fat[1::100] = -5 * 0.005

    psr_normal = compute_psr(pd.Series(base), sr_benchmark=0.0)
    psr_fat = compute_psr(pd.Series(fat), sr_benchmark=0.0)
    # Avec kurt plus élevée et mean similaire, var_factor augmente → PSR baisse
    assert psr_fat <= psr_normal + 1e-6


def test_min_trl_finite_for_typical_values():
    """SR_hat=0.1 (annualisé ~ Sharpe 1.6), SR*=0.05, g3=0, g4=3, α=5%.

    MinTRL doit être un entier fini, ordre de grandeur raisonnable (1000-2000).
    """
    trl = min_track_record_length(sr_hat=0.1, sr_benchmark=0.05, g3=0.0, g4=3.0, alpha=0.05)
    assert 100 < trl < 100000
    assert isinstance(trl, int)


def test_psr_decreases_with_drawdown():
    """Returns qui se dégradent au fil du temps → PSR rolling baisse."""
    np.random.seed(33)
    n = 200
    # Première moitié performante, deuxième mauvaise
    good = np.random.normal(0.002, 0.005, size=n // 2)
    bad = np.random.normal(-0.003, 0.005, size=n // 2)

    psr_good = compute_psr(pd.Series(good), sr_benchmark=0.0)
    psr_full = compute_psr(pd.Series(np.concatenate([good, bad])), sr_benchmark=0.0)
    assert psr_full < psr_good


def test_psr_at_benchmark_is_05():
    """PSR(SR_benchmark = SR_hat) = exactement 0.5 (Φ(0) = 0.5).

    Sanity check : la formule Bailey-LdP collapse à 0.5 quand on compare
    le SR observé à lui-même.
    """
    np.random.seed(7)
    returns = pd.Series(np.random.normal(0.001, 0.01, size=300))
    sr_hat = float(returns.mean() / returns.std(ddof=1))
    psr = compute_psr(returns, sr_benchmark=sr_hat)
    assert abs(psr - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Tests complémentaires
# ---------------------------------------------------------------------------


def test_psr_too_few_obs_returns_05():
    """n < 2 → PSR = 0.5 (indéterminé, safe)."""
    assert compute_psr(pd.Series([0.001]), sr_benchmark=0.0) == 0.5
    assert compute_psr(pd.Series([], dtype=float), sr_benchmark=0.0) == 0.5


def test_psr_zero_sigma_returns_05():
    """σ = 0 (returns constants) → PSR = 0.5."""
    assert compute_psr(pd.Series([0.001] * 50), sr_benchmark=0.0) == 0.5


def test_min_trl_returns_huge_when_sr_below_benchmark():
    """SR_hat <= SR_benchmark → MinTRL = 10**9 (jamais atteint avec ce sr_hat)."""
    assert min_track_record_length(sr_hat=0.05, sr_benchmark=0.10) == 10 ** 9
    assert min_track_record_length(sr_hat=0.05, sr_benchmark=0.05) == 10 ** 9


def test_classify_psr_alert_thresholds():
    """Seuils opérationnels: 0.5 ok, 0.3 warn, 0.1 alpha_decay, 0 kill_system."""
    assert classify_psr_alert(0.99) == "ok"
    assert classify_psr_alert(0.50) == "ok"
    assert classify_psr_alert(0.49) == "warn"
    assert classify_psr_alert(0.30) == "warn"
    assert classify_psr_alert(0.29) == "alpha_decay"
    assert classify_psr_alert(0.10) == "alpha_decay"
    assert classify_psr_alert(0.09) == "kill_system"
    assert classify_psr_alert(0.0) == "kill_system"
    # NaN -> ok safe
    assert classify_psr_alert(float("nan")) == "ok"
