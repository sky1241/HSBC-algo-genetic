"""P7 — Tests VPIN (Easley-LdP-O'Hara 2012)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vpin import (
    build_volume_buckets,
    bvc_buy_share,
    classify_vpin,
    compute_vpin,
    estimate_bucket_size,
)


# ---------------------------------------------------------------------------
# bvc_buy_share
# ---------------------------------------------------------------------------


def test_bvc_buy_share_returns_05_for_zero_returns():
    """Returns nuls (price flat) → P(buy) = 0.5 (équiprobable)."""
    returns = pd.Series([0.0] * 100)
    p_buy = bvc_buy_share(returns)
    # Φ(0) = 0.5, ou neutre via fillna(0)
    assert all(0.4 <= v <= 0.6 for v in p_buy.iloc[10:])  # skip 1ers (NaN std)


def test_bvc_buy_share_high_for_positive_returns():
    """Returns positifs forts (avec variabilité) → P(buy) → 1.

    On garde une légère variabilité dans les spikes pour éviter std=0 qui
    rendrait Z indéfini.
    """
    np.random.seed(7)
    base = pd.Series(np.random.normal(0, 0.01, 200))
    spike = pd.Series(np.random.normal(0.05, 0.005, 50))  # +5% mean, vol 0.5%
    returns = pd.concat([base, spike], ignore_index=True)
    p_buy = bvc_buy_share(returns)
    # Le dernier point est ~+5σ vs vol rolling → Φ(z) très proche de 1
    assert p_buy.iloc[-1] > 0.95


# ---------------------------------------------------------------------------
# build_volume_buckets
# ---------------------------------------------------------------------------


def test_buckets_split_at_target_volume():
    """200 trades volume=1 chacun, bucket_size=50 → 4 buckets complets + 0 résidu.

    Prix constants → returns = 0 → BVC p_buy = Φ(0) = 0.5 exactement → chaque
    trade contribue exactement 1.0 en (buy+sell), pas de drift float.
    """
    n = 200
    prices = pd.Series([100.0] * n)
    volumes = pd.Series([1.0] * n)
    buckets = build_volume_buckets(prices, volumes, bucket_size_v=50.0)
    complete = [b for b in buckets if not b.get("incomplete")]
    assert len(complete) == 4  # 200 / 50 = 4
    for b in complete:
        assert b["buy_volume"] + b["sell_volume"] == pytest.approx(50.0, rel=1e-9)


def test_buckets_returns_empty_on_zero_input():
    assert build_volume_buckets(pd.Series([], dtype=float), pd.Series([], dtype=float), 50.0) == []
    # bucket_size <= 0 → empty
    assert build_volume_buckets(pd.Series([1.0]), pd.Series([1.0]), 0) == []


# ---------------------------------------------------------------------------
# compute_vpin (cas obligatoires)
# ---------------------------------------------------------------------------


def test_vpin_balanced_flow_near_zero():
    """Buckets équilibrés (50/50 buy/sell) → VPIN ≈ 0."""
    buckets = [
        {"buy_volume": 50.0, "sell_volume": 50.0, "n_trades": 100} for _ in range(50)
    ]
    vpin = compute_vpin(buckets)
    assert vpin == pytest.approx(0.0)


def test_vpin_extreme_buy_pressure_near_one():
    """Buckets 100% buy → VPIN = 1.0 (ratio |b-s|/(b+s) = 1)."""
    buckets = [
        {"buy_volume": 100.0, "sell_volume": 0.0, "n_trades": 100} for _ in range(50)
    ]
    vpin = compute_vpin(buckets)
    assert vpin == pytest.approx(1.0)


def test_vpin_bucket_size_matches_spec():
    """estimate_bucket_size(V, N) = V/N (Easley 2012 standard, default N=50)."""
    assert estimate_bucket_size(10000.0, n_buckets=50) == pytest.approx(200.0)
    assert estimate_bucket_size(10000.0, n_buckets=20) == pytest.approx(500.0)
    # daily_volume nul ou négatif → 0
    assert estimate_bucket_size(0.0) == 0.0
    assert estimate_bucket_size(-100) == 0.0


def test_vpin_window_50_buckets():
    """compute_vpin(buckets, window=50) prend les 50 derniers buckets complets.

    On crée 100 buckets : 50 vieux balanced (VPIN=0), 50 récents 100%-buy (VPIN=1).
    Le VPIN sur window=50 doit refléter les 50 récents = 1.0.
    """
    old = [{"buy_volume": 50.0, "sell_volume": 50.0, "n_trades": 100} for _ in range(50)]
    new = [{"buy_volume": 100.0, "sell_volume": 0.0, "n_trades": 100} for _ in range(50)]
    buckets = old + new
    vpin = compute_vpin(buckets, window=50)
    assert vpin == pytest.approx(1.0)


def test_vpin_monotonic_with_imbalance():
    """Plus le déséquilibre buy/sell augmente, plus le VPIN augmente."""
    def mk(buy_share):
        return [{"buy_volume": buy_share * 100, "sell_volume": (1 - buy_share) * 100, "n_trades": 50}
                for _ in range(50)]
    vpin_50_50 = compute_vpin(mk(0.5))
    vpin_60_40 = compute_vpin(mk(0.6))
    vpin_80_20 = compute_vpin(mk(0.8))
    vpin_100_0 = compute_vpin(mk(1.0))
    assert vpin_50_50 < vpin_60_40 < vpin_80_20 < vpin_100_0


def test_vpin_handles_low_volume_period():
    """Si la majorité des buckets sont incomplets, VPIN utilise les complets seulement.
    Si aucun bucket complet, retourne 0.0 (pas de signal)."""
    # Tous incomplete
    buckets = [{"buy_volume": 5.0, "sell_volume": 5.0, "n_trades": 1, "incomplete": True} for _ in range(10)]
    assert compute_vpin(buckets) == 0.0

    # Quelques complets parmi des incomplets
    buckets_mixed = (
        [{"buy_volume": 50.0, "sell_volume": 50.0, "n_trades": 100} for _ in range(5)]
        + [{"buy_volume": 5.0, "sell_volume": 5.0, "n_trades": 1, "incomplete": True} for _ in range(50)]
    )
    vpin = compute_vpin(buckets_mixed)
    assert vpin == 0.0  # buckets complets sont tous balanced (5 × ratio 0)


# ---------------------------------------------------------------------------
# classify_vpin
# ---------------------------------------------------------------------------


def test_classify_vpin_thresholds():
    assert classify_vpin(0.1) == "balanced"
    assert classify_vpin(0.5) == "normal"
    assert classify_vpin(0.75) == "toxic"
    assert classify_vpin(0.85) == "cascade"
    assert classify_vpin(float("nan")) == "balanced"


# ---------------------------------------------------------------------------
# Intégration end-to-end
# ---------------------------------------------------------------------------


def test_end_to_end_balanced_market():
    """Marché équilibré (returns IID) → VPIN proche de 0.5 (random walk noise)."""
    np.random.seed(42)
    n = 5000
    returns = np.random.normal(0, 0.001, n)
    prices = pd.Series(100.0 * np.cumprod(1 + returns))
    volumes = pd.Series(np.random.uniform(1, 5, n))
    buckets = build_volume_buckets(prices, volumes, bucket_size_v=100.0)
    vpin = compute_vpin(buckets)
    # Random walk : VPIN devrait être modéré (pas extrême)
    assert 0.0 <= vpin <= 0.8


def test_end_to_end_strong_uptrend_high_vpin():
    """Marché en uptrend fort persistant → VPIN haut (buys dominent)."""
    n = 5000
    # Prix en croissance régulière : tous returns positifs
    prices = pd.Series([100.0 * (1.0001 ** i) for i in range(n)])
    volumes = pd.Series([2.0] * n)
    buckets = build_volume_buckets(prices, volumes, bucket_size_v=100.0)
    vpin = compute_vpin(buckets)
    # Tous les returns sont positifs → BVC classifie tout comme buy → VPIN haut
    assert vpin > 0.5
