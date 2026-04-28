"""P7 — VPIN (Volume-synchronized Probability of Informed Trading).

Détecte la "toxicité" de l'order flow via la déséquilibration buy/sell
agrégée sur des buckets volume-synchronisés (= temps non uniforme, fenêtre
de volume constant).

Référence:
    Easley, D., López de Prado, M., O'Hara, M. (2012).
    "The Volume Clock: Insights into the High-Frequency Paradigm."
    Journal of Portfolio Management, 39(1), 19-29.

Validation crypto:
    Kitvanitphasu et al. (2025). "Bitcoin wild moves: Evidence from order
    flow toxicity and price jumps."

Méthodologie:
    1. Bulk Volume Classification (BVC) :
       Z = (price_t - price_{t-1}) / sigma_returns_rolling
       P(buy_aggressive) = Φ(Z)  (CDF normale)
       buy_volume = volume × Φ(Z)
       sell_volume = volume × (1 - Φ(Z))
    2. Volume buckets : on accumule trades jusqu'à atteindre `bucket_size_v`
       (= V_daily / N, default N=50). Chaque bucket stocke (buy_vol, sell_vol).
    3. VPIN sur les `window` derniers buckets (default 50) :
       VPIN = (1/n) × Σ |buy_vol_i - sell_vol_i| / (buy_vol_i + sell_vol_i)

Interprétation:
    VPIN > 0.7 : toxic, mouvement directionnel imminent
    VPIN < 0.3 (après spike) : flux absorbé, mean reversion possible
    VPIN > 0.8 + faible OBI : signal de liquidation cascade

Hyperparamètres figés (pas d'optim Optuna pour éviter snooping) :
    N = 50 (taille des buckets / nb buckets dans la fenêtre)
    Volatility lookback = 50 returns (cf Easley et al. 2012)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


_DEFAULT_N_BUCKETS = 50
_DEFAULT_VOL_LOOKBACK = 50  # returns rolling pour BVC


def bvc_buy_share(returns: pd.Series, lookback: int = _DEFAULT_VOL_LOOKBACK) -> pd.Series:
    """Bulk Volume Classification : retourne P(buy_aggressive) ∈ [0, 1] par trade.

    Args:
        returns: pd.Series de returns par trade (price_t / price_{t-1} - 1).
        lookback: fenêtre rolling pour la vol des returns.

    Returns:
        pd.Series alignée sur returns.index, valeurs dans [0, 1] (Φ(Z)).
        NaN remplacé par 0.5 (équiprobable, neutre).
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if returns.empty:
        return pd.Series(dtype=float)
    sigma = returns.rolling(window=lookback, min_periods=2).std(ddof=1)
    z = returns.divide(sigma).replace([np.inf, -np.inf], np.nan)
    p_buy = pd.Series(stats.norm.cdf(z.fillna(0.0)), index=returns.index)
    p_buy = p_buy.clip(lower=0.0, upper=1.0)
    return p_buy


def build_volume_buckets(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size_v: float,
    bvc_lookback: int = _DEFAULT_VOL_LOOKBACK,
) -> list[dict]:
    """Construit des buckets de volume constant via BVC.

    Args:
        prices: pd.Series prix de chaque trade (ou bar).
        volumes: pd.Series volume associé (même index que prices).
        bucket_size_v: target volume par bucket (= V_daily / N).
        bvc_lookback: fenêtre vol pour BVC.

    Returns:
        Liste de dicts {buy_volume, sell_volume, n_trades, ts_start_ms, ts_end_ms}.
        Le dernier bucket peut être incomplet (volume cumulé < bucket_size_v).
    """
    if prices.empty or volumes.empty:
        return []
    if bucket_size_v <= 0:
        return []

    returns = prices.astype(float).pct_change().fillna(0.0)
    p_buy = bvc_buy_share(returns, lookback=bvc_lookback)

    buy_vol_per_trade = volumes.astype(float).values * p_buy.values
    sell_vol_per_trade = volumes.astype(float).values * (1.0 - p_buy.values)

    buckets: list[dict] = []
    accum_buy = 0.0
    accum_sell = 0.0
    n_trades = 0
    ts_start = None
    ts_end = None
    has_index_dates = isinstance(prices.index, pd.DatetimeIndex)

    for i in range(len(prices)):
        if ts_start is None:
            ts_start = prices.index[i]
        accum_buy += float(buy_vol_per_trade[i])
        accum_sell += float(sell_vol_per_trade[i])
        n_trades += 1
        ts_end = prices.index[i]
        if (accum_buy + accum_sell) >= bucket_size_v:
            bucket = {
                "buy_volume": accum_buy,
                "sell_volume": accum_sell,
                "n_trades": n_trades,
            }
            if has_index_dates:
                bucket["ts_start_ms"] = int(pd.Timestamp(ts_start).timestamp() * 1000)
                bucket["ts_end_ms"] = int(pd.Timestamp(ts_end).timestamp() * 1000)
            buckets.append(bucket)
            accum_buy = 0.0
            accum_sell = 0.0
            n_trades = 0
            ts_start = None
            ts_end = None

    # Dernier bucket incomplet (optionnel, on le garde si non vide)
    if n_trades > 0:
        bucket = {
            "buy_volume": accum_buy,
            "sell_volume": accum_sell,
            "n_trades": n_trades,
            "incomplete": True,
        }
        if has_index_dates:
            bucket["ts_start_ms"] = int(pd.Timestamp(ts_start).timestamp() * 1000)
            bucket["ts_end_ms"] = int(pd.Timestamp(ts_end).timestamp() * 1000)
        buckets.append(bucket)
    return buckets


def compute_vpin(
    buckets: list[dict],
    window: int = _DEFAULT_N_BUCKETS,
) -> float:
    """VPIN sur les `window` derniers buckets COMPLETS.

    VPIN = mean( |buy - sell| / (buy + sell) ) sur window buckets.

    Args:
        buckets: liste de dicts retournée par build_volume_buckets.
        window: nombre de buckets dans la fenêtre VPIN (default 50, Easley 2012).

    Returns:
        VPIN ∈ [0, 1]. 0.0 si pas assez de buckets ou tous bidon.
    """
    if not buckets:
        return 0.0
    # On exclut les buckets incomplets (volume sous bucket_size_v)
    complete = [b for b in buckets if not b.get("incomplete", False)]
    if not complete:
        return 0.0
    recent = complete[-window:] if len(complete) >= window else complete
    if not recent:
        return 0.0
    ratios: list[float] = []
    for b in recent:
        total = float(b["buy_volume"]) + float(b["sell_volume"])
        if total <= 0:
            continue
        diff = abs(float(b["buy_volume"]) - float(b["sell_volume"]))
        ratios.append(diff / total)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def classify_vpin(vpin: float) -> str:
    """Classifie le niveau de toxicité.

    Returns:
        "balanced"  : VPIN < 0.3 (flux équilibré)
        "normal"    : 0.3 <= VPIN < 0.7
        "toxic"     : 0.7 <= VPIN < 0.8 (mouvement directionnel imminent)
        "cascade"   : VPIN >= 0.8 (signal liquidation cascade probable)
    """
    if not np.isfinite(vpin):
        return "balanced"
    if vpin < 0.3:
        return "balanced"
    if vpin < 0.7:
        return "normal"
    if vpin < 0.8:
        return "toxic"
    return "cascade"


def estimate_bucket_size(
    daily_volume_avg: float,
    n_buckets: int = _DEFAULT_N_BUCKETS,
) -> float:
    """Calcule la taille de bucket = V_daily_moyen / N (Easley 2012 standard)."""
    if daily_volume_avg <= 0:
        return 0.0
    return float(daily_volume_avg) / float(max(1, n_buckets))


__all__ = [
    "bvc_buy_share",
    "build_volume_buckets",
    "compute_vpin",
    "classify_vpin",
    "estimate_bucket_size",
]
