"""Alpha strategies — comparative backtest harness for ALPHA-2 research.

Implements four signal generators on top of a Ichimoku K3 baseline:

V1 — Ichimoku K3 nu (Tenkan=21, Kijun=35, Senkou A/B shifted +44, lookback 90).
V2 — V1 + filtre régime ER>0.30 ET ADX>22 (Kaufman/Wilder).
V3 — V2 + veto Choppiness Index > 61.8 (Dreiss).
V4 — V1 + range_score < 0.4 (composite `src.range_detector.range_score`).

API : ``generate_signals(df, variant)`` retourne une ``pd.Series`` ∈ {-1, 0, +1}
représentant la position détenue à la barre t. Le P&L se calcule ensuite via
``simulate_returns(df, signal)`` qui applique les coûts B5 (`src.cost_model`).

Convention :
    * Signal calculé en t (sur infos t et antérieures) puis ``shift(1)`` →
      la position au PnL[t+1] est celle décidée en t. Pas de look-ahead.
    * Pas d'entrée si le filtre régime invalide. La position courante
      reste neutre (0) tant que le filtre est faux ; on n'ajoute pas de
      logique exit-only-when-flip pour rester comparable à V1.

Le module n'optimise rien : 4 stratégies, paramètres figés ⇒ ``n_trials=4``
pour le DSR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import cost_model
from . import range_detector as rd


Variant = Literal["V1", "V2", "V3", "V4"]
VARIANTS: tuple[Variant, ...] = ("V1", "V2", "V3", "V4")


# ---------------------------------------------------------------------------
# Ichimoku K3 baseline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IchimokuK3Params:
    tenkan: int = 21
    kijun: int = 35
    senkou_b: int = 90
    shift: int = 44


def _hh_ll(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return (hh + ll) / 2.0


def ichimoku_components(df: pd.DataFrame, p: IchimokuK3Params = IchimokuK3Params()) -> pd.DataFrame:
    """Calcule Tenkan/Kijun/Senkou A/Senkou B sur OHLC.

    NOTE — pour préserver le décalage causal, Senkou A/B sont décalés ``+shift``
    bars. Quand on lit ``senkou_a[t]`` à la barre t, sa valeur a été calculée
    à ``t - shift`` et représente "le cloud au-dessus / dessous de la barre t".
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tenkan = _hh_ll(high, low, p.tenkan)
    kijun = _hh_ll(high, low, p.kijun)
    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_b_raw = _hh_ll(high, low, p.senkou_b)
    senkou_a = senkou_a_raw.shift(p.shift)
    senkou_b = senkou_b_raw.shift(p.shift)
    out = pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
    }, index=df.index)
    return out


def ichimoku_k3_position(df: pd.DataFrame, p: IchimokuK3Params = IchimokuK3Params()) -> pd.Series:
    """Position binaire {-1, 0, +1} sous le baseline V1.

    Long si Tenkan>Kijun ET close>max(senkA,senkB).
    Short si Tenkan<Kijun ET close<min(senkA,senkB).
    Sinon flat.
    Stateless (pas de hold persistent) — la position est recomputée chaque bar.
    """
    comp = ichimoku_components(df, p)
    close = df["close"].astype(float)
    cloud_top = comp[["senkou_a", "senkou_b"]].max(axis=1)
    cloud_bot = comp[["senkou_a", "senkou_b"]].min(axis=1)

    bull = (comp["tenkan"] > comp["kijun"]) & (close > cloud_top)
    bear = (comp["tenkan"] < comp["kijun"]) & (close < cloud_bot)

    pos = pd.Series(0.0, index=df.index)
    pos[bull] = 1.0
    pos[bear] = -1.0
    return pos


# ---------------------------------------------------------------------------
# Filtres régime
# ---------------------------------------------------------------------------

def regime_filter_er_adx(
    df: pd.DataFrame,
    er_period: int = 30,
    er_thr: float = 0.30,
    adx_period: int = 14,
    adx_thr: float = 22.0,
) -> pd.Series:
    """True si ER>thr ET ADX>thr — autorise l'entrée."""
    er = rd.efficiency_ratio(df["close"], er_period)
    a = rd.adx(df["high"], df["low"], df["close"], adx_period)
    return (er > er_thr) & (a > adx_thr)


def chop_veto(df: pd.DataFrame, period: int = 14, threshold: float = 61.8) -> pd.Series:
    """True si CHOP <= seuil — c'est-à-dire **pas range** (autorise l'entrée)."""
    ci = rd.choppiness_index(df["high"], df["low"], df["close"], period)
    return ci <= threshold


def range_score_veto(df: pd.DataFrame, threshold: float = 0.4) -> pd.Series:
    """True si range_score < threshold — c'est-à-dire trend (autorise l'entrée)."""
    scored = rd.range_score(df)
    return scored["range_score"] < threshold


# ---------------------------------------------------------------------------
# Signal dispatcher
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, variant: Variant) -> pd.Series:
    """Retourne la position détenue (avant ``shift(1)``) pour la stratégie demandée."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    base = ichimoku_k3_position(df)
    if variant == "V1":
        return base.fillna(0.0)
    if variant == "V2":
        ok = regime_filter_er_adx(df).fillna(False)
        return base.where(ok, 0.0).fillna(0.0)
    if variant == "V3":
        ok = regime_filter_er_adx(df).fillna(False) & chop_veto(df).fillna(False)
        return base.where(ok, 0.0).fillna(0.0)
    if variant == "V4":
        ok = range_score_veto(df).fillna(False)
        return base.where(ok, 0.0).fillna(0.0)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Net return simulator (uses the realistic cost model B5)
# ---------------------------------------------------------------------------

def simulate_returns(
    df: pd.DataFrame,
    position: pd.Series,
    *,
    funding_rates: Optional[pd.Series] = None,
    fees: Optional[cost_model.BinanceFutureFees] = None,
    leverage: float = 1.0,
    depth_usdt: float = 50_000.0,
    is_taker: bool = True,
) -> pd.Series:
    """Convertit une position en returns nets (frais + funding + slippage).

    On ``shift(1)`` la position avant de la multiplier par les returns du
    sous-jacent — la position décidée à la barre t est en risque sur
    ``ret[t+1]``. ``apply_costs_to_returns`` reçoit la même position (déjà
    décalée) pour que ``Δposition`` soit aligné avec les changements
    réels d'exposition.
    """
    close = df["close"].astype(float)
    underlying = close.pct_change().fillna(0.0)
    pos_held = position.shift(1).fillna(0.0).astype(float)
    raw = pos_held * underlying

    # ATR pour le slippage dépendant de la vol.
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().bfill()

    return cost_model.apply_costs_to_returns(
        raw,
        position=pos_held,
        price=close,
        atr=atr,
        funding_rates=funding_rates,
        fees=fees,
        leverage=leverage,
        depth_usdt=depth_usdt,
        is_taker=is_taker,
    )


def count_trades(position: pd.Series) -> int:
    """Nombre de changements de position (≠0 ⇒ open ou flip)."""
    pos = position.fillna(0.0).astype(float)
    delta = pos.diff().fillna(pos.iloc[0] if len(pos) else 0.0)
    return int((delta.abs() > 0).sum())


__all__ = [
    "VARIANTS",
    "Variant",
    "IchimokuK3Params",
    "ichimoku_components",
    "ichimoku_k3_position",
    "regime_filter_er_adx",
    "chop_veto",
    "range_score_veto",
    "generate_signals",
    "simulate_returns",
    "count_trades",
]
