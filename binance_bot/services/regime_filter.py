"""Régime filter — bloque les entrées Ichimoku quand le marché est en range.

Hypothèse ALPHA-1 (audit quant 2026-04-26): trader Ichimoku K3 uniquement
quand `efficiency_ratio > min_er` ET `adx > min_adx`. Sinon flat.
Par défaut DÉSACTIVÉ — activer dans bot_settings.yaml après validation
empirique par ALPHA-2 (backtest WFA + Hansen SPA).

Module léger autonome (pas de dépendance src/) pour rester dans le scope bot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)


def _wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False).mean()


def adx_value(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Renvoie la dernière valeur d'ADX (Wilder)."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn
    tr = _true_range(high, low, close)
    atr = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return float(_wilder(dx, period).iloc[-1])


def efficiency_ratio_value(close: pd.Series, period: int = 30) -> float:
    """Renvoie la dernière valeur d'Efficiency Ratio (Kaufman)."""
    if len(close) <= period:
        return float("nan")
    direction = abs(float(close.iloc[-1]) - float(close.iloc[-period - 1]))
    volatility = float(close.diff().abs().iloc[-period:].sum())
    if volatility <= 0:
        return float("nan")
    return min(1.0, max(0.0, direction / volatility))


@dataclass(slots=True)
class RegimeFilterConfig:
    enabled: bool = False
    min_er: float = 0.30
    min_adx: float = 22.0
    er_period: int = 30
    adx_period: int = 14


def is_trend_regime(df: pd.DataFrame, config: RegimeFilterConfig) -> bool:
    """Retourne True si on peut prendre des positions (régime tendance).

    Si désactivé → toujours True (passe-tout).
    Si données insuffisantes → fallback True (pas bloquer par sécurité).

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close' (chronologique).
        config: RegimeFilterConfig.
    """
    if not config.enabled:
        return True
    if len(df) < max(config.er_period + 1, config.adx_period * 2 + 1):
        return True
    er = efficiency_ratio_value(df['close'], config.er_period)
    a = adx_value(df['high'], df['low'], df['close'], config.adx_period)
    if not (np.isfinite(er) and np.isfinite(a)):
        return True  # données dégradées → ne pas bloquer
    return er >= config.min_er and a >= config.min_adx


def filter_entry_signals_inplace(df_ichimoku: pd.DataFrame, config: RegimeFilterConfig) -> dict:
    """Si on est en range, neutralise signal_long/signal_short sur la DERNIÈRE bar.

    Garde les sorties (TP/SL/trailing) intactes — on veut juste bloquer les entrées.

    Returns:
        dict avec status ('trend'|'range'|'disabled'|'insufficient_data'),
        er, adx, et n_signals_neutralized.
    """
    info = {
        "status": "disabled",
        "er": None,
        "adx": None,
        "n_signals_neutralized": 0,
    }
    if not config.enabled:
        return info
    if len(df_ichimoku) < max(config.er_period + 1, config.adx_period * 2 + 1):
        info["status"] = "insufficient_data"
        return info

    er = efficiency_ratio_value(df_ichimoku['close'], config.er_period)
    a = adx_value(df_ichimoku['high'], df_ichimoku['low'], df_ichimoku['close'], config.adx_period)
    info["er"] = er
    info["adx"] = a

    if not (np.isfinite(er) and np.isfinite(a)):
        info["status"] = "insufficient_data"
        return info

    if er >= config.min_er and a >= config.min_adx:
        info["status"] = "trend"
        return info

    # Range: neutraliser entrées sur la dernière bar
    info["status"] = "range"
    n_neut = 0
    for col in ("signal_long", "signal_short"):
        if col in df_ichimoku.columns:
            if bool(df_ichimoku.iloc[-1][col]):
                df_ichimoku.iloc[-1, df_ichimoku.columns.get_loc(col)] = False
                n_neut += 1
    info["n_signals_neutralized"] = n_neut
    return info


__all__ = [
    "RegimeFilterConfig",
    "adx_value",
    "efficiency_ratio_value",
    "is_trend_regime",
    "filter_entry_signals_inplace",
]
