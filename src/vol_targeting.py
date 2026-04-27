"""Vol-targeted Kelly fractionnaire — B16 (NICE-TO-HAVE).

Adapte le sizing en fonction de la volatilité réalisée pour cibler une vol
annualisée constante. Plus stable que Kelly fixe sur crypto.

Références:
- Man Group, "The Impact of Volatility Targeting" (2018)
- Concretum Group, "Position Sizing in Trend Following: Vol Targeting vs Vol Parity"
- Kelly Criterion classique : f* = mean / variance pour returns log
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def realized_volatility(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    window: int = 20,
) -> float:
    """Vol annualisée des `window` dernières returns.

    Args:
        returns: série de returns par période.
        periods_per_year: 252 (daily), 365*12 (H2 crypto), etc.
        window: nb de périodes à utiliser. Si len(returns) < window, prend tout.

    Returns: vol annualisée (sigma_annual).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    arr = arr[-window:] if arr.size > window else arr
    if arr.size < 2:
        return float("nan")
    daily_std = float(np.std(arr, ddof=1))
    return daily_std * float(np.sqrt(periods_per_year))


def kelly_fraction(
    mean_return_per_period: float,
    var_return_per_period: float,
    fraction: float = 0.5,
) -> float:
    """Kelly fractionnaire (default Half-Kelly).

    f* (full Kelly) = mean/var pour returns log normaux.
    Fraction recommandée: 0.25-0.5 pour limiter le drawdown au prix d'un peu de growth.

    Returns: poids de capital à allouer (peut être > 1 ou < 0).
    """
    if var_return_per_period <= 0:
        return 0.0
    f_full = mean_return_per_period / var_return_per_period
    return float(fraction * f_full)


def vol_targeted_size(
    equity_usdt: float,
    price: float,
    kelly_size: float,
    target_vol_annual: float,
    realized_vol_annual: float,
    max_leverage: float = 3.0,
) -> float:
    """Combine Kelly et vol targeting.

    Le sizing final = min(kelly_size, target_vol / realized_vol) * equity / price.
    Plafonné par max_leverage en notional.

    Args:
        equity_usdt: equity actuelle en USDT.
        price: prix de l'actif.
        kelly_size: fraction Kelly (e.g. 0.25 = 25% capital).
        target_vol_annual: vol cible (e.g. 0.15 = 15%).
        realized_vol_annual: vol réalisée récente.
        max_leverage: cap final en termes de notional/equity.

    Returns: qty (en unités de l'actif). 0 si vol indéfinie.

    Comportement:
    - Si realized_vol > target_vol → on RÉDUIT (vol scaling < 1).
    - Si realized_vol < target_vol → on AUGMENTE (vol scaling > 1, capé par leverage).
    - Le min entre kelly et vol-scaled garde le plus prudent.
    """
    if equity_usdt <= 0 or price <= 0 or realized_vol_annual <= 0 or np.isnan(realized_vol_annual):
        return 0.0
    vol_scaling = target_vol_annual / realized_vol_annual
    fraction = min(abs(kelly_size), vol_scaling)
    fraction = max(0.0, min(fraction, max_leverage))
    notional = equity_usdt * fraction
    return notional / price


__all__ = [
    "realized_volatility",
    "kelly_fraction",
    "vol_targeted_size",
]
