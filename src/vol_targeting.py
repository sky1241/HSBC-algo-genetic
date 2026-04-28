"""Vol-targeted Kelly fractionnaire — B16 + P0 portfolio-aware.

Adapte le sizing en fonction de la volatilité réalisée pour cibler une vol
annualisée constante. Plus stable que Kelly fixe sur crypto.

P0 (2026-04-28): Kelly portfolio-aware via paramètre `portfolio_state` qui
plafonne le risk agrégé multi-symbole. Évite l'over-bet quand BTC+ETH+SOL
sont co-corrélés ~0.85 en stress (cf MDPI 2025).

Références:
- Man Group, "The Impact of Volatility Targeting" (2018)
- Concretum Group, "Position Sizing in Trend Following: Vol Targeting vs Vol Parity"
- Kelly Criterion classique : f* = mean / variance pour returns log
- Roncalli T., "The Risk Parity Page" (formules portfolio variance canoniques)
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

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


def compute_aggregated_var(
    open_positions: Mapping[str, Mapping[str, Any]],
    rolling_correlations: pd.DataFrame,
    rolling_volatilities: Mapping[str, float],
    total_capital: float,
) -> float:
    """VaR-like agrégée du portefeuille en fraction du capital (1 sigma annualisé).

    Formule canonique (Markowitz / Roncalli):
        sigma_pf = sqrt(w' × Σ × w)
        Σ_ij    = corr_ij × sigma_i × sigma_j
        w_i     = signed_notional_i / total_capital  (long: +, short: -)

    Args:
        open_positions: {symbol: {"side": "long"|"short", "notional": float USDT}}.
        rolling_correlations: DataFrame carré symbol×symbol des corrélations Pearson
            (returns 1h, 30 jours rolling). Doit inclure tous les symbols ouverts.
        rolling_volatilities: {symbol: vol annualisée fraction (e.g. 0.6 = 60%)}.
        total_capital: équité totale du compte en USDT.

    Returns:
        Sigma agrégé du portefeuille en FRACTION du capital (e.g. 0.04 = 4%).
        Retourne 0.0 si pas de position ouverte ou capital invalide.
    """
    if total_capital <= 0 or not open_positions:
        return 0.0

    symbols = list(open_positions.keys())
    weights = np.zeros(len(symbols), dtype=float)
    sigmas = np.zeros(len(symbols), dtype=float)
    for i, sym in enumerate(symbols):
        pos = open_positions[sym]
        side = str(pos.get("side", "long")).lower()
        notional = float(pos.get("notional", 0.0))
        sign = 1.0 if side == "long" else -1.0
        weights[i] = sign * notional / total_capital
        sigmas[i] = float(rolling_volatilities.get(sym, 0.0))

    # Construire la matrice de covariance Σ = D × C × D où D=diag(sigma)
    cov = np.zeros((len(symbols), len(symbols)), dtype=float)
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i == j:
                cov[i, j] = sigmas[i] * sigmas[i]  # variance
            else:
                try:
                    rho = float(rolling_correlations.loc[si, sj])
                except (KeyError, ValueError):
                    rho = 0.0  # si paire absente, on ignore la corrélation
                if not np.isfinite(rho):
                    rho = 0.0
                cov[i, j] = rho * sigmas[i] * sigmas[j]

    var_pf = float(weights @ cov @ weights)
    if var_pf <= 0:
        return 0.0
    return float(np.sqrt(var_pf))


def kelly_fraction(
    mean_return_per_period: float,
    var_return_per_period: float,
    fraction: float = 0.5,
    portfolio_state: Optional[Mapping[str, Any]] = None,
    max_portfolio_risk: float = 0.06,
) -> float:
    """Kelly fractionnaire (default Half-Kelly), optionnellement portfolio-aware.

    Comportement standard (portfolio_state=None) :
        f* (full Kelly) = mean/var pour returns log normaux.
        Fraction recommandée: 0.25-0.5 pour limiter le drawdown au prix d'un peu de growth.

    Comportement portfolio-aware (P0, portfolio_state fourni) :
        On calcule le risk agrégé déjà utilisé par les positions ouvertes.
        Si proche du budget max → on pénalise le Kelly.
            kelly_new = kelly_solo × min(1.0, risk_budget_remaining / max_portfolio_risk)
        Quand toutes les positions ouvertes sont co-corrélées (BTC+ETH+SOL ~0.85
        en stress), Kelly est massivement réduit pour éviter l'over-bet.

    Args:
        mean_return_per_period: rendement moyen par période.
        var_return_per_period: variance des rendements.
        fraction: Kelly fractionnaire (0.25-0.5 typique).
        portfolio_state: dict optionnel:
            {
                "open_positions": {symbol: {"side": ..., "notional": ...}},
                "total_capital": float,
                "rolling_correlations": pd.DataFrame symbol×symbol,
                "rolling_volatilities": {symbol: vol_ann_fraction}
            }
        max_portfolio_risk: budget de risque agrégé (0.06 = 6% sigma_pf).

    Returns:
        Fraction du capital à allouer (peut être > 1 ou < 0 sans portfolio_state).
        Avec portfolio_state, plafonné par le budget restant.
    """
    if var_return_per_period <= 0:
        return 0.0
    f_full = mean_return_per_period / var_return_per_period
    kelly_solo = float(fraction * f_full)

    if portfolio_state is None:
        return kelly_solo

    # P0 portfolio-aware: réduire Kelly selon risk déjà utilisé
    open_positions = portfolio_state.get("open_positions") or {}
    total_capital = float(portfolio_state.get("total_capital", 0.0) or 0.0)
    correlations = portfolio_state.get("rolling_correlations")
    volatilities = portfolio_state.get("rolling_volatilities") or {}

    if total_capital <= 0 or correlations is None or correlations.empty or not open_positions:
        # Pas assez d'info pour calculer la VaR agrégée → on retourne Kelly solo
        return kelly_solo

    portfolio_risk_used = compute_aggregated_var(
        open_positions=open_positions,
        rolling_correlations=correlations,
        rolling_volatilities=volatilities,
        total_capital=total_capital,
    )
    risk_budget_remaining = max(0.0, max_portfolio_risk - portfolio_risk_used)
    if max_portfolio_risk <= 0:
        return 0.0
    scale = min(1.0, risk_budget_remaining / max_portfolio_risk)
    return float(kelly_solo * scale)


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
    "compute_aggregated_var",
]
