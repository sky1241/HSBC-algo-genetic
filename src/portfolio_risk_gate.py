"""P3 — Portfolio risk gate (Value-at-Risk 95% / 24h) via historical simulation.

Distinct du Kelly portfolio-aware (P0) qui MODULE la taille au prorata du
budget restant. Ici on **BLOQUE** les nouvelles entrées si la VaR projetée
du portefeuille dépasserait le seuil (default 8% du capital).

Méthodologie:
    1. Construire le vecteur de poids signed `w` (long: +, short: -)
       à partir de positions {symbol: {"side", "notional_pct"}}.
    2. Calculer le P&L scénario sur horizon 24h via :
       - Historical simulation : bootstrap stationnaire (Politis-Romano) sur
         les returns 1h des 30 derniers jours, somme glissante 24 bars.
       - OU Monte Carlo Gaussien fallback : tirages corrélés via Cholesky de
         la matrice Σ = D × C × D, avec C = corrélations rolling, D = vol/√(8760).
    3. VaR_95% = -percentile(pnl_distribution, 5).

Référence:
    - Politis & Romano (1994), "The stationary bootstrap" — block-resampling
      pour séries auto-corrélées.
    - Jorion P., "Value at Risk" 3rd ed. (2007).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


_DEFAULT_THRESHOLD = 0.08  # 8% capital
_DEFAULT_BOOTSTRAP = 1000
_DEFAULT_HORIZON_HOURS = 24


def _construct_weights(
    positions: Mapping[str, Mapping[str, Any]],
    symbols_order: list[str],
) -> np.ndarray:
    """Vecteur de poids signed (fraction du capital) dans l'ordre `symbols_order`."""
    w = np.zeros(len(symbols_order), dtype=float)
    for i, sym in enumerate(symbols_order):
        if sym not in positions:
            continue
        pos = positions[sym]
        side = str(pos.get("side", "long")).lower()
        notional_pct = float(pos.get("notional_pct", 0.0))
        sign = 1.0 if side == "long" else -1.0
        w[i] = sign * notional_pct
    return w


def _bootstrap_24h_pnl(
    returns_history: pd.DataFrame,
    weights: np.ndarray,
    horizon_hours: int,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    """Block-bootstrap des returns 1h pour générer N scénarios PnL sur `horizon_hours`.

    Hypothèse : `returns_history` colonnes alignées sur l'ordre des poids;
    rows = bars 1h. On tire `n_scenarios` blocs glissants de `horizon_hours`
    consécutifs (avec remise), somme cumulée → PnL scénario.
    """
    rng = np.random.default_rng(seed)
    arr = returns_history.to_numpy(dtype=float)
    n_bars, _ = arr.shape
    if n_bars < horizon_hours + 1:
        return np.zeros(n_scenarios, dtype=float)
    starts = rng.integers(low=0, high=n_bars - horizon_hours, size=n_scenarios)
    pnls = np.empty(n_scenarios, dtype=float)
    for k, s in enumerate(starts):
        block = arr[s:s + horizon_hours]
        # PnL portefeuille = sum_t(weights · returns_t) sur l'horizon
        pnl_per_bar = block @ weights  # shape (horizon_hours,)
        pnls[k] = float(np.sum(pnl_per_bar))
    return pnls


def _monte_carlo_24h_pnl(
    weights: np.ndarray,
    rolling_correlations: pd.DataFrame,
    rolling_volatilities: list[float],
    horizon_hours: int,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo Gaussien fallback (corr + vol fournis, pas d'historique).

    Tire des returns multivariés via Cholesky de Σ_horizon = h × Σ_1h, où
    Σ_1h = D × C × D et D = diag(sigma_annual / sqrt(periods_per_year)).
    Approximation : on suppose returns 1h ~ N(0, sigma_1h).
    """
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    # Vol annuelle → vol 1h (√(365×24) périodes/an pour crypto 24/7)
    periods_per_year = 365 * 24
    sigma_1h = np.array(rolling_volatilities, dtype=float) / np.sqrt(periods_per_year)
    C = rolling_correlations.to_numpy(dtype=float)
    # Σ_1h = D C D
    D = np.diag(sigma_1h)
    cov_1h = D @ C @ D
    # Σ_horizon = horizon × Σ_1h
    cov_h = horizon_hours * cov_1h
    # Cholesky (avec petite régularisation si numériquement non PSD)
    try:
        L = np.linalg.cholesky(cov_h + 1e-12 * np.eye(n_assets))
    except np.linalg.LinAlgError:
        return np.zeros(n_scenarios, dtype=float)
    # n_scenarios tirages multivariés
    z = rng.standard_normal(size=(n_scenarios, n_assets))
    returns_scenarios = z @ L.T  # shape (n_scenarios, n_assets)
    pnls = returns_scenarios @ weights
    return pnls


def portfolio_var_95(
    positions: Mapping[str, Mapping[str, Any]],
    returns_history: Optional[pd.DataFrame] = None,
    rolling_correlations: Optional[pd.DataFrame] = None,
    rolling_volatilities: Optional[Mapping[str, float]] = None,
    horizon_hours: int = _DEFAULT_HORIZON_HOURS,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 42,
) -> float:
    """VaR au confidence level (default 95%) sur horizon `horizon_hours`.

    Args:
        positions: {symbol: {"side": "long"|"short", "notional_pct": float}}
            où notional_pct est la fraction signed du capital.
        returns_history: DataFrame colonnes=symbols, rows=bars 1h sur 30j.
            Si fourni, historical simulation. Sinon, fallback MC Gaussien.
        rolling_correlations: DataFrame symbol×symbol Pearson (P0).
        rolling_volatilities: {symbol: vol_ann_fraction} (P0).
        horizon_hours: horizon de risque (default 24h).
        n_bootstrap: nombre de scénarios (default 1000).
        confidence: niveau (0.95).
        seed: RNG seed pour reproductibilité.

    Returns:
        VaR positive en fraction du capital (e.g. 0.05 = 5%). 0.0 si positions
        vides ou info insuffisante.
    """
    if not positions:
        return 0.0

    symbols_order = list(positions.keys())
    weights = _construct_weights(positions, symbols_order)
    if np.allclose(weights, 0):
        return 0.0

    # Path 1 : historical simulation (préférable si returns_history fourni)
    if returns_history is not None and not returns_history.empty:
        cols = [s for s in symbols_order if s in returns_history.columns]
        if len(cols) == len(symbols_order):
            history = returns_history[cols]
            pnls = _bootstrap_24h_pnl(
                history, weights,
                horizon_hours=horizon_hours,
                n_scenarios=n_bootstrap,
                seed=seed,
            )
            if pnls.size > 0:
                var = -float(np.percentile(pnls, (1.0 - confidence) * 100.0))
                return max(0.0, var)

    # Path 2 : Monte Carlo Gaussien fallback
    if rolling_correlations is not None and rolling_volatilities is not None and not rolling_correlations.empty:
        # Filtrer corr/vol sur les symboles présents
        try:
            corr_sub = rolling_correlations.loc[symbols_order, symbols_order]
        except KeyError:
            return 0.0
        vols = [float(rolling_volatilities.get(s, 0.0)) for s in symbols_order]
        if any(v <= 0 for v in vols):
            return 0.0
        pnls = _monte_carlo_24h_pnl(
            weights, corr_sub, vols,
            horizon_hours=horizon_hours,
            n_scenarios=n_bootstrap,
            seed=seed,
        )
        if pnls.size > 0:
            var = -float(np.percentile(pnls, (1.0 - confidence) * 100.0))
            return max(0.0, var)

    return 0.0  # info insuffisante


def can_enter_new_position(
    symbol: str,
    side: str,
    notional_pct: float,
    current_state: Mapping[str, Any],
    threshold: float = _DEFAULT_THRESHOLD,
    horizon_hours: int = _DEFAULT_HORIZON_HOURS,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    seed: int = 42,
) -> Tuple[bool, str]:
    """Vérifie si ouvrir une nouvelle position garde la VaR95 sous le seuil.

    Args:
        symbol: paire (e.g. "BTC/USDT").
        side: "long" ou "short".
        notional_pct: fraction du capital pour la nouvelle position (e.g. 0.01 = 1%).
        current_state: dict avec :
            "open_positions": {sym: {"side", "notional_pct"}}
            "returns_history": optional DataFrame
            "rolling_correlations": optional DataFrame (P0)
            "rolling_volatilities": optional dict (P0)
        threshold: VaR seuil (default 0.08 = 8%).

    Returns:
        (autorisé: bool, raison: str). raison vide si autorisé.
    """
    open_positions = dict(current_state.get("open_positions") or {})
    # Projection : ajouter ou agréger la nouvelle position
    if symbol in open_positions:
        existing = open_positions[symbol]
        if str(existing.get("side", "long")).lower() == side.lower():
            # Même side : agrégation notional
            new_notional = float(existing.get("notional_pct", 0.0)) + float(notional_pct)
            open_positions[symbol] = {"side": side, "notional_pct": new_notional}
        # Si opposite side : conflit (règle XOR signal_engine), on n'écrase pas
    else:
        open_positions[symbol] = {"side": side, "notional_pct": float(notional_pct)}

    var = portfolio_var_95(
        positions=open_positions,
        returns_history=current_state.get("returns_history"),
        rolling_correlations=current_state.get("rolling_correlations"),
        rolling_volatilities=current_state.get("rolling_volatilities"),
        horizon_hours=horizon_hours,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if var > threshold:
        return False, f"VaR95 projected {var*100:.3f}% > threshold {threshold*100:.2f}%"
    return True, ""


__all__ = [
    "portfolio_var_95",
    "can_enter_new_position",
]
