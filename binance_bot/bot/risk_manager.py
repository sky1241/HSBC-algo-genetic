#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk Manager: gère stop global, position sizing, levier.

P0 (2026-04-28): support portfolio-aware via paramètre `portfolio_state` qui
plafonne la taille agrégée multi-symbole (BTC+ETH+SOL co-corrélés).
"""
import os
import sys
from typing import Any, Dict, Mapping, Optional

# Permettre l'import de src.vol_targeting depuis ce module
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class RiskManager:
    """Vérifie contraintes de risque (stop global, position sizing, levier)."""

    def __init__(
        self,
        initial_capital: float = 1000.0,
        stop_global_pct: float = 0.50,
        position_size_pct: float = 0.01,
        max_leverage: float = 10.0,
        max_portfolio_risk: float = 0.06,
    ):
        """
        Args:
            initial_capital: capital initial en USDT
            stop_global_pct: seuil equity global (0.50 = stop à 50%)
            position_size_pct: taille position en % du capital (0.01 = 1%)
            max_leverage: levier maximum autorisé
            max_portfolio_risk: budget de risque agrégé (sigma_pf), 0.06 = 6%.
                Utilisé seulement si `portfolio_state` est passé à
                calculate_position_size.
        """
        self.initial_capital = initial_capital
        self.stop_global_threshold = initial_capital * stop_global_pct
        self.position_size_pct = position_size_pct
        self.max_leverage = max_leverage
        self.max_portfolio_risk = float(max_portfolio_risk)

    def check_global_stop(self, current_equity_usdt: float) -> bool:
        """
        Vérifie si stop global est atteint.

        Returns:
            True si stop atteint (doit fermer tout et arrêter bot)
        """
        return current_equity_usdt <= self.stop_global_threshold

    def calculate_position_size(
        self,
        current_equity_usdt: float,
        price: float,
        leverage: float = 1.0,
        portfolio_state: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """Calcule taille position en unités natives, levier-aware (BUG-005).

        P0 (portfolio_state fourni) : multiplie la taille par un facteur
        d'échelle [0, 1] basé sur le sigma_pf agrégé déjà utilisé. Évite
        l'over-bet quand BTC+ETH+SOL sont co-corrélés.

        Args:
            current_equity_usdt: equity actuelle en USDT
            price: prix actuel
            leverage: levier appliqué (1.0 = pas de levier)
            portfolio_state: dict optionnel (cf src/vol_targeting.py:kelly_fraction):
                {
                    "open_positions": {symbol: {"side": ..., "notional": ...}},
                    "total_capital": float,
                    "rolling_correlations": pd.DataFrame symbol×symbol,
                    "rolling_volatilities": {symbol: vol_ann_fraction}
                }

        Returns:
            qty (ex: 0.001 BTC). 0 si portfolio_state pousse au-dessus du budget.
        """
        # On valide le levier (clamp 1..max_leverage)
        eff_leverage = self.validate_leverage(leverage)
        position_value = current_equity_usdt * self.position_size_pct * eff_leverage

        # P0 portfolio-aware : applique scale factor si portfolio_state fourni
        if portfolio_state is not None:
            scale = self._portfolio_scale_factor(portfolio_state)
            position_value = position_value * scale

        qty = position_value / price
        return round(qty, 3)  # Arrondi Binance (3 décimales pour BTC/USDT)

    def _portfolio_scale_factor(self, portfolio_state: Mapping[str, Any]) -> float:
        """Retourne un facteur [0, 1] selon le risk agrégé déjà utilisé.

        scale = min(1.0, risk_budget_remaining / max_portfolio_risk)
        risk_budget_remaining = max(0, max_portfolio_risk - sigma_pf_agrégé)

        Si pas de positions ouvertes → 1.0 (pas de pénalisation).
        Si sigma_pf >= max_portfolio_risk → 0.0 (interdit nouvelle entrée).
        """
        # Import lazy pour éviter dépendance dure à pandas au module-level
        from vol_targeting import compute_aggregated_var

        open_positions = portfolio_state.get("open_positions") or {}
        if not open_positions:
            return 1.0

        total_capital = float(portfolio_state.get("total_capital", 0.0) or 0.0)
        correlations = portfolio_state.get("rolling_correlations")
        volatilities = portfolio_state.get("rolling_volatilities") or {}
        if total_capital <= 0 or correlations is None or correlations.empty:
            return 1.0  # info insuffisante, on ne pénalise pas

        sigma_pf = compute_aggregated_var(
            open_positions=open_positions,
            rolling_correlations=correlations,
            rolling_volatilities=volatilities,
            total_capital=total_capital,
        )
        if self.max_portfolio_risk <= 0:
            return 0.0
        remaining = max(0.0, self.max_portfolio_risk - sigma_pf)
        return float(min(1.0, remaining / self.max_portfolio_risk))
    
    def validate_leverage(self, leverage: float) -> float:
        """S'assure que levier ne dépasse pas max."""
        return min(max(1.0, leverage), self.max_leverage)


if __name__ == "__main__":
    # Test
    rm = RiskManager(initial_capital=1000, stop_global_pct=0.50)
    
    equity = 1200  # +20% depuis début
    print(f"Stop global atteint? {rm.check_global_stop(equity)}")
    
    qty = rm.calculate_position_size(equity, price=50000)
    print(f"Position size pour equity={equity} USDT: {qty} BTC")
    
    equity_loss = 400  # -60%
    print(f"Stop global atteint? {rm.check_global_stop(equity_loss)}")

