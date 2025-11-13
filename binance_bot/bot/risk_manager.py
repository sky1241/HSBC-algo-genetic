#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk Manager: gère stop global, position sizing, levier."""
from typing import Dict


class RiskManager:
    """Vérifie contraintes de risque (stop global, position sizing, levier)."""
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        stop_global_pct: float = 0.50,
        position_size_pct: float = 0.01,
        max_leverage: float = 10.0
    ):
        """
        Args:
            initial_capital: capital initial en USDT
            stop_global_pct: seuil equity global (0.50 = stop à 50%)
            position_size_pct: taille position en % du capital (0.01 = 1%)
            max_leverage: levier maximum autorisé
        """
        self.initial_capital = initial_capital
        self.stop_global_threshold = initial_capital * stop_global_pct
        self.position_size_pct = position_size_pct
        self.max_leverage = max_leverage
    
    def check_global_stop(self, current_equity_usdt: float) -> bool:
        """
        Vérifie si stop global est atteint.
        
        Returns:
            True si stop atteint (doit fermer tout et arrêter bot)
        """
        return current_equity_usdt <= self.stop_global_threshold
    
    def calculate_position_size(self, current_equity_usdt: float, price: float) -> float:
        """
        Calcule taille position en BTC.
        
        Args:
            current_equity_usdt: equity actuelle en USDT
            price: prix BTC actuel
        
        Returns:
            qty BTC (ex: 0.01 BTC)
        """
        position_value = current_equity_usdt * self.position_size_pct
        qty = position_value / price
        return round(qty, 3)  # Arrondi Binance (3 décimales pour BTC/USDT)
    
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

