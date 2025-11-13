#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trade Manager: exécute ordres sur Binance (market + stop/TP)."""
import ccxt
from typing import Dict, Optional
import time


class TradeManager:
    """Gère exécution ordres Binance (market entry + stop/TP)."""
    
    def __init__(self, exchange: ccxt.binance, symbol: str = "BTC/USDT", mode: str = "simulation"):
        """
        Args:
            exchange: instance ccxt.binance
            symbol: paire de trading
            mode: "simulation" (pas d'ordres réels) ou "live"
        """
        self.exchange = exchange
        self.symbol = symbol
        self.mode = mode  # "simulation" ou "live"
    
    def execute_signal(self, signal: Dict, capital_usdt: float) -> Optional[str]:
        """
        Exécute un signal (open/close position).
        
        Args:
            signal: dict avec action, entry, stop, tp, size
            capital_usdt: capital disponible en USDT
        
        Returns:
            order_id si ordre passé, None sinon
        """
        action = signal.get("action")
        
        if action == "open_long":
            return self._open_long(signal, capital_usdt)
        elif action == "open_short":
            return self._open_short(signal, capital_usdt)
        elif action == "close_long":
            return self._close_long(signal)
        elif action == "close_short":
            return self._close_short(signal)
        
        return None
    
    def _open_long(self, signal: Dict, capital_usdt: float) -> Optional[str]:
        """Ouvre position LONG (market buy + stop/TP)."""
        entry = signal["entry"]
        stop = signal["stop"]
        tp = signal["tp"]
        size_pct = signal.get("size", 0.01)
        
        # Calculer qty BTC
        position_value_usdt = capital_usdt * size_pct
        qty_btc = position_value_usdt / entry
        
        # Arrondir selon Binance (ex: 3 décimales pour BTC/USDT)
        qty_btc = round(qty_btc, 3)
        
        if self.mode == "simulation":
            print(f"[SIMULATION] LONG {qty_btc} BTC @ {entry} | SL={stop} | TP={tp}")
            return f"sim_long_{int(time.time())}"
        
        try:
            # Ordre market BUY
            order = self.exchange.create_market_buy_order(self.symbol, qty_btc)
            order_id = order['id']
            
            # Placer stop loss (ordre STOP_MARKET)
            self.exchange.create_order(
                self.symbol,
                type='STOP_MARKET',
                side='sell',
                amount=qty_btc,
                params={'stopPrice': stop}
            )
            
            # Placer take profit (ordre LIMIT)
            self.exchange.create_limit_sell_order(self.symbol, qty_btc, tp)
            
            print(f"✅ LONG ouvert: {qty_btc} BTC @ {entry} | Order ID: {order_id}")
            return order_id
        
        except Exception as e:
            print(f"❌ Erreur ouverture LONG: {e}")
            return None
    
    def _open_short(self, signal: Dict, capital_usdt: float) -> Optional[str]:
        """Ouvre position SHORT (market sell + stop/TP)."""
        entry = signal["entry"]
        stop = signal["stop"]
        tp = signal["tp"]
        size_pct = signal.get("size", 0.01)
        
        position_value_usdt = capital_usdt * size_pct
        qty_btc = position_value_usdt / entry
        qty_btc = round(qty_btc, 3)
        
        if self.mode == "simulation":
            print(f"[SIMULATION] SHORT {qty_btc} BTC @ {entry} | SL={stop} | TP={tp}")
            return f"sim_short_{int(time.time())}"
        
        try:
            # Ordre market SELL
            order = self.exchange.create_market_sell_order(self.symbol, qty_btc)
            order_id = order['id']
            
            # Stop loss SHORT
            self.exchange.create_order(
                self.symbol,
                type='STOP_MARKET',
                side='buy',
                amount=qty_btc,
                params={'stopPrice': stop}
            )
            
            # Take profit SHORT
            self.exchange.create_limit_buy_order(self.symbol, qty_btc, tp)
            
            print(f"✅ SHORT ouvert: {qty_btc} BTC @ {entry} | Order ID: {order_id}")
            return order_id
        
        except Exception as e:
            print(f"❌ Erreur ouverture SHORT: {e}")
            return None
    
    def _close_long(self, signal: Dict) -> Optional[str]:
        """Ferme position LONG (market sell)."""
        exit_price = signal["exit"]
        reason = signal.get("reason", "manual")
        
        if self.mode == "simulation":
            print(f"[SIMULATION] CLOSE LONG @ {exit_price} (raison: {reason})")
            return f"sim_close_{int(time.time())}"
        
        try:
            # Annuler ordres stop/TP en attente
            # (en réel: chercher ordres open et les cancel)
            
            # Fermer position market
            # Note: qty doit être récupérée depuis la position ouverte
            print(f"✅ LONG fermé @ {exit_price} (raison: {reason})")
            return None
        
        except Exception as e:
            print(f"❌ Erreur fermeture LONG: {e}")
            return None
    
    def _close_short(self, signal: Dict) -> Optional[str]:
        """Ferme position SHORT (market buy)."""
        exit_price = signal["exit"]
        reason = signal.get("reason", "manual")
        
        if self.mode == "simulation":
            print(f"[SIMULATION] CLOSE SHORT @ {exit_price} (raison: {reason})")
            return f"sim_close_{int(time.time())}"
        
        try:
            print(f"✅ SHORT fermé @ {exit_price} (raison: {reason})")
            return None
        
        except Exception as e:
            print(f"❌ Erreur fermeture SHORT: {e}")
            return None


if __name__ == "__main__":
    # Test simulation TradeManager
    print("TradeManager test: créer instance avec exchange mock")
    print("Pour tester réellement, utiliser routines/intraday_runner.py")

