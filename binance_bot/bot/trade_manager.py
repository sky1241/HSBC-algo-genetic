#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trade Manager: exécute ordres sur Binance (market + stop/TP)."""
import ccxt
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Optional

import requests

from bot.binance_client import BinanceClient


class TradeManager:
    """Gère exécution ordres Binance (market entry + stop/TP)."""

    def __init__(
        self,
        exchange: ccxt.binance,
        symbol: str = "BTC/USDT",
        mode: str = "simulation",
        leverage: float = 1.0,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        algo_base_url: str = "https://testnet.binancefuture.com",
        position_mode: str = "hedge",  # "hedge" ou "oneway"
        binance_client: Optional[BinanceClient] = None,
    ):
        """
        Args:
            exchange: instance ccxt.binance
            symbol: paire de trading
            mode: "simulation" (pas d'ordres réels) ou "live"
            leverage: levier à appliquer côté Binance et utilisé pour le sizing
            api_key, api_secret: identifiants pour les algo orders (BUG-ALGO).
                Si None, lus depuis exchange.apiKey/exchange.secret.
            algo_base_url: base URL pour /fapi/v1/algoOrder (testnet ou mainnet)
            position_mode: "hedge" (positionSide LONG/SHORT explicite) ou "oneway"
                (utilise reduceOnly à la fermeture).
        """
        self.exchange = exchange
        self.symbol = symbol
        self.mode = mode  # "simulation" ou "live"
        self.leverage = float(leverage)
        self.api_key = api_key or getattr(exchange, 'apiKey', None)
        self.api_secret = api_secret or getattr(exchange, 'secret', None)
        self.algo_base_url = algo_base_url.rstrip('/')
        self.position_mode = position_mode  # "hedge" ou "oneway"
        # BinanceClient partagé pour B9 (rate limits) + B11 (time sync).
        # Si non fourni et que les credentials existent, on en crée un.
        # `auto_sync=False` pour ne pas faire d'appel réseau dans les tests
        # qui construisent un TradeManager sans patcher requests.
        if binance_client is not None:
            self.client = binance_client
        elif isinstance(self.api_key, str) and isinstance(self.api_secret, str):
            self.client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url=self.algo_base_url,
                auto_sync=False,
            )
        else:
            self.client = None

    def _side_params(self, position_side: str) -> dict:
        """Retourne les params side-specific selon le mode du compte.

        Hedge mode: positionSide=LONG/SHORT, pas de reduceOnly.
        One-way mode: pas de positionSide, reduceOnly=True à la fermeture.
        """
        if self.position_mode == "hedge":
            return {"positionSide": position_side.upper()}
        return {}  # One-way: caller ajoute reduceOnly si fermeture

    def _close_params(self, position_side: str) -> dict:
        """Params pour les ordres de fermeture (close)."""
        if self.position_mode == "hedge":
            return {"positionSide": position_side.upper()}
        return {"reduceOnly": True}

    def _sign_params(self, params: dict) -> str:
        """Construit la query signée HMAC-SHA256 (Binance signed endpoint)."""
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode('utf-8'),
            query.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()
        return query + "&signature=" + sig

    def _place_algo_order(
        self,
        side: str,
        order_type: str,
        qty: float,
        trigger_price: float,
        position_side: str = "BOTH",
    ) -> Optional[str]:
        """Place un algo order via /fapi/v1/algoOrder (HMAC manuel — BUG-ALGO).

        Utilise `self.client.signed_post` (BinanceClient) pour bénéficier du
        time-sync (B11) et du rate-limit/backoff (B9).

        Args:
            side: "BUY" ou "SELL"
            order_type: "STOP_MARKET" ou "TAKE_PROFIT_MARKET"
            qty: quantité
            trigger_price: prix de déclenchement (stopPrice côté API)
            position_side: "LONG"/"SHORT" en hedge mode, "BOTH" en one-way

        Note: Binance déduit algoType=CONDITIONAL automatiquement.
        En hedge mode, on n'envoie PAS reduceOnly (incompatible) — c'est
        positionSide qui sert de discriminant.
        """
        if not self.api_key or not self.api_secret:
            print("⚠️ api_key/api_secret manquants — impossible de placer l'algo order")
            return None
        if self.client is None:
            return None
        symbol_no_slash = self.symbol.replace("/", "")
        # Normaliser qty et triggerPrice à la précision du symbole pour éviter
        # l'erreur Binance -1111 "Precision is over the maximum" (sur ETH/SOL la
        # tickSize est plus large que pour BTC).
        try:
            qty_norm = float(self.exchange.amount_to_precision(self.symbol, qty))
            trigger_norm = float(self.exchange.price_to_precision(self.symbol, trigger_price))
        except Exception:
            qty_norm, trigger_norm = qty, trigger_price
        # API actuelle (2025-12-09+): algotype + triggerprice obligatoires (lowercase).
        # Les autres params Futures restent en camelCase.
        params = {
            "algotype": "CONDITIONAL",
            "symbol": symbol_no_slash,
            "side": side.upper(),
            "type": order_type,
            "quantity": qty_norm,
            "triggerprice": trigger_norm,
            "workingType": "MARK_PRICE",
        }
        if self.position_mode == "hedge":
            params["positionSide"] = position_side.upper()
        else:
            params["reduceOnly"] = "true"
        try:
            data = self.client.signed_post("/fapi/v1/algoOrder", params=params)
            if isinstance(data, dict) and ("algoId" in data or "algoid" in data):
                algo_id = data.get("algoId") or data.get("algoid")
                print(f"✅ algoOrder {order_type} {side} placé (algoId={algo_id})")
                return str(algo_id)
            print(f"⚠️ algoOrder error: {data}")
            return None
        except Exception as e:
            print(f"❌ algoOrder request failed: {e}")
            return None

    def fetch_open_algo_orders(self, symbol: Optional[str] = None) -> list:
        """Liste les algo orders ouverts via GET /fapi/v1/openAlgoOrders (BUG-ALGO).

        Args:
            symbol: si fourni, filtre côté serveur (ex: "BTCUSDT" ou "BTC/USDT").
                Si None, on filtre par self.symbol côté Binance (weight 1).
                Si on veut tous les symboles: passer "" (string vide → omet le param, weight 40).

        Returns:
            Liste de dicts (algoId, symbol, orderType, side, triggerPrice, ...).
            [] si aucune credentiale ou en cas d'erreur.
        """
        if not isinstance(self.api_key, str) or not isinstance(self.api_secret, str):
            return []
        if self.client is None:
            return []
        params: dict = {}
        sym = self.symbol if symbol is None else symbol
        if sym:
            params["symbol"] = sym.replace("/", "")
        try:
            data = self.client.signed_get("/fapi/v1/openAlgoOrders", params=params)
            if isinstance(data, list):
                return data
            print(f"⚠️ fetch_open_algo_orders error: {data}")
            return []
        except Exception as e:
            print(f"❌ fetch_open_algo_orders failed: {e}")
            return []

    def cancel_algo_order(self, algo_id) -> bool:
        """Annule un algo order par son algoId via DELETE /fapi/v1/algoOrder.

        Args:
            algo_id: identifiant retourné par /fapi/v1/algoOrder POST.

        Returns:
            True si Binance répond `code='200'` ou `msg='success'`, False sinon.
        """
        if not isinstance(self.api_key, str) or not isinstance(self.api_secret, str):
            return False
        if self.client is None:
            return False
        params = {"algoId": algo_id}
        try:
            data = self.client.signed_delete("/fapi/v1/algoOrder", params=params)
            if isinstance(data, dict):
                code = str(data.get("code", ""))
                msg = str(data.get("msg", "")).lower()
                if code == "200" or msg == "success":
                    return True
                print(f"⚠️ cancel_algo_order {algo_id} error: {data}")
            return False
        except Exception as e:
            print(f"❌ cancel_algo_order {algo_id} failed: {e}")
            return False

    def cancel_all_algo_orders(self) -> int:
        """Annule TOUS les algo orders ouverts sur self.symbol. Retourne le nombre annulé."""
        cancelled = 0
        for o in self.fetch_open_algo_orders():
            aid = o.get("algoId") or o.get("algoid")
            if aid is None:
                continue
            if self.cancel_algo_order(aid):
                cancelled += 1
        return cancelled

    def apply_leverage_on_exchange(self) -> bool:
        """Pousse le levier configuré côté Binance. No-op en simulation.

        Returns:
            True si l'appel set_leverage a réussi, False sinon.
        """
        if self.mode == "simulation":
            print(f"[SIMULATION] set_leverage skipped (leverage={self.leverage})")
            return False
        try:
            self.exchange.set_leverage(int(self.leverage), self.symbol)
            print(f"✅ Levier {int(self.leverage)}x appliqué sur {self.symbol}")
            return True
        except Exception as e:
            print(f"⚠️ set_leverage failed: {e}")
            return False
    
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

        # Calculer qty natif au symbole (BUG-005: tenir compte du levier dans le notional)
        position_value_usdt = capital_usdt * size_pct * self.leverage
        raw_qty = position_value_usdt / entry

        # Précision auto par symbole (BTC=1e-05, ETH=0.0001, SOL=0.001 etc.)
        try:
            qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))
        except Exception:
            qty = round(raw_qty, 3)

        if self.mode == "simulation":
            print(f"[SIMULATION] LONG {qty} {self.symbol} @ {entry} | SL={stop} | TP={tp}")
            return f"sim_long_{int(time.time())}"

        try:
            order = self.exchange.create_market_buy_order(
                self.symbol, qty, params=self._side_params("LONG")
            )
            order_id = order['id']

            # SL et TP via algoOrder (BUG-ALGO)
            self._place_algo_order("SELL", "STOP_MARKET", qty, stop, position_side="LONG")
            self._place_algo_order("SELL", "TAKE_PROFIT_MARKET", qty, tp, position_side="LONG")

            print(f"✅ LONG ouvert: {qty} {self.symbol} @ {entry} | Order ID: {order_id}")
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

        # BUG-005: tenir compte du levier dans le notional
        position_value_usdt = capital_usdt * size_pct * self.leverage
        raw_qty = position_value_usdt / entry

        try:
            qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))
        except Exception:
            qty = round(raw_qty, 3)

        if self.mode == "simulation":
            print(f"[SIMULATION] SHORT {qty} {self.symbol} @ {entry} | SL={stop} | TP={tp}")
            return f"sim_short_{int(time.time())}"

        try:
            order = self.exchange.create_market_sell_order(
                self.symbol, qty, params=self._side_params("SHORT")
            )
            order_id = order['id']

            self._place_algo_order("BUY", "STOP_MARKET", qty, stop, position_side="SHORT")
            self._place_algo_order("BUY", "TAKE_PROFIT_MARKET", qty, tp, position_side="SHORT")

            print(f"✅ SHORT ouvert: {qty} {self.symbol} @ {entry} | Order ID: {order_id}")
            return order_id

        except Exception as e:
            print(f"❌ Erreur ouverture SHORT: {e}")
            return None
    
    def _cancel_pending_orders(self) -> int:
        """Annule tous les ordres ouverts (SL/TP) sur le symbol. Retourne le nombre annulé."""
        cancelled = 0
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['id'], self.symbol)
                    cancelled += 1
                except Exception as e:
                    print(f"⚠️ Cancel order {order.get('id')} failed: {e}")
        except Exception as e:
            print(f"⚠️ fetch_open_orders failed: {e}")
        return cancelled

    def _matches_symbol(self, p: Dict) -> bool:
        """Vrai si la position concerne self.symbol.

        ccxt retourne `BTC/USDT:USDT` pour les futures linéaires; on accepte
        aussi le format plain (BTC/USDT) et le format Binance brut (BTCUSDT).
        """
        canon = self.symbol.replace("/", "")
        return (
            p.get('symbol') == self.symbol
            or p.get('symbol') == f"{self.symbol}:USDT"
            or (p.get('info') or {}).get('symbol') == canon
        )

    def _fetch_position_qty(self, side: str) -> float:
        """Récupère la qty absolue ouverte côté Binance pour le symbol. side: 'long' ou 'short'."""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for p in positions:
                if not self._matches_symbol(p):
                    continue
                contracts = float(p.get('contracts') or 0)
                if contracts == 0:
                    continue
                p_side = (p.get('side') or '').lower()
                if side == 'long' and p_side == 'long':
                    return abs(contracts)
                if side == 'short' and p_side == 'short':
                    return abs(contracts)
        except Exception as e:
            print(f"⚠️ fetch_positions failed: {e}")
        return 0.0

    def _close_long(self, signal: Dict) -> Optional[str]:
        """Ferme position LONG (cancel SL/TP + market sell reduceOnly)."""
        exit_price = signal["exit"]
        reason = signal.get("reason", "manual")

        if self.mode == "simulation":
            print(f"[SIMULATION] CLOSE LONG @ {exit_price} (raison: {reason})")
            return f"sim_close_{int(time.time())}"

        try:
            cancelled = self._cancel_pending_orders()
            cancelled_algo = self.cancel_all_algo_orders()
            qty = self._fetch_position_qty('long')
            if qty <= 0:
                print(f"⚠️ Aucune position LONG ouverte côté Binance — rien à fermer")
                return None
            order = self.exchange.create_market_sell_order(
                self.symbol,
                qty,
                params=self._close_params("LONG"),
            )
            order_id = order['id']
            print(f"✅ LONG fermé: {qty} @ ~{exit_price} (raison: {reason}, "
                  f"{cancelled} ordres + {cancelled_algo} algo annulés) | Order ID: {order_id}")
            return order_id

        except Exception as e:
            print(f"❌ Erreur fermeture LONG: {e}")
            return None

    def _close_short(self, signal: Dict) -> Optional[str]:
        """Ferme position SHORT (cancel SL/TP + market buy reduceOnly)."""
        exit_price = signal["exit"]
        reason = signal.get("reason", "manual")

        if self.mode == "simulation":
            print(f"[SIMULATION] CLOSE SHORT @ {exit_price} (raison: {reason})")
            return f"sim_close_{int(time.time())}"

        try:
            cancelled = self._cancel_pending_orders()
            cancelled_algo = self.cancel_all_algo_orders()
            qty = self._fetch_position_qty('short')
            if qty <= 0:
                print(f"⚠️ Aucune position SHORT ouverte côté Binance — rien à fermer")
                return None
            order = self.exchange.create_market_buy_order(
                self.symbol,
                qty,
                params=self._close_params("SHORT"),
            )
            order_id = order['id']
            print(f"✅ SHORT fermé: {qty} @ ~{exit_price} (raison: {reason}, "
                  f"{cancelled} ordres + {cancelled_algo} algo annulés) | Order ID: {order_id}")
            return order_id

        except Exception as e:
            print(f"❌ Erreur fermeture SHORT: {e}")
            return None


if __name__ == "__main__":
    # Test simulation TradeManager
    print("TradeManager test: créer instance avec exchange mock")
    print("Pour tester réellement, utiliser routines/intraday_runner.py")

