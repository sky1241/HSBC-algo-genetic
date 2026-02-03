#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIVE TRADER ADAPTIVE - Bot de trading avec adaptation de phase automatique

Ce module intègre:
1. LivePhaseAdapter pour la détection de phase
2. Stratégie Ichimoku avec params adaptatifs
3. Connexion Binance (testnet ou real)
4. Gestion des positions et du risque

Usage:
    python src/live_trader_adaptive.py --testnet  # Mode testnet
    python src/live_trader_adaptive.py            # Mode réel (ATTENTION!)

Version: 1.0
Date: 2025-02-03
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Ajouter le root au path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_phase_adapter import LivePhaseAdapter, PhaseParams, aggregate_wfa_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "outputs" / "live_trader_adaptive.log"),
    ]
)
logger = logging.getLogger("LiveTraderAdaptive")


@dataclass
class TradeSignal:
    """Signal de trading généré par la stratégie."""
    timestamp: datetime
    direction: str  # "long", "short", "close", "none"
    entry_price: float
    stop_loss: float
    take_profit: float
    phase: str
    confidence: float
    params_used: Dict[str, Any]


@dataclass
class Position:
    """Position ouverte."""
    entry_time: datetime
    direction: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    phase_at_entry: str


class IchimokuCalculator:
    """Calcule les indicateurs Ichimoku avec params dynamiques."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        tenkan: int,
        kijun: int,
        senkou_b: int,
        shift: int,
    ) -> pd.DataFrame:
        """
        Calcule les lignes Ichimoku.

        Returns:
            DataFrame avec colonnes: tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(tenkan).max()
        tenkan_low = low.rolling(tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(kijun).max()
        kijun_low = low.rolling(kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(shift)

        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(senkou_b).max()
        senkou_b_low = low.rolling(senkou_b).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(shift)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-shift)

        result = df.copy()
        result["tenkan_sen"] = tenkan_sen
        result["kijun_sen"] = kijun_sen
        result["senkou_a"] = senkou_a
        result["senkou_b"] = senkou_span_b
        result["chikou"] = chikou

        return result

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr


class AdaptiveStrategy:
    """
    Stratégie Ichimoku adaptive basée sur les phases HMM.

    Génère des signaux long/short basés sur:
    - Croisements Tenkan/Kijun
    - Position par rapport au nuage
    - Params adaptés à la phase actuelle
    """

    def __init__(self, phase_adapter: LivePhaseAdapter):
        self.adapter = phase_adapter
        self.ichimoku = IchimokuCalculator()

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        """
        Génère un signal de trading basé sur les données actuelles.

        Args:
            df: DataFrame OHLCV avec au moins 100 barres

        Returns:
            TradeSignal avec direction et niveaux
        """
        # Obtenir les params pour la phase actuelle
        params = self.adapter.get_current_params(df)

        # Calculer Ichimoku avec ces params
        ichi_df = self.ichimoku.compute(
            df,
            tenkan=params.tenkan,
            kijun=params.kijun,
            senkou_b=params.senkou_b,
            shift=params.shift,
        )

        # Calculer ATR pour SL/TP
        atr = self.ichimoku.compute_atr(df)

        # Dernières valeurs
        last = ichi_df.iloc[-1]
        prev = ichi_df.iloc[-2]
        current_price = float(last["close"])
        current_atr = float(atr.iloc[-1])

        # Déterminer la direction du signal
        direction = self._determine_direction(last, prev)

        # Calculer SL et TP
        if direction == "long":
            stop_loss = current_price - (current_atr * params.atr_mult)
            take_profit = current_price + (current_atr * params.tp_mult)
        elif direction == "short":
            stop_loss = current_price + (current_atr * params.atr_mult)
            take_profit = current_price - (current_atr * params.tp_mult)
        else:
            stop_loss = 0.0
            take_profit = 0.0

        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            phase=self.adapter.state.current_phase,
            confidence=params.confidence,
            params_used=params.to_dict(),
        )

    def _determine_direction(self, last: pd.Series, prev: pd.Series) -> str:
        """
        Détermine la direction du signal basé sur Ichimoku.

        Règles:
        - Long: Tenkan croise Kijun vers le haut + prix au-dessus du nuage
        - Short: Tenkan croise Kijun vers le bas + prix en-dessous du nuage
        """
        tenkan = float(last["tenkan_sen"])
        kijun = float(last["kijun_sen"])
        tenkan_prev = float(prev["tenkan_sen"])
        kijun_prev = float(prev["kijun_sen"])
        close = float(last["close"])
        senkou_a = float(last["senkou_a"]) if not np.isnan(last["senkou_a"]) else close
        senkou_b = float(last["senkou_b"]) if not np.isnan(last["senkou_b"]) else close

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Croisement haussier: Tenkan passe au-dessus de Kijun
        bullish_cross = tenkan > kijun and tenkan_prev <= kijun_prev

        # Croisement baissier: Tenkan passe en-dessous de Kijun
        bearish_cross = tenkan < kijun and tenkan_prev >= kijun_prev

        # Position par rapport au nuage
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom

        if bullish_cross and above_cloud:
            return "long"
        elif bearish_cross and below_cloud:
            return "short"
        else:
            return "none"


class LiveTraderAdaptive:
    """
    Bot de trading live avec adaptation de phase.

    Gère:
    - Connexion à Binance
    - Exécution des trades
    - Gestion des positions
    - Logging et monitoring
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "2h",
        testnet: bool = True,
        risk_per_trade: float = 0.01,  # 1% du capital par trade
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.testnet = testnet
        self.risk_per_trade = risk_per_trade

        # Adapter et stratégie
        self.adapter = LivePhaseAdapter(k=5)
        self.strategy = AdaptiveStrategy(self.adapter)

        # Position actuelle
        self.position: Optional[Position] = None

        # Exchange (initialisé dans connect())
        self.exchange = None

        # Stats
        self.stats = {
            "trades_total": 0,
            "trades_won": 0,
            "trades_lost": 0,
            "total_pnl": 0.0,
            "phase_transitions": 0,
        }

    def connect(self) -> bool:
        """Connecte à Binance."""
        try:
            import ccxt

            if self.testnet:
                self.exchange = ccxt.binance({
                    "apiKey": os.environ.get("BINANCE_TESTNET_API_KEY", ""),
                    "secret": os.environ.get("BINANCE_TESTNET_SECRET", ""),
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                self.exchange.set_sandbox_mode(True)
                logger.info("Connected to Binance TESTNET")
            else:
                self.exchange = ccxt.binance({
                    "apiKey": os.environ.get("BINANCE_API_KEY", ""),
                    "secret": os.environ.get("BINANCE_SECRET", ""),
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                logger.warning("Connected to Binance REAL - LIVE TRADING!")

            # Test connection
            self.exchange.fetch_balance()
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def load_params(self, params_path: str | Path) -> bool:
        """Charge les paramètres appris."""
        return self.adapter.load_learned_params(params_path)

    def fetch_ohlcv(self, limit: int = 150) -> pd.DataFrame:
        """Récupère les données OHLCV."""
        if self.exchange is None:
            raise RuntimeError("Not connected to exchange")

        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            limit=limit,
        )

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        return df

    def get_balance(self) -> float:
        """Récupère le solde USDT disponible."""
        if self.exchange is None:
            return 0.0

        balance = self.exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0.0))

    def calculate_position_size(self, entry: float, stop_loss: float) -> float:
        """
        Calcule la taille de position basée sur le risque.

        Risque = risk_per_trade * capital
        Size = Risque / |entry - stop_loss|
        """
        capital = self.get_balance()
        risk_amount = capital * self.risk_per_trade
        risk_per_unit = abs(entry - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        size = risk_amount / risk_per_unit

        # Arrondir à la précision du symbole
        return round(size, 4)

    def execute_trade(self, signal: TradeSignal) -> bool:
        """Exécute un trade basé sur le signal."""
        if self.exchange is None:
            logger.error("Not connected to exchange")
            return False

        if signal.direction == "none":
            return False

        # Fermer position existante si direction opposée
        if self.position is not None:
            if (self.position.direction == "long" and signal.direction == "short") or \
               (self.position.direction == "short" and signal.direction == "long"):
                self.close_position("signal_reversal")

        # Ouvrir nouvelle position
        if self.position is None and signal.direction in ("long", "short"):
            size = self.calculate_position_size(signal.entry_price, signal.stop_loss)

            if size <= 0:
                logger.warning("Position size too small, skipping")
                return False

            try:
                side = "buy" if signal.direction == "long" else "sell"

                # Order market
                order = self.exchange.create_market_order(
                    self.symbol,
                    side,
                    size,
                )

                # Créer la position
                self.position = Position(
                    entry_time=datetime.now(timezone.utc),
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    size=size,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    phase_at_entry=signal.phase,
                )

                logger.info(
                    f"OPENED {signal.direction.upper()} | "
                    f"Size: {size} | Entry: {signal.entry_price:.2f} | "
                    f"SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f} | "
                    f"Phase: {signal.phase}"
                )

                self.stats["trades_total"] += 1
                return True

            except Exception as e:
                logger.error(f"Order failed: {e}")
                return False

        return False

    def close_position(self, reason: str = "manual") -> Optional[float]:
        """Ferme la position actuelle."""
        if self.position is None or self.exchange is None:
            return None

        try:
            side = "sell" if self.position.direction == "long" else "buy"

            order = self.exchange.create_market_order(
                self.symbol,
                side,
                self.position.size,
            )

            # Calculer PnL
            exit_price = float(order.get("average", order.get("price", 0)))
            if self.position.direction == "long":
                pnl = (exit_price - self.position.entry_price) * self.position.size
            else:
                pnl = (self.position.entry_price - exit_price) * self.position.size

            logger.info(
                f"CLOSED {self.position.direction.upper()} | "
                f"PnL: {pnl:+.2f} USDT | Reason: {reason}"
            )

            # Stats
            self.stats["total_pnl"] += pnl
            if pnl > 0:
                self.stats["trades_won"] += 1
            else:
                self.stats["trades_lost"] += 1

            self.position = None
            return pnl

        except Exception as e:
            logger.error(f"Close order failed: {e}")
            return None

    def check_stop_loss_take_profit(self, current_price: float) -> None:
        """Vérifie et exécute SL/TP si atteint."""
        if self.position is None:
            return

        if self.position.direction == "long":
            if current_price <= self.position.stop_loss:
                self.close_position("stop_loss")
            elif current_price >= self.position.take_profit:
                self.close_position("take_profit")

        elif self.position.direction == "short":
            if current_price >= self.position.stop_loss:
                self.close_position("stop_loss")
            elif current_price <= self.position.take_profit:
                self.close_position("take_profit")

    def run_once(self) -> Dict[str, Any]:
        """
        Exécute un cycle de trading.

        Returns:
            Dict avec le statut du cycle
        """
        try:
            # Récupérer données
            df = self.fetch_ohlcv(limit=150)
            current_price = float(df["close"].iloc[-1])

            # Vérifier SL/TP
            self.check_stop_loss_take_profit(current_price)

            # Générer signal
            signal = self.strategy.generate_signal(df)

            # Exécuter si signal
            if signal.direction != "none":
                self.execute_trade(signal)

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": current_price,
                "phase": self.adapter.state.current_phase,
                "signal": signal.direction,
                "position": self.position.direction if self.position else None,
                "stats": self.stats.copy(),
            }

        except Exception as e:
            logger.error(f"Run cycle error: {e}")
            return {"error": str(e)}

    def run_loop(self, interval_seconds: int = 7200) -> None:
        """
        Boucle principale de trading.

        Args:
            interval_seconds: Intervalle entre checks (défaut: 2h = 7200s)
        """
        logger.info(f"Starting trading loop (interval: {interval_seconds}s)")
        logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        logger.info(f"Risk per trade: {self.risk_per_trade:.1%}")

        while True:
            try:
                status = self.run_once()
                logger.info(f"Cycle: {json.dumps(status, default=str)}")

                # Sauvegarder état
                self.adapter.export_state(ROOT / "outputs" / "adapter_state.json")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Stopping trading loop...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(60)  # Attendre 1 min avant retry


def main():
    parser = argparse.ArgumentParser(description="Live Trader Adaptive")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="2h", help="Timeframe")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (0.01 = 1%)")
    parser.add_argument("--params", type=str, help="Path to learned params JSON")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("LIVE TRADER ADAPTIVE")
    print("=" * 60)
    print(f"Mode:      {'TESTNET' if args.testnet else 'REAL (ATTENTION!)'}")
    print(f"Symbol:    {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Risk:      {args.risk:.1%}")
    print("=" * 60)

    # Créer le trader
    trader = LiveTraderAdaptive(
        symbol=args.symbol,
        timeframe=args.timeframe,
        testnet=args.testnet,
        risk_per_trade=args.risk,
    )

    # Charger les params appris
    if args.params:
        params_path = Path(args.params)
    else:
        params_path = ROOT / "outputs" / "wfa_phase_k5" / "aggregated_params.json"

    if params_path.exists():
        trader.load_params(params_path)
        print(f"Loaded params from: {params_path}")
    else:
        print(f"WARNING: No learned params at {params_path}")
        print("Using default Ichimoku params.")
        print("\nTo generate params:")
        print("  1. Run: .\\scripts\\launch_30_seeds_k5.ps1")
        print("  2. Aggregate results")

    # Connecter
    if not trader.connect():
        print("Failed to connect to exchange!")
        return 1

    # Lancer
    if args.once:
        status = trader.run_once()
        print(json.dumps(status, indent=2, default=str))
    else:
        trader.run_loop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
