#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data Fetcher: récupère OHLCV depuis Binance via bibliothèque officielle Binance."""
try:
    from binance.spot import Spot
    from binance.um_futures import UMFutures  # Pour Futures
except ImportError:
    # Si binance-connector n'est pas installé, utiliser ccxt comme fallback
    print("⚠️ binance-connector non installé. Installer avec: pip install binance-connector")
    raise

import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class DataFetcherBinanceOfficial:
    """Récupère données OHLCV depuis Binance via bibliothèque officielle."""
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "2h", use_futures: bool = True):
        """
        Args:
            symbol: paire de trading (ex: BTCUSDT - pas de slash avec lib officielle)
            timeframe: granularité (ex: 2h, 1h, 1d)
            use_futures: True pour Futures, False pour Spot
        """
        self.symbol = symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
        self.timeframe = timeframe
        self.use_futures = use_futures
        
        # Configuration Binance
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        # Initialiser client Binance
        if use_futures:
            if testnet:
                base_url = "https://testnet.binancefuture.com"
            else:
                base_url = "https://fapi.binance.com"  # Futures live
            
            self.client = UMFutures(
                key=api_key,
                secret=api_secret,
                base_url=base_url
            )
        else:
            if testnet:
                base_url = "https://testnet.binance.vision"
            else:
                base_url = "https://api.binance.com"  # Spot live
            
            self.client = Spot(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url
            )
    
    def _convert_timeframe(self, tf: str) -> str:
        """Convertit timeframe CCXT vers format Binance."""
        mapping = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
            "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
        }
        return mapping.get(tf, "1h")
    
    def get_ohlcv(self, limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère bougies OHLCV.
        
        Args:
            limit: nombre de bougies à récupérer
            since: timestamp ms (optionnel)
        
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        interval = self._convert_timeframe(self.timeframe)
        
        # Paramètres pour klines
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max 1000
        }
        
        if since:
            params["startTime"] = since
        
        # Appel API
        if self.use_futures:
            klines = self.client.klines(**params)
        else:
            klines = self.client.klines(**params)
        
        # Convertir en DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
        )
        
        # Convertir types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_current_price(self) -> float:
        """Récupère le prix actuel (ticker)."""
        if self.use_futures:
            ticker = self.client.ticker_price(symbol=self.symbol)
        else:
            ticker = self.client.ticker_price(symbol=self.symbol)
        
        return float(ticker['price'])
    
    def get_account_balance(self) -> float:
        """Récupère le solde USDT disponible."""
        if self.use_futures:
            account = self.client.account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
        else:
            account = self.client.account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
        
        return 0.0


if __name__ == "__main__":
    # Test rapide (nécessite clés API dans .env)
    fetcher = DataFetcherBinanceOfficial(symbol="BTCUSDT", timeframe="2h", use_futures=True)
    df = fetcher.get_ohlcv(limit=100)
    print(f"Récupéré {len(df)} bougies H2")
    print(df.tail())
    print(f"\nPrix actuel: {fetcher.get_current_price()}")

