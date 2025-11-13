#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data Fetcher: récupère OHLCV depuis Binance via CCXT."""
import ccxt
import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class DataFetcher:
    """Récupère données OHLCV depuis Binance (testnet ou live)."""
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "2h"):
        """
        Args:
            symbol: paire de trading (ex: BTC/USDT)
            timeframe: granularité (ex: 2h, 1h, 1d)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Configuration Binance
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Futures (si tu veux du levier)
        }
        
        if testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com',
                    'private': 'https://testnet.binancefuture.com',
                }
            }
        
        self.exchange = ccxt.binance(config)
    
    def get_ohlcv(self, limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère bougies OHLCV.
        
        Args:
            limit: nombre de bougies à récupérer
            since: timestamp ms (optionnel)
        
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        candles = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            limit=limit,
            since=since
        )
        
        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convertir timestamp ms -> datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_current_price(self) -> float:
        """Récupère le prix actuel (ticker)."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        return float(ticker['last'])
    
    def get_account_balance(self) -> float:
        """Récupère le solde USDT disponible."""
        balance = self.exchange.fetch_balance()
        return float(balance['USDT']['free'])


if __name__ == "__main__":
    # Test rapide (nécessite clés API dans .env)
    fetcher = DataFetcher()
    df = fetcher.get_ohlcv(limit=100)
    print(f"Récupéré {len(df)} bougies H2")
    print(df.tail())
    print(f"\nPrix actuel: {fetcher.get_current_price()}")

