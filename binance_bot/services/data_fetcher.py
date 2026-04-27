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

        # Configuration Binance USDM (futures linéaires)
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        # Note: ccxt.binanceusdm + set_sandbox_mode est la voie propre pour testnet
        # futures. ccxt.binance avec urls custom ne route pas sapi correctement.
        self.exchange = ccxt.binanceusdm({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
    
    @staticmethod
    def _filter_unclosed_last_candle(
        df: pd.DataFrame,
        timeframe: str,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Drop la dernière bougie si elle n'est pas encore clôturée.

        Une bougie ouverte au timestamp T en timeframe TF est clôturée à T+TF.
        Si T+TF > now → on la retire (signal Ichimoku basé sur bougie complète seulement).
        """
        if df.empty:
            return df
        if now_utc is None:
            now_utc = pd.Timestamp.now(tz='UTC')
        tf_delta = pd.Timedelta(timeframe)
        last_ts = df.iloc[-1]['timestamp']
        if last_ts + tf_delta > now_utc:
            return df.iloc[:-1].reset_index(drop=True)
        return df

    def get_ohlcv(self, limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère bougies OHLCV (sans la bougie en cours de formation).

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

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # BUG-004: ne jamais signaler sur une bougie non-clôturée
        df = self._filter_unclosed_last_candle(df, self.timeframe)

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

