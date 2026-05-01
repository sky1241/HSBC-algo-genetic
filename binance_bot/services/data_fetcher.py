#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data Fetcher: récupère OHLCV depuis Binance via CCXT.

P-MTF-1 (mission multi-TF) : ajout de `get_ohlcv_multi_tf()` pour permettre
au runner d'exécution 15m de réutiliser la donnée H2 (cache TTL) sans
re-fetch toutes les 15 min.
"""
import time
import ccxt
import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# P-MTF-1 : TTL par défaut par timeframe (en secondes).
# H2 cachée 1h (resync mid-bar OK), 1h cachée 10min, 15m cachée 1min, 5m cachée 30s.
_DEFAULT_TTL_SECONDS = {
    "2h": 3600,
    "1h": 600,
    "30m": 120,
    "15m": 60,
    "5m": 30,
    "1m": 10,
}


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
    
    # ------------------------------------------------------------------
    # P-MTF-1 : multi-TF fetcher avec cache mémoire
    # ------------------------------------------------------------------

    # Cache class-level partagé entre toutes les instances pour le même
    # process Python. Clé : (symbol, timeframe) → (df, fetched_at_ts_seconds).
    _ohlcv_cache: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}

    def get_ohlcv_multi_tf(
        self,
        timeframes: list[str],
        limit: int = 300,
        use_cache: bool = True,
        cache_ttl_seconds: Optional[dict[str, int]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV pour plusieurs timeframes, avec cache TTL par TF.

        P-MTF-1 — utilisé par execution_runner_15m (qui a besoin de H2 trend
        depuis state.json, recalculé au max 1×/h via cache) et par
        h2_trend_runner (force refresh via use_cache=False ou TTL court).

        Args:
            timeframes: liste de TFs (ex: ["2h", "15m"]).
            limit: nombre de bougies à récupérer par TF.
            use_cache: si False, force re-fetch sans regarder le cache.
            cache_ttl_seconds: override des TTL (default _DEFAULT_TTL_SECONDS).
                Format : {"2h": 3600, "15m": 60, ...}.

        Returns:
            dict {tf: pd.DataFrame OHLCV} pour chaque TF demandé.
            Si un fetch échoue pour un TF, ce TF est absent du dict (pas
            d'exception levée — robuste pour orchestration multi-symbole).
        """
        ttl = {**_DEFAULT_TTL_SECONDS, **(cache_ttl_seconds or {})}
        now = time.time()
        result: dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            cache_key = (self.symbol, tf)
            cached = self._ohlcv_cache.get(cache_key) if use_cache else None
            if cached is not None:
                df_cached, fetched_at = cached
                if now - fetched_at < ttl.get(tf, 60):
                    result[tf] = df_cached
                    continue

            # Cache miss ou TTL expiré → re-fetch
            try:
                candles = self.exchange.fetch_ohlcv(
                    self.symbol, timeframe=tf, limit=limit
                )
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.sort_values("timestamp").reset_index(drop=True)
                df = self._filter_unclosed_last_candle(df, tf)
                self._ohlcv_cache[cache_key] = (df, now)
                result[tf] = df
            except Exception as e:
                # Robuste : on log et on continue sans ce TF (pas de crash
                # du runner si Binance hoquette sur un TF particulier).
                print(
                    f"⚠️ DataFetcher.get_ohlcv_multi_tf: fetch failed "
                    f"for {self.symbol} {tf}: {e}"
                )

        return result

    @classmethod
    def _clear_cache(cls) -> None:
        """Reset cache (utile pour tests)."""
        cls._ohlcv_cache.clear()

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

