#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ichimoku Engine: calcule indicateurs Ichimoku + ATR (copie fidèle du backtest)."""
import pandas as pd
import numpy as np


def calculate_true_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcule ATR (Average True Range) exact comme dans le backtest.
    
    Args:
        df: DataFrame avec colonnes high, low, close
        period: période pour rolling mean (défaut 14)
    
    Returns:
        Series ATR
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int, shift: int) -> pd.DataFrame:
    """
    Calcule indicateurs Ichimoku (copie exacte du backtest).
    
    Args:
        df: DataFrame OHLCV
        tenkan, kijun, senkou_b, shift: paramètres Ichimoku
    
    Returns:
        DataFrame enrichi avec: tenkan_sen, kijun_sen, senkou_a, senkou_b, cloud_top, cloud_bottom
    """
    data = df.copy()
    
    # Tenkan-sen (conversion line)
    data['tenkan_sen'] = (
        data['high'].rolling(window=tenkan).max() +
        data['low'].rolling(window=tenkan).min()
    ) / 2.0
    
    # Kijun-sen (base line)
    data['kijun_sen'] = (
        data['high'].rolling(window=kijun).max() +
        data['low'].rolling(window=kijun).min()
    ) / 2.0
    
    # Senkou Span A (leading span A) - décalé de 'shift' vers le futur
    senkou_a = (data['tenkan_sen'] + data['kijun_sen']) / 2.0
    data['senkou_a'] = senkou_a.shift(shift)
    
    # Senkou Span B (leading span B) - décalé de 'shift' vers le futur
    senkou_b_val = (
        data['high'].rolling(window=senkou_b).max() +
        data['low'].rolling(window=senkou_b).min()
    ) / 2.0
    data['senkou_b'] = senkou_b_val.shift(shift)
    
    # Nuage (cloud)
    data['cloud_top'] = data[['senkou_a', 'senkou_b']].max(axis=1)
    data['cloud_bottom'] = data[['senkou_a', 'senkou_b']].min(axis=1)
    
    # ATR pour stops/TP
    data['ATR'] = calculate_true_atr(data, period=14)
    
    # Signaux (croisements Tenkan/Kijun)
    data['bull_cross'] = (
        (data['tenkan_sen'] > data['kijun_sen']) &
        (data['tenkan_sen'].shift(1) <= data['kijun_sen'].shift(1))
    )
    data['bear_cross'] = (
        (data['tenkan_sen'] < data['kijun_sen']) &
        (data['tenkan_sen'].shift(1) >= data['kijun_sen'].shift(1))
    )
    
    # Signal LONG: bull cross + close au-dessus du nuage
    data['signal_long'] = data['bull_cross'] & (data['close'] > data['cloud_top'])
    
    # Signal SHORT: bear cross + close en-dessous du nuage
    data['signal_short'] = data['bear_cross'] & (data['close'] < data['cloud_bottom'])
    
    return data


if __name__ == "__main__":
    # Test avec données factices
    dates = pd.date_range('2025-01-01', periods=200, freq='2h')
    df_test = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(200) * 1000,
        'high': 51000 + np.random.randn(200) * 1000,
        'low': 49000 + np.random.randn(200) * 1000,
        'close': 50000 + np.random.randn(200) * 1000,
        'volume': 100 + np.random.rand(200) * 50
    })
    
    result = calculate_ichimoku(df_test, tenkan=27, kijun=100, senkou_b=180, shift=93)
    print(f"Ichimoku calculé: {len(result)} lignes")
    print(f"Signaux LONG: {result['signal_long'].sum()}")
    print(f"Signaux SHORT: {result['signal_short'].sum()}")
    print(f"\nDernières lignes:")
    print(result[['close', 'tenkan_sen', 'kijun_sen', 'ATR', 'signal_long', 'signal_short']].tail())

