#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculer ATR médian BTC et convertir multiplicateurs en USD."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/BTC_FUSED_2h_clean.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Calculer ATR (True Range sur 14 périodes)
df['h-l'] = df['high'] - df['low']
df['h-pc'] = (df['high'] - df['close'].shift(1)).abs()
df['l-pc'] = (df['low'] - df['close'].shift(1)).abs()
df['TR'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
df['ATR'] = df['TR'].rolling(14).mean()

# Stats sur 5000 dernières barres (récent)
recent = df.tail(5000)
atr_median = recent['ATR'].median()
price_median = recent['close'].median()
atr_pct = (atr_median / price_median) * 100

print("="*70)
print("📊 ATR MÉDIAN BTC (5000 dernières barres H2)")
print("="*70)
print(f"\nATR médian: {atr_median:.2f} USD")
print(f"Prix médian: {price_median:.2f} USD")
print(f"ATR / Prix: {atr_pct:.2f}%")

print("\n" + "="*70)
print("💰 STOP-LOSS & TAKE-PROFIT en USD (selon multiplicateur)")
print("="*70)
print("\n| ATR_mult | Stop/TP (USD) | % du prix | Commentaire |")
print("|----------|---------------|-----------|-------------|")

examples = [
    (5.0, "Serré (Phase 2 sideways)"),
    (10.0, "Moyen (Phase 2 équilibré)"),
    (11.8, "K3 Phase 0 médian"),
    (15.0, "Large"),
    (19.3, "K3 Phase 1 médian (momentum)"),
    (25.0, "Très large (max Optuna)")
]

for mult, comment in examples:
    stop_usd = mult * atr_median
    pct = (stop_usd / price_median) * 100
    print(f"| {mult:8.1f} | {stop_usd:13.0f} | {pct:9.2f}% | {comment} |")

print("\n" + "="*70)
print("📈 EXEMPLE CONCRET (Prix BTC = 50,000 USD)")
print("="*70)
price_example = 50000
atr_example = price_example * (atr_pct / 100)  # Proportionnel

print(f"\nATR estimé: {atr_example:.0f} USD ({atr_pct:.2f}% du prix)")
print("\n| ATR_mult | Stop (USD) | TP (USD) | Range (%) |")
print("|----------|------------|----------|-----------|")

for mult, _ in examples:
    stop = mult * atr_example
    print(f"| {mult:8.1f} | {price_example - stop:10.0f} | {price_example + stop:8.0f} | ±{(stop/price_example)*100:8.2f}% |")

print("\n" + "="*70)

