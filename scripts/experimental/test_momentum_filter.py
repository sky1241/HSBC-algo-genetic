#!/usr/bin/env python3
"""Test comparatif: AVEC vs SANS filtre momentum"""

import os
os.environ['USE_FUSED_H2'] = '1'
import sys
sys.path.insert(0, '.')

import pandas as pd

# Charger donnees
import ichimoku_pipeline_web_v4_8_fixed as pipe

print("Chargement donnees...")
df = pipe._load_local_csv_if_configured('BTC/USDT', '2h')
df = pipe.ensure_utc_index(df)
print(f"Donnees: {len(df)} barres")

# Params medians K3 Phase 1
tenkan, kijun, senkou_b, shift = 29, 58, 232, 96
atr_mult = 19.5

print("\n" + "="*50)
print("TEST AVEC FILTRE MOMENTUM 6h (code actuel)")
print("="*50)

m = pipe.backtest_long_short(df.copy(), tenkan, kijun, senkou_b, shift, atr_mult,
                              loss_mult=3.0, symbol='BTC/USDT')

print(f"\nResultats AVEC filtre:")
print(f"  Equity finale: {m.get('equity_final', 0):.4f}")
print(f"  Max Drawdown: {m.get('max_drawdown', 0)*100:.1f}%")
print(f"  Trades: {m.get('trades', 0)}")
print(f"  CAGR: {m.get('CAGR', 0)*100:.2f}%")
