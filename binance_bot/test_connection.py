#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test connexion Binance avec clés API configurées."""
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Charger .env
env_file = ROOT / ".env"
if not env_file.exists():
    print("❌ Fichier .env introuvable!")
    sys.exit(1)

load_dotenv(env_file)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

print("="*70)
print("🔍 TEST CONNEXION BINANCE")
print("="*70)

if not api_key or not api_secret:
    print("❌ Clés API non trouvées dans .env")
    sys.exit(1)

print(f"✅ API Key: {api_key[:10]}...{api_key[-5:]}")
print(f"✅ API Secret: {'*' * 20}...{api_secret[-5:]}")
print(f"✅ Mode: {'TESTNET' if testnet else 'LIVE (COMPTE RÉEL)'}")

# Test connexion
try:
    import ccxt
    
    config = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    }
    
    if testnet:
        config['urls'] = {
            'api': {
                'public': 'https://testnet.binancefuture.com',
                'private': 'https://testnet.binancefuture.com',
            }
        }
    
    exchange = ccxt.binance(config)
    
    print("\n📡 Test connexion...")
    balance = exchange.fetch_balance()
    
    print("\n✅ CONNEXION RÉUSSIE!")
    print(f"\n💰 Solde disponible:")
    if 'USDT' in balance:
        free = balance['USDT']['free']
        used = balance['USDT']['used']
        total = balance['USDT']['total']
        print(f"   USDT libre: {free:.2f}")
        print(f"   USDT utilisé: {used:.2f}")
        print(f"   USDT total: {total:.2f}")
    
    # Test récupération prix
    print("\n📈 Test récupération prix BTC/USDT...")
    ticker = exchange.fetch_ticker("BTC/USDT")
    print(f"   Prix actuel: {ticker['last']:.2f} USDT")
    
    print("\n" + "="*70)
    print("✅ TOUT FONCTIONNE! Le bot peut se connecter à Binance.")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERREUR CONNEXION: {e}")
    print("\nVérifie:")
    print("  1. Clés API correctes dans .env")
    print("  2. Permissions API activées sur Binance")
    print("  3. Connexion internet")
    sys.exit(1)

