#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration sécurisée des clés API Binance."""
import os
from pathlib import Path
from getpass import getpass

ROOT = Path(__file__).parent
ENV_FILE = ROOT / ".env"

print("="*70)
print("🔐 CONFIGURATION CLÉS API BINANCE")
print("="*70)
print("\n⚠️  Les clés API seront stockées localement dans .env")
print("⚠️  Le fichier .env est dans .gitignore (NE SERA PAS sur GitHub)")
print("="*70)

# Demander clés API
print("\n📝 Entrez vos clés API Binance:")
api_key = input("API Key: ").strip()
api_secret = getpass("API Secret (masqué): ").strip()
testnet = input("Mode testnet? (o/n, défaut=n): ").strip().lower()

if not api_key or not api_secret:
    print("\n❌ Erreur: API Key et Secret requis!")
    exit(1)

# Créer .env
env_content = f"""# Configuration Binance - NE PAS COMMITER SUR GITHUB
# Ce fichier est dans .gitignore et reste local uniquement
BINANCE_API_KEY={api_key}
BINANCE_API_SECRET={api_secret}
BINANCE_TESTNET={'true' if testnet == 'o' else 'false'}
"""

with open(ENV_FILE, 'w') as f:
    f.write(env_content)

# Chmod 600 (sécurité Unix/Linux)
try:
    os.chmod(ENV_FILE, 0o600)
except:
    pass  # Windows ignore chmod

print("\n" + "="*70)
print("✅ Clés API configurées et sauvegardées dans .env")
print(f"   Fichier: {ENV_FILE}")
print("   ⚠️  Ce fichier est LOCAL uniquement (pas sur GitHub)")
print("="*70)

# Vérifier que .gitignore protège bien
gitignore_file = ROOT / ".gitignore"
if gitignore_file.exists():
    content = gitignore_file.read_text()
    if ".env" in content:
        print("\n✅ .env est bien dans .gitignore (sécurisé)")
    else:
        print("\n⚠️  Ajouter .env dans .gitignore pour sécurité")
else:
    print("\n⚠️  Créer .gitignore avec .env dedans")

print("\n📋 Prochaines étapes:")
print("1. python check_real.py  # Vérifier configuration")
print("2. python routines/daily_phase_job.py  # Test daily")
print("="*70)



