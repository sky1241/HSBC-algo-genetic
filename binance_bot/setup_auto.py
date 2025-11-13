#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup automatique complet du bot Binance - TOUT EN UN."""
import sys
from pathlib import Path
import shutil
import json

ROOT = Path(__file__).parent

print("="*70)
print("🚀 SETUP AUTOMATIQUE BOT BINANCE")
print("="*70)

# 1. Créer .env depuis template
print("\n1️⃣ Configuration .env...")
env_file = ROOT / ".env"
env_template = ROOT / "configs" / "env.real.example"

if not env_file.exists():
    if env_template.exists():
        shutil.copy(env_template, env_file)
        print(f"   ✅ .env créé depuis template")
        print(f"   ⚠️  ÉDITER MAINTENANT: {env_file}")
        print(f"      - Mettre vos vraies clés API Binance")
        print(f"      - BINANCE_TESTNET=false pour compte réel")
    else:
        # Créer .env basique
        with open(env_file, 'w') as f:
            f.write("""# Configuration Binance
BINANCE_API_KEY=votre_cle_api_ici
BINANCE_API_SECRET=votre_secret_api_ici
BINANCE_TESTNET=false
""")
        print(f"   ✅ .env créé (template basique)")
else:
    print(f"   ✅ .env existe déjà")

# 2. Vérifier/copier labels K3
print("\n2️⃣ Labels K3 1D stable...")
labels_dst = ROOT / "data" / "K3_1d_stable.csv"
labels_sources = [
    ROOT.parent / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h" / "K3_1d_stable.csv",
    Path("E:/ichimoku_runs/wfa_phase_k3_1d_stable") / "K3_1d_stable.csv",
]

if labels_dst.exists():
    print(f"   ✅ Labels trouvés: {labels_dst}")
else:
    found = False
    for src in labels_sources:
        if src.exists():
            shutil.copy(src, labels_dst)
            print(f"   ✅ Labels copiés depuis: {src}")
            found = True
            break
    
    if not found:
        print(f"   ⚠️  Labels introuvables. Copier manuellement:")
        print(f"      Source: outputs/fourier/labels_frozen/BTC_FUSED_2h/K3_1d_stable.csv")
        print(f"      Destination: {labels_dst}")

# 3. Vérifier paramètres phases
print("\n3️⃣ Paramètres phases K3...")
params_file = ROOT / "configs" / "phase_params_K3.json"
if params_file.exists():
    with open(params_file, 'r') as f:
        params = json.load(f)
    if len(params) == 3:
        print(f"   ✅ Paramètres phases trouvés (3 phases)")
    else:
        print(f"   ⚠️  Paramètres incomplets. Régénérer avec:")
        print(f"      python scripts/extract_k3_params_for_bot.py")
else:
    print(f"   ⚠️  Paramètres manquants. Régénérer avec:")
    print(f"      python scripts/extract_k3_params_for_bot.py")

# 4. Créer dossiers nécessaires
print("\n4️⃣ Structure dossiers...")
for d in ["data", "logs"]:
    (ROOT / d).mkdir(exist_ok=True)
    print(f"   ✅ {d}/")

# 5. Vérifier dépendances
print("\n5️⃣ Dépendances Python...")
print("   Installer avec: pip install -r requirements.txt")
print("   Dépendances: ccxt, pandas, numpy, pyyaml, python-dotenv")

# Résumé
print("\n" + "="*70)
print("✅ SETUP TERMINÉ")
print("="*70)
print("\n📋 PROCHAINES ÉTAPES:")
print("1. Éditer .env avec vos clés API Binance")
print("2. Installer dépendances: pip install -r requirements.txt")
print("3. Vérifier configuration: python check_real.py")
print("4. Tester daily: python routines/daily_phase_job.py")
print("5. Tester intraday (simulation): python routines/intraday_runner.py")
print("\n⚠️  Pour compte réel:")
print("   - BINANCE_TESTNET=false dans .env")
print("   - trade_mode = 'live' dans routines/intraday_runner.py (ligne 62)")
print("="*70)

