#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script rapide: vérifier configuration compte réel avant lancement."""
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("🔍 VÉRIFICATION CONFIGURATION COMPTE RÉEL")
print("="*70)

errors = []
warnings = []

# 1. Vérifier .env
env_file = ROOT / ".env"
if not env_file.exists():
    errors.append("❌ Fichier .env introuvable. Copier configs/env.real.example")
else:
    load_dotenv(env_file)
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower()
    
    if not api_key or api_key == "votre_cle_api_reelle":
        errors.append("❌ BINANCE_API_KEY non configurée dans .env")
    else:
        print(f"✅ API Key configurée: {api_key[:10]}...")
    
    if not api_secret or api_secret == "votre_secret_api_reel":
        errors.append("❌ BINANCE_API_SECRET non configurée dans .env")
    else:
        print(f"✅ API Secret configurée")
    
    if testnet == "true":
        warnings.append("⚠️ BINANCE_TESTNET=true (mode testnet). Pour compte réel: false")
    else:
        print(f"✅ Mode compte réel activé (TESTNET=false)")

# 2. Vérifier phase_params_K3.json
params_file = ROOT / "configs" / "phase_params_K3.json"
if params_file.exists():
    import json
    with open(params_file, 'r') as f:
        params = json.load(f)
    if len(params) == 3 and all(str(i) in params for i in [0, 1, 2]):
        print(f"✅ Paramètres phases K3 trouvés (3 phases)")
        for phase in [0, 1, 2]:
            p = params[str(phase)]
            if 'tp_mult' not in p:
                warnings.append(f"⚠️ Phase {phase}: tp_mult manquant")
    else:
        errors.append("❌ phase_params_K3.json invalide (doit contenir phases 0, 1, 2)")
else:
    errors.append("❌ configs/phase_params_K3.json introuvable")

# 3. Vérifier labels K3
labels_file = ROOT / "data" / "K3_1d_stable.csv"
if labels_file.exists():
    print(f"✅ Labels K3 1D stable trouvés: {labels_file}")
else:
    errors.append(f"❌ data/K3_1d_stable.csv introuvable")

# 4. Vérifier mode live dans intraday_runner
runner_file = ROOT / "routines" / "intraday_runner.py"
if runner_file.exists():
    content = runner_file.read_text(encoding='utf-8')
    if 'mode="live"' in content or "mode='live'" in content:
        print(f"✅ Mode LIVE activé dans intraday_runner.py")
    elif 'mode="simulation"' in content or "mode='simulation'" in content:
        warnings.append("⚠️ Mode SIMULATION activé. Changer en 'live' pour compte réel")
    else:
        warnings.append("⚠️ Mode non détecté dans intraday_runner.py")

# Résumé
print("\n" + "="*70)
if errors:
    print(f"❌ {len(errors)} ERREUR(S) CRITIQUE(S):")
    for e in errors:
        print(f"   {e}")
    print("\n⚠️ CORRIGER AVANT DE LANCER EN LIVE!")
    sys.exit(1)
elif warnings:
    print(f"⚠️ {len(warnings)} AVERTISSEMENT(S):")
    for w in warnings:
        print(f"   {w}")
    print("\n✅ Configuration OK mais vérifier les avertissements")
    sys.exit(0)
else:
    print("✅ CONFIGURATION COMPLÈTE ET VALIDE!")
    print("\nPrêt pour compte réel. Lancer:")
    print("  1. python routines/daily_phase_job.py  # 1×/jour")
    print("  2. python routines/intraday_runner.py  # Toutes les 2h")
    print("="*70)
    sys.exit(0)

