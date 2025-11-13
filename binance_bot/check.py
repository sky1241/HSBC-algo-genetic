#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script de vérification: teste que tous les modules sont importables."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("🔍 VÉRIFICATION ARCHITECTURE BOT BINANCE")
print("="*70)

errors = []

# 1. Vérifier fichiers config
print("\n📁 Vérification fichiers config...")
configs = [
    "configs/env.example",
    "configs/bot_settings.yaml",
    "configs/phase_params_K3.json"
]
for cfg in configs:
    p = ROOT / cfg
    if p.exists():
        print(f"  ✅ {cfg}")
    else:
        print(f"  ❌ {cfg} MANQUANT")
        errors.append(f"Fichier manquant: {cfg}")

# 2. Vérifier imports modules
print("\n📦 Vérification imports modules...")
modules = [
    ("services.phase_labeller", "PhaseLabeller"),
    ("services.params_loader", "ParamsLoader"),
    ("services.data_fetcher", "DataFetcher"),
    ("services.ichimoku_engine", "calculate_ichimoku"),
    ("services.signal_engine", "SignalEngine"),
    ("bot.state_manager", "StateManager"),
    ("bot.trade_manager", "TradeManager"),
    ("bot.risk_manager", "RiskManager"),
]

for mod_name, class_name in modules:
    try:
        mod = __import__(mod_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
        print(f"  ✅ {mod_name}.{class_name}")
    except Exception as e:
        print(f"  ❌ {mod_name}.{class_name} - ERREUR: {e}")
        errors.append(f"Import error {mod_name}.{class_name}: {e}")

# 3. Vérifier routines
print("\n🔄 Vérification routines...")
routines = [
    "routines/daily_phase_job.py",
    "routines/intraday_runner.py"
]
for r in routines:
    p = ROOT / r
    if p.exists():
        print(f"  ✅ {r}")
    else:
        print(f"  ❌ {r} MANQUANT")
        errors.append(f"Routine manquante: {r}")

# 4. Vérifier structure dossiers
print("\n📂 Vérification structure...")
dirs = ["configs", "data", "services", "bot", "routines", "tests", "logs"]
for d in dirs:
    p = ROOT / d
    if p.exists() and p.is_dir():
        print(f"  ✅ {d}/")
    else:
        print(f"  ❌ {d}/ MANQUANT")
        errors.append(f"Dossier manquant: {d}")

# Résumé
print("\n" + "="*70)
if errors:
    print(f"❌ {len(errors)} ERREUR(S) DÉTECTÉE(S):")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("✅ TOUT EST OK! Architecture complète et fonctionnelle.")
    print("\nPROCHAINES ÉTAPES:")
    print("1. python setup.py  # Configuration initiale")
    print("2. Éditer .env avec vos clés API")
    print("3. Copier data/K3_1d_stable.csv")
    print("4. python routines/daily_phase_job.py  # Test daily")
    print("5. python routines/intraday_runner.py  # Test intraday")
    print("="*70)
    sys.exit(0)

