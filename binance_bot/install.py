#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script tout-en-un: installation + configuration + test."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

print("="*70)
print("🎯 INSTALLATION COMPLÈTE BOT BINANCE")
print("="*70)

# 1. Setup automatique
print("\n1️⃣ Configuration fichiers...")
try:
    import setup_auto
    setup_auto.main()
except Exception as e:
    print(f"   ⚠️  Erreur setup: {e}")

# 2. Installer dépendances
print("\n2️⃣ Installation dépendances Python...")
print("   Exécution: pip install -r requirements.txt")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("   ✅ Dépendances installées")
    else:
        print(f"   ⚠️  Erreur installation: {result.stderr}")
        print("   Installer manuellement: pip install -r requirements.txt")
except Exception as e:
    print(f"   ⚠️  Erreur: {e}")
    print("   Installer manuellement: pip install -r requirements.txt")

# 3. Vérification finale
print("\n3️⃣ Vérification...")
try:
    result = subprocess.run(
        [sys.executable, "check_real.py"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
except Exception as e:
    print(f"   ⚠️  Erreur vérification: {e}")

print("\n" + "="*70)
print("✅ INSTALLATION TERMINÉE")
print("="*70)
print("\n📝 IMPORTANT:")
print("1. Éditer .env avec vos clés API Binance")
print("2. Tester: python routines/daily_phase_job.py")
print("="*70)

