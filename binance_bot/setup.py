#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup initial: copie fichiers nécessaires + vérifie configuration."""
from pathlib import Path
import shutil
import os


def main():
    print("="*70)
    print("🔧 SETUP BOT BINANCE")
    print("="*70)
    
    root = Path(__file__).parent
    
    # 1. Vérifier .env
    env_file = root / ".env"
    env_example = root / "configs" / "env.example"
    
    if not env_file.exists():
        print("\n⚠️ Fichier .env introuvable. Création depuis env.example...")
        shutil.copy(env_example, env_file)
        print(f"✅ .env créé. ÉDITER MAINTENANT avec vos clés API!")
        print(f"   Chemin: {env_file}")
    else:
        print(f"\n✅ .env trouvé: {env_file}")
    
    # 2. Vérifier labels K3
    labels_src = root.parent / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h" / "K3_1d_stable.csv"
    labels_dst = root / "data" / "K3_1d_stable.csv"
    
    if not labels_dst.exists() and labels_src.exists():
        print(f"\n📋 Copie labels K3 1D stable...")
        shutil.copy(labels_src, labels_dst)
        print(f"✅ Labels copiés: {labels_dst}")
    elif labels_dst.exists():
        print(f"\n✅ Labels K3 trouvés: {labels_dst}")
    else:
        print(f"\n⚠️ Labels K3 introuvables. Copier manuellement:")
        print(f"   Source: {labels_src}")
        print(f"   Destination: {labels_dst}")
    
    # 3. Vérifier phase_params_K3.json
    params_file = root / "configs" / "phase_params_K3.json"
    if params_file.exists():
        print(f"\n✅ Paramètres phases trouvés: {params_file}")
        print(f"   VÉRIFIER que les valeurs correspondent à vos analyses WFA!")
    else:
        print(f"\n⚠️ Paramètres phases introuvables: {params_file}")
        print(f"   Générer depuis vos analyses K3 (médianes par phase)")
    
    # 4. Créer dossiers si nécessaire
    (root / "logs").mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("✅ SETUP TERMINÉ")
    print("="*70)
    print("\nPROCHAINES ÉTAPES:")
    print("1. Éditer .env avec vos clés API Binance")
    print("2. Vérifier configs/phase_params_K3.json (médianes WFA)")
    print("3. Copier data/K3_1d_stable.csv si absent")
    print("4. Tester: python tests/test_signal_engine.py")
    print("5. Lancer daily job: python routines/daily_phase_job.py")
    print("6. Lancer intraday: python routines/intraday_runner.py")
    print("\n⚠️ TOUJOURS tester sur TESTNET d'abord!")
    print("="*70)


if __name__ == "__main__":
    main()

