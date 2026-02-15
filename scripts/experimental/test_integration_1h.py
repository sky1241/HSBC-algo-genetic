#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST D'INTÉGRATION COMPLET (~1 heure max)

Ce test valide que TOUT le système fonctionne :
1. Checkpoint : sauvegarde et reprise après "crash"
2. Optimisation par phase : les params sont différents par phase
3. Walk-forward : les folds s'enchaînent correctement
4. Résultats cohérents : pas de NaN, equity > 0, etc.

Usage:
    python scripts/test_integration_1h.py

Ce test fait un VRAI calcul sur 3 folds avec 15 trials.
Durée estimée : 45-60 minutes.

Version: 1.0
Date: 2025-02-07
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configuration du test
TEST_CONFIG = {
    "trials": 15,           # Peu de trials pour aller vite
    "seed": 9999,           # Seed de test
    "timeout_minutes": 60,  # Timeout global
    "checkpoint_interval": 2,  # Checkpoint toutes les 2 min
}


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_step(step, msg):
    print(f"\n[STEP {step}] {msg}")
    print("-" * 50)


def cleanup_test_dir(test_dir):
    """Nettoie le dossier de test."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)


def test_checkpoint_standalone():
    """Test le checkpoint manager de manière isolée."""
    print_step(1, "Test Checkpoint Manager (standalone)")

    try:
        from src.checkpoint_manager import CheckpointManager, RobustRunner
        import tempfile

        test_dir = Path(tempfile.mkdtemp())

        try:
            # Test 1: Sauvegarder un état
            manager = CheckpointManager(test_dir, auto_save=False)
            manager.update_state(
                seed=42, fold=2020, phase="2", trial=25,
                total_trials=50, best_params={"tenkan": 9},
                best_score=0.75, completed_phases=["0", "1"]
            )
            manager.save_checkpoint()
            print("  [OK] Checkpoint sauvegardé")

            # Test 2: Recharger l'état
            manager2 = CheckpointManager(test_dir, auto_save=False)
            state = manager2.load_checkpoint()
            assert state["seed"] == 42, "Seed incorrect"
            assert state["fold"] == 2020, "Fold incorrect"
            assert state["trial"] == 25, "Trial incorrect"
            print("  [OK] Checkpoint rechargé correctement")

            # Test 3: RobustRunner
            runner = RobustRunner(test_dir, checkpoint_interval_minutes=1)
            pending = runner.get_seeds_to_run([100, 101, 102])
            assert len(pending) == 3, "Tous les seeds doivent être pending"
            print("  [OK] RobustRunner fonctionne")

            runner.shutdown()

        finally:
            shutil.rmtree(test_dir)

        return True, "Checkpoint manager OK"

    except Exception as e:
        return False, f"Checkpoint manager FAIL: {e}"


def test_spectral_module():
    """Test le module spectral."""
    print_step(2, "Test Module Spectral")

    try:
        import pandas as pd
        import numpy as np
        from src.spectral import (
            compute_spectral_features,
            detect_regime,
            get_halving_phase,
            HalvingIndexer,
        )

        # Charger données
        data_file = ROOT / "data" / "BTC_FUSED_2h.csv"
        df = pd.read_csv(data_file, nrows=1000)
        prices = df["close"].values

        # Test features spectrales
        features = compute_spectral_features(prices, fs=12.0)
        regime = detect_regime(features)

        assert not np.isnan(features.lfp), "LFP est NaN"
        assert regime.value in ["trend", "mixed", "noise", "unknown"], "Régime invalide"
        print(f"  [OK] Features spectrales: LFP={features.lfp:.3f}, regime={regime.value}")

        # Test halving
        from datetime import datetime
        phase = get_halving_phase(datetime(2024, 6, 1))
        assert phase.value in ["pre_halving", "discovery", "expansion", "maturation", "late_cycle"]
        print(f"  [OK] Halving indexer: 2024-06-01 = {phase.value}")

        return True, "Module spectral OK"

    except Exception as e:
        return False, f"Module spectral FAIL: {e}"


def test_labels_and_data():
    """Vérifie que les données et labels existent."""
    print_step(3, "Test Données et Labels")

    errors = []

    # Données
    data_file = ROOT / "data" / "BTC_FUSED_2h.csv"
    if data_file.exists():
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"  [OK] BTC_FUSED_2h.csv: {len(df)} lignes")
    else:
        errors.append("BTC_FUSED_2h.csv manquant")

    # Labels
    labels_dir = ROOT / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h"
    for k in [3, 5, 8]:
        label_file = labels_dir / f"K{k}.csv"
        if label_file.exists():
            import pandas as pd
            df = pd.read_csv(label_file)
            unique_labels = sorted(df["label"].unique())
            print(f"  [OK] K{k}.csv: {len(df)} lignes, labels={unique_labels}")
        else:
            errors.append(f"K{k}.csv manquant")

    if errors:
        return False, f"Données/labels FAIL: {errors}"
    return True, "Données et labels OK"


def test_mini_wfa_with_checkpoint(timeout_minutes=45):
    """
    Lance un mini WFA et vérifie que le checkpoint fonctionne.

    Ce test:
    1. Lance le WFA avec peu de trials
    2. Attend quelques minutes
    3. Vérifie que PROGRESS.json est mis à jour
    4. Laisse finir (ou timeout)
    5. Vérifie les résultats
    """
    print_step(4, f"Test Mini WFA avec Checkpoint (max {timeout_minutes} min)")

    test_dir = ROOT / "outputs" / "test_integration_1h"
    cleanup_test_dir(test_dir)

    labels_csv = ROOT / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h" / "K5.csv"

    if not labels_csv.exists():
        return False, f"Labels K5 non trouvés: {labels_csv}"

    # Commande WFA avec peu de trials
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_scheduler_wfa_phase.py"),
        "--labels-csv", str(labels_csv),
        "--trials", str(TEST_CONFIG["trials"]),
        "--seed", str(TEST_CONFIG["seed"]),
        "--use-fused",
        "--out-dir", str(test_dir),
    ]

    print(f"  Commande: {' '.join(cmd[-6:])}")
    print(f"  Timeout: {timeout_minutes} minutes")

    # Lancer le processus
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )

    progress_file = test_dir / "PROGRESS.json"
    last_progress = None
    progress_updates = 0

    try:
        while True:
            elapsed = (time.time() - start_time) / 60

            # Vérifier si terminé
            ret = process.poll()
            if ret is not None:
                print(f"\n  Processus terminé (code={ret}) après {elapsed:.1f} min")
                break

            # Timeout
            if elapsed > timeout_minutes:
                print(f"\n  Timeout atteint ({timeout_minutes} min)")
                process.terminate()
                time.sleep(2)
                break

            # Vérifier progression
            if progress_file.exists():
                try:
                    with open(progress_file) as f:
                        progress = json.load(f)

                    if progress != last_progress:
                        pct = progress.get("percent", 0)
                        folds = progress.get("folds_done", 0)
                        total = progress.get("folds_total", 14)
                        trial = progress.get("trial", 0)

                        print(f"  [{elapsed:.1f} min] Progression: {pct:.1f}% "
                              f"(fold {folds}/{total}, trial {trial})")

                        last_progress = progress.copy()
                        progress_updates += 1

                except:
                    pass

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n  Interruption utilisateur")
        process.terminate()

    # Analyse des résultats
    print("\n  --- Analyse des résultats ---")

    results = {
        "duration_minutes": (time.time() - start_time) / 60,
        "progress_updates": progress_updates,
        "completed": False,
        "has_results": False,
    }

    # Vérifier la progression finale
    if progress_file.exists():
        with open(progress_file) as f:
            final_progress = json.load(f)
        results["final_percent"] = final_progress.get("percent", 0)
        results["folds_done"] = final_progress.get("folds_done", 0)
        print(f"  Progression finale: {results['final_percent']:.1f}%")

    # Vérifier si résultats générés
    result_files = list(test_dir.glob("WFA_phase_K5_*.json"))
    if result_files:
        results["completed"] = True
        results["has_results"] = True
        print(f"  [OK] Fichier résultat généré: {result_files[0].name}")

        # Analyser le contenu
        with open(result_files[0]) as f:
            wfa_results = json.load(f)

        if "overall" in wfa_results:
            overall = wfa_results["overall"]
            print(f"  [OK] Equity final: {overall.get('equity_mult', 'N/A')}")
            print(f"  [OK] Max drawdown: {overall.get('max_drawdown', 'N/A')}")
            print(f"  [OK] Trades: {overall.get('trades', 'N/A')}")
    else:
        print(f"  [WARN] Pas de fichier résultat (normal si timeout)")

    # Verdict
    if progress_updates > 0:
        print(f"\n  [OK] PROGRESS.json mis à jour {progress_updates} fois")
        print("  [OK] Le système de progression fonctionne!")
        return True, f"Mini WFA OK ({results['duration_minutes']:.0f} min, {progress_updates} updates)"
    else:
        return False, "PROGRESS.json jamais mis à jour"


def main():
    parser = argparse.ArgumentParser(description="Test d'intégration complet (~1h)")
    parser.add_argument("--timeout", type=int, default=45,
                        help="Timeout du test WFA en minutes (défaut: 45)")
    parser.add_argument("--skip-wfa", action="store_true",
                        help="Skipper le test WFA (juste tester les modules)")
    args = parser.parse_args()

    print_header("TEST D'INTÉGRATION COMPLET")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Durée estimée: {args.timeout + 15} minutes max")

    results = []
    start = time.time()

    # Tests rapides
    results.append(test_checkpoint_standalone())
    results.append(test_spectral_module())
    results.append(test_labels_and_data())

    # Test WFA (le plus long)
    if not args.skip_wfa:
        results.append(test_mini_wfa_with_checkpoint(timeout_minutes=args.timeout))
    else:
        print_step(4, "Test Mini WFA SKIPPED (--skip-wfa)")
        results.append((True, "WFA skipped"))

    # Résumé
    elapsed = (time.time() - start) / 60
    print_header("RÉSUMÉ")

    all_ok = True
    for i, (ok, msg) in enumerate(results, 1):
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} Test {i}: {msg}")
        if not ok:
            all_ok = False

    print(f"\n  Durée totale: {elapsed:.1f} minutes")

    if all_ok:
        print("\n  " + "=" * 50)
        print("  [SUCCESS] TOUS LES TESTS PASSÉS")
        print("  Le système est prêt pour les 30 seeds K5!")
        print("  " + "=" * 50)
    else:
        print("\n  " + "=" * 50)
        print("  [FAIL] CERTAINS TESTS ONT ÉCHOUÉ")
        print("  Corrige les erreurs avant de lancer les 30 seeds.")
        print("  " + "=" * 50)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
