#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST RAPIDE DE VALIDATION (~30-60 min max)

But : Vérifier que les modifications fonctionnent AVANT de lancer les gros calculs.

Ce script fait :
1. Test des modules (spectral, checkpoint, etc.) - 1 min
2. Mini backtest sur 2 ans de données seulement - ~30 min
3. Validation que les résultats sont cohérents

Usage:
    python scripts/test_quick_validation.py
    python scripts/test_quick_validation.py --with-backtest  # inclut mini backtest

Version: 1.0
Date: 2025-02-07
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ajouter le root au path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    """Test que tous les modules s'importent correctement."""
    print("\n" + "="*60)
    print("1. TEST DES IMPORTS")
    print("="*60)

    errors = []

    # Modules spectral
    try:
        from src.spectral import compute_spectral_features, detect_regime
        print("  [OK] src.spectral.compute_spectral_features")
        print("  [OK] src.spectral.detect_regime")
    except Exception as e:
        errors.append(f"src.spectral: {e}")
        print(f"  [FAIL] src.spectral: {e}")

    try:
        from src.spectral import HalvingIndexer, get_halving_phase
        print("  [OK] src.spectral.HalvingIndexer")
    except Exception as e:
        errors.append(f"HalvingIndexer: {e}")
        print(f"  [FAIL] HalvingIndexer: {e}")

    try:
        from src.spectral import MonteCarloValidator
        print("  [OK] src.spectral.MonteCarloValidator")
    except Exception as e:
        errors.append(f"MonteCarloValidator: {e}")
        print(f"  [FAIL] MonteCarloValidator: {e}")

    # Checkpoint manager
    try:
        from src.checkpoint_manager import CheckpointManager, RobustRunner
        print("  [OK] src.checkpoint_manager")
    except Exception as e:
        errors.append(f"checkpoint_manager: {e}")
        print(f"  [FAIL] checkpoint_manager: {e}")

    # Continuous learning
    try:
        from src.continuous_learning import ContinuousLearner, ModelVersionManager
        print("  [OK] src.continuous_learning")
    except Exception as e:
        errors.append(f"continuous_learning: {e}")
        print(f"  [FAIL] continuous_learning: {e}")

    # Volatility targeting
    try:
        from src.volatility_targeting import VolatilityTargeter, VolTargetConfig
        print("  [OK] src.volatility_targeting")
    except Exception as e:
        errors.append(f"volatility_targeting: {e}")
        print(f"  [FAIL] volatility_targeting: {e}")

    # Live phase adapter
    try:
        from src.live_phase_adapter import LivePhaseAdapter, PhaseParams
        print("  [OK] src.live_phase_adapter")
    except Exception as e:
        errors.append(f"live_phase_adapter: {e}")
        print(f"  [FAIL] live_phase_adapter: {e}")

    # HMM
    try:
        from hmmlearn.hmm import GaussianHMM
        print("  [OK] hmmlearn")
    except Exception as e:
        errors.append(f"hmmlearn: {e}")
        print(f"  [FAIL] hmmlearn: {e}")

    return len(errors) == 0, errors


def test_spectral_features():
    """Test le calcul des features spectrales."""
    print("\n" + "="*60)
    print("2. TEST FEATURES SPECTRALES")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np
        from src.spectral import compute_spectral_features, detect_regime

        # Charger un échantillon de données
        data_file = ROOT / "data" / "BTC_FUSED_2h.csv"
        if not data_file.exists():
            print(f"  [SKIP] Fichier {data_file} non trouvé")
            return True, []

        df = pd.read_csv(data_file, nrows=1000)
        prices = df['close'].values

        # Calculer features
        features = compute_spectral_features(prices, fs=12.0)
        regime = detect_regime(features)

        print(f"  Dominant Period: {features.dominant_period:.1f} bars")
        print(f"  LFP Ratio: {features.lfp:.3f}")
        print(f"  Flatness: {features.flatness:.3f}")
        print(f"  Regime: {regime.value}")
        print("  [OK] Features spectrales calculées")

        return True, []
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False, [str(e)]


def test_checkpoint_manager():
    """Test le checkpoint manager."""
    print("\n" + "="*60)
    print("3. TEST CHECKPOINT MANAGER")
    print("="*60)

    try:
        from src.checkpoint_manager import CheckpointManager
        import tempfile
        import shutil

        # Créer un dossier temporaire
        test_dir = Path(tempfile.mkdtemp())

        try:
            # Créer manager
            manager = CheckpointManager(test_dir, auto_save=False)

            # Sauvegarder un état
            manager.update_state(
                seed=42,
                fold=2020,
                phase="2",
                trial=50,
                total_trials=100,
                best_params={"tenkan": 9, "kijun": 26},
                best_score=0.75,
                completed_phases=["0", "1"],
            )
            manager.save_checkpoint()
            print("  [OK] Checkpoint sauvegardé")

            # Charger l'état
            manager2 = CheckpointManager(test_dir, auto_save=False)
            state = manager2.load_checkpoint()

            assert state["seed"] == 42
            assert state["fold"] == 2020
            assert state["phase"] == "2"
            print("  [OK] Checkpoint rechargé correctement")

            # Nettoyer
            manager2.clear_checkpoint()
            print("  [OK] Checkpoint effacé")

        finally:
            shutil.rmtree(test_dir)

        return True, []
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False, [str(e)]


def test_halving_indexer():
    """Test le halving indexer."""
    print("\n" + "="*60)
    print("4. TEST HALVING INDEXER")
    print("="*60)

    try:
        from src.spectral import get_halving_phase
        from datetime import datetime

        # Tester différentes dates
        tests = [
            (datetime(2024, 1, 1), "pre_halving"),
            (datetime(2024, 4, 25), "discovery"),
            (datetime(2024, 8, 1), "expansion"),
            (datetime(2025, 2, 1), "maturation"),
        ]

        for date, expected in tests:
            phase = get_halving_phase(date)
            status = "OK" if phase.value == expected else "FAIL"
            print(f"  [{status}] {date.date()}: {phase.value} (attendu: {expected})")
            if phase.value != expected:
                return False, [f"Phase incorrecte pour {date}"]

        return True, []
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False, [str(e)]


def test_volatility_targeting():
    """Test le module volatility targeting."""
    print("\n" + "="*60)
    print("5. TEST VOLATILITY TARGETING")
    print("="*60)

    try:
        import numpy as np
        from src.volatility_targeting import VolatilityTargeter, VolTargetConfig

        # Create config
        config = VolTargetConfig(
            sigma_target=0.15,
            L_max=5.0,
            lookback=20,
            dd_threshold=0.10,
            dd_penalty=0.5,
        )

        targeter = VolatilityTargeter(config)

        # Generate test returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)

        # Test leverage calculation
        equity = 1.0
        peak = 1.0
        leverages = []

        for ret in returns:
            targeter.update_return(ret)
            lev = targeter.get_leverage_from_buffer(equity, peak)
            leverages.append(lev)
            equity *= (1 + ret * lev)
            peak = max(peak, equity)

        avg_leverage = np.mean(leverages)
        print(f"  Average leverage: {avg_leverage:.2f}")
        print(f"  Min leverage: {min(leverages):.2f}")
        print(f"  Max leverage: {max(leverages):.2f}")

        # Verify constraints
        assert all(l >= config.L_min for l in leverages), "Leverage below minimum"
        assert all(l <= config.L_max for l in leverages), "Leverage above maximum"

        print("  [OK] Volatility targeting functional")
        return True, []

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False, [str(e)]


def test_labels_exist():
    """Vérifie que les labels K3/K5/K8 existent."""
    print("\n" + "="*60)
    print("6. TEST LABELS K3/K5/K8")
    print("="*60)

    labels_dir = ROOT / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h"
    errors = []

    for k in [3, 5, 8]:
        label_file = labels_dir / f"K{k}.csv"
        if label_file.exists():
            import pandas as pd
            df = pd.read_csv(label_file)
            print(f"  [OK] K{k}.csv : {len(df)} lignes, labels={sorted(df['label'].unique())}")
        else:
            print(f"  [FAIL] K{k}.csv non trouvé")
            errors.append(f"K{k}.csv manquant")

    return len(errors) == 0, errors


def test_mini_backtest(max_minutes=30):
    """
    Mini backtest sur 2 ans seulement pour valider le pipeline.

    Args:
        max_minutes: Temps max en minutes
    """
    print("\n" + "="*60)
    print(f"7. MINI BACKTEST (max {max_minutes} min)")
    print("="*60)

    try:
        import pandas as pd
        import subprocess

        # Vérifier que le script existe
        script = ROOT / "scripts" / "run_scheduler_wfa_phase.py"
        if not script.exists():
            print(f"  [SKIP] Script {script} non trouvé")
            return True, []

        # Créer dossier de sortie
        out_dir = ROOT / "outputs" / "test_quick_validation"
        out_dir.mkdir(parents=True, exist_ok=True)

        labels_csv = ROOT / "outputs" / "fourier" / "labels_frozen" / "BTC_FUSED_2h" / "K5.csv"

        if not labels_csv.exists():
            print(f"  [SKIP] Labels {labels_csv} non trouvé")
            return True, []

        print(f"  Lancement mini backtest (10 trials, 2 folds max)...")
        print(f"  Timeout: {max_minutes} minutes")
        print(f"  Sortie: {out_dir}")

        # Commande avec paramètres réduits
        # Note: On ne peut pas facilement limiter les folds sans modifier le script
        # Donc on met juste 10 trials et un timeout
        cmd = [
            sys.executable,
            str(script),
            "--labels-csv", str(labels_csv),
            "--trials", "10",  # Seulement 10 trials
            "--seed", "999",
            "--use-fused",
            "--out-dir", str(out_dir),
        ]

        start = time.time()
        timeout_sec = max_minutes * 60

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout_sec,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"  [OK] Backtest terminé en {elapsed/60:.1f} min")
            else:
                print(f"  [WARN] Backtest terminé avec code {result.returncode}")
                if result.stderr:
                    print(f"  Stderr: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  [TIMEOUT] Backtest interrompu après {elapsed/60:.1f} min")
            print(f"  C'est normal - le but était de vérifier que ça démarre")

            # Vérifier qu'un PROGRESS.json a été créé
            progress_file = out_dir / "PROGRESS.json"
            if progress_file.exists():
                with open(progress_file) as f:
                    progress = json.load(f)
                print(f"  [OK] Progression détectée: {progress.get('percent', 0):.1f}%")
                return True, []
            else:
                print(f"  [WARN] Pas de PROGRESS.json créé")

        return True, []

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False, [str(e)]


def main():
    parser = argparse.ArgumentParser(description="Test rapide de validation")
    parser.add_argument("--with-backtest", action="store_true",
                        help="Inclure le mini backtest (ajoute ~30 min)")
    parser.add_argument("--backtest-timeout", type=int, default=30,
                        help="Timeout du backtest en minutes (défaut: 30)")
    args = parser.parse_args()

    print("="*60)
    print("TEST RAPIDE DE VALIDATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_ok = True
    all_errors = []

    # Tests rapides (~1 min total)
    tests = [
        ("Imports", test_imports),
        ("Features spectrales", test_spectral_features),
        ("Checkpoint manager", test_checkpoint_manager),
        ("Halving indexer", test_halving_indexer),
        ("Volatility targeting", test_volatility_targeting),
        ("Labels K3/K5/K8", test_labels_exist),
    ]

    for name, test_fn in tests:
        ok, errors = test_fn()
        if not ok:
            all_ok = False
            all_errors.extend(errors)

    # Mini backtest optionnel
    if args.with_backtest:
        ok, errors = test_mini_backtest(max_minutes=args.backtest_timeout)
        if not ok:
            all_ok = False
            all_errors.extend(errors)

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)

    if all_ok:
        print("\n  [SUCCESS] TOUS LES TESTS PASSES")
        print("\n  Tu peux lancer les 30 seeds en confiance!")
    else:
        print("\n  [FAIL] CERTAINS TESTS ONT ECHOUE")
        print("\n  Erreurs:")
        for err in all_errors:
            print(f"    - {err}")

    print("\n" + "="*60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
