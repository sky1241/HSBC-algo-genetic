#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHECKPOINT MANAGER - Sauvegarde et reprise après crash

Gère:
- Checkpoints automatiques toutes les N minutes
- Reprise exacte après crash/arrêt
- Sauvegarde des résultats (pas des logs)
- Nettoyage intelligent des fichiers temporaires

Usage:
    manager = CheckpointManager("outputs/wfa_phase_k5")

    # Sauvegarder progression
    manager.save_checkpoint(seed=101, fold=2023, phase="3", trial=150)

    # Reprendre après crash
    state = manager.load_checkpoint()
    if state:
        resume_from(state["seed"], state["fold"], state["phase"], state["trial"])

Version: 1.0
Date: 2025-02-03
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckpointManager")


@dataclass
class CheckpointState:
    """État d'un checkpoint."""
    timestamp: str
    seed: int
    fold: int
    phase: str
    trial: int
    total_trials: int
    best_params: Dict[str, Any]
    best_score: float
    completed_phases: List[str]
    elapsed_seconds: float


class CheckpointManager:
    """
    Gestionnaire de checkpoints pour reprise après crash.

    Sauvegarde périodiquement:
    - État de progression (seed, fold, phase, trial)
    - Meilleurs paramètres trouvés
    - Résultats intermédiaires
    """

    def __init__(
        self,
        output_dir: str | Path,
        checkpoint_interval_minutes: int = 10,
        auto_save: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "CHECKPOINT.json"
        self.backup_file = self.output_dir / "CHECKPOINT_BACKUP.json"
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self.auto_save = auto_save

        # État courant
        self.current_state: Optional[CheckpointState] = None
        self.start_time = time.time()

        # Thread de sauvegarde auto
        self._stop_auto_save = threading.Event()
        self._auto_save_thread: Optional[threading.Thread] = None

        if auto_save:
            self._start_auto_save()

    def _start_auto_save(self) -> None:
        """Démarre le thread de sauvegarde automatique."""
        def _auto_save_loop():
            while not self._stop_auto_save.wait(timeout=self.checkpoint_interval):
                if self.current_state:
                    self.save_checkpoint()
                    logger.debug("Auto-saved checkpoint")

        self._auto_save_thread = threading.Thread(target=_auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        logger.info(f"Auto-save enabled every {self.checkpoint_interval // 60} minutes")

    def stop_auto_save(self) -> None:
        """Arrête le thread de sauvegarde automatique."""
        self._stop_auto_save.set()
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5)

    def update_state(
        self,
        seed: int,
        fold: int,
        phase: str,
        trial: int,
        total_trials: int,
        best_params: Dict[str, Any],
        best_score: float,
        completed_phases: List[str],
    ) -> None:
        """Met à jour l'état courant."""
        self.current_state = CheckpointState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=seed,
            fold=fold,
            phase=phase,
            trial=trial,
            total_trials=total_trials,
            best_params=best_params,
            best_score=best_score,
            completed_phases=completed_phases,
            elapsed_seconds=time.time() - self.start_time,
        )

    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Sauvegarde le checkpoint actuel.

        Utilise une stratégie de double fichier pour éviter la corruption:
        1. Écrit dans CHECKPOINT_BACKUP.json
        2. Renomme en CHECKPOINT.json (atomique)
        """
        if self.current_state is None:
            return False

        try:
            # Mettre à jour le timestamp
            self.current_state.timestamp = datetime.now(timezone.utc).isoformat()
            self.current_state.elapsed_seconds = time.time() - self.start_time

            # Écrire dans le backup d'abord
            state_dict = asdict(self.current_state)

            with open(self.backup_file, "w") as f:
                json.dump(state_dict, f, indent=2)

            # Renommer (atomique sur la plupart des systèmes)
            shutil.move(str(self.backup_file), str(self.checkpoint_file))

            logger.debug(f"Checkpoint saved: seed={self.current_state.seed}, "
                        f"fold={self.current_state.fold}, phase={self.current_state.phase}, "
                        f"trial={self.current_state.trial}/{self.current_state.total_trials}")

            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Charge le dernier checkpoint.

        Returns:
            Dict avec l'état sauvegardé, ou None si pas de checkpoint
        """
        # Essayer le fichier principal
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    state = json.load(f)
                logger.info(f"Loaded checkpoint: seed={state.get('seed')}, "
                           f"fold={state.get('fold')}, phase={state.get('phase')}, "
                           f"trial={state.get('trial')}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load main checkpoint: {e}")

        # Essayer le backup
        if self.backup_file.exists():
            try:
                with open(self.backup_file, "r") as f:
                    state = json.load(f)
                logger.info(f"Loaded backup checkpoint: seed={state.get('seed')}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load backup checkpoint: {e}")

        return None

    def clear_checkpoint(self) -> None:
        """Efface les checkpoints (après complétion réussie)."""
        for f in [self.checkpoint_file, self.backup_file]:
            if f.exists():
                f.unlink()
        logger.info("Checkpoints cleared")

    def mark_seed_complete(self, seed: int, results: Dict[str, Any]) -> None:
        """
        Marque un seed comme complet et sauvegarde ses résultats.

        Les résultats sont sauvegardés de manière permanente (pas effacés).
        """
        seed_dir = self.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les résultats finaux
        results_file = seed_dir / "RESULTS_FINAL.json"
        results["completion_time"] = datetime.now(timezone.utc).isoformat()
        results["status"] = "completed"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Seed {seed} marked complete: {results_file}")

    def get_completed_seeds(self) -> List[int]:
        """Retourne la liste des seeds déjà complétés."""
        completed = []
        for seed_dir in self.output_dir.glob("seed_*"):
            results_file = seed_dir / "RESULTS_FINAL.json"
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        data = json.load(f)
                    if data.get("status") == "completed":
                        seed = int(seed_dir.name.split("_")[-1])
                        completed.append(seed)
                except Exception:
                    continue
        return sorted(completed)

    def get_pending_seeds(self, all_seeds: List[int]) -> List[int]:
        """Retourne les seeds pas encore complétés."""
        completed = set(self.get_completed_seeds())
        return [s for s in all_seeds if s not in completed]


class DiskSpaceManager:
    """
    Gestionnaire d'espace disque.

    Nettoie automatiquement les fichiers temporaires
    tout en préservant les résultats d'apprentissage.
    """

    # Extensions à garder (résultats d'apprentissage)
    KEEP_EXTENSIONS = {".json", ".pkl", ".csv"}
    KEEP_FILENAMES = {"RESULTS_FINAL.json", "CHECKPOINT.json", "aggregated_params.json"}

    # Extensions à nettoyer (logs temporaires)
    CLEAN_EXTENSIONS = {".log", ".tmp", ".temp"}
    CLEAN_PATTERNS = ["run.log", "optuna_*.log", "*.tmp", "progress_*"]

    def __init__(self, output_dir: str | Path, min_free_gb: float = 10.0):
        self.output_dir = Path(output_dir)
        self.min_free_gb = min_free_gb

    def get_free_space_gb(self) -> float:
        """Retourne l'espace libre en Go."""
        import shutil
        total, used, free = shutil.disk_usage(self.output_dir)
        return free / (1024 ** 3)

    def cleanup_seed_logs(self, seed_dir: Path, keep_last_n_lines: int = 1000) -> int:
        """
        Nettoie les logs d'un seed tout en gardant les dernières lignes.

        Returns:
            Nombre d'octets libérés
        """
        freed = 0

        for log_file in seed_dir.glob("*.log"):
            try:
                size_before = log_file.stat().st_size

                # Garder les dernières lignes
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if len(lines) > keep_last_n_lines:
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.writelines(lines[-keep_last_n_lines:])

                    size_after = log_file.stat().st_size
                    freed += size_before - size_after

            except Exception as e:
                logger.warning(f"Failed to cleanup {log_file}: {e}")

        return freed

    def cleanup_temp_files(self) -> int:
        """
        Supprime tous les fichiers temporaires.

        Returns:
            Nombre d'octets libérés
        """
        freed = 0

        for pattern in ["**/*.tmp", "**/*.temp", "**/progress_*.json"]:
            for f in self.output_dir.glob(pattern):
                try:
                    size = f.stat().st_size
                    f.unlink()
                    freed += size
                except Exception:
                    continue

        return freed

    def auto_cleanup(self) -> Dict[str, Any]:
        """
        Nettoyage automatique si l'espace est insuffisant.

        Returns:
            Dict avec les stats de nettoyage
        """
        stats = {
            "free_before_gb": self.get_free_space_gb(),
            "freed_bytes": 0,
            "actions": [],
        }

        # Vérifier si nettoyage nécessaire
        if stats["free_before_gb"] >= self.min_free_gb:
            stats["actions"].append("No cleanup needed")
            return stats

        # 1. Nettoyer les fichiers temp
        freed = self.cleanup_temp_files()
        stats["freed_bytes"] += freed
        if freed > 0:
            stats["actions"].append(f"Cleaned temp files: {freed / 1024 / 1024:.1f} MB")

        # 2. Tronquer les logs si encore insuffisant
        if self.get_free_space_gb() < self.min_free_gb:
            for seed_dir in self.output_dir.glob("seed_*"):
                freed = self.cleanup_seed_logs(seed_dir)
                stats["freed_bytes"] += freed
            if stats["freed_bytes"] > 0:
                stats["actions"].append(f"Truncated logs")

        stats["free_after_gb"] = self.get_free_space_gb()

        logger.info(f"Cleanup: freed {stats['freed_bytes'] / 1024 / 1024:.1f} MB, "
                   f"free space: {stats['free_after_gb']:.1f} GB")

        return stats


# =============================================================================
# WRAPPER POUR INTÉGRATION FACILE
# =============================================================================

class RobustRunner:
    """
    Runner robuste avec checkpoints et gestion d'espace.

    Usage:
        runner = RobustRunner("outputs/wfa_phase_k5")

        for seed in runner.get_seeds_to_run([101, 102, ...]):
            for fold, phase in runner.get_work_items(seed):
                result = do_optimization(...)
                runner.save_progress(seed, fold, phase, result)
            runner.complete_seed(seed, final_results)
    """

    def __init__(
        self,
        output_dir: str | Path,
        checkpoint_interval_minutes: int = 10,
        min_free_gb: float = 10.0,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_mgr = CheckpointManager(
            output_dir,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
        )
        self.disk_mgr = DiskSpaceManager(output_dir, min_free_gb=min_free_gb)

        # État de progression
        self._results_buffer: Dict[str, Any] = {}

    def get_seeds_to_run(self, all_seeds: List[int]) -> List[int]:
        """Retourne les seeds restants à exécuter."""
        # Vérifier les seeds complétés
        completed = self.checkpoint_mgr.get_completed_seeds()
        pending = [s for s in all_seeds if s not in completed]

        logger.info(f"Seeds: {len(completed)} completed, {len(pending)} pending")

        return pending

    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        """Retourne l'état pour reprendre après crash."""
        return self.checkpoint_mgr.load_checkpoint()

    def save_progress(
        self,
        seed: int,
        fold: int,
        phase: str,
        trial: int,
        total_trials: int,
        best_params: Dict[str, Any],
        best_score: float,
        completed_phases: List[str],
    ) -> None:
        """Sauvegarde la progression courante."""
        self.checkpoint_mgr.update_state(
            seed=seed,
            fold=fold,
            phase=phase,
            trial=trial,
            total_trials=total_trials,
            best_params=best_params,
            best_score=best_score,
            completed_phases=completed_phases,
        )

        # Auto-cleanup périodique
        if trial % 50 == 0:
            self.disk_mgr.auto_cleanup()

    def complete_seed(self, seed: int, results: Dict[str, Any]) -> None:
        """Marque un seed comme complété."""
        self.checkpoint_mgr.mark_seed_complete(seed, results)
        self.checkpoint_mgr.clear_checkpoint()

    def shutdown(self) -> None:
        """Arrêt propre."""
        self.checkpoint_mgr.save_checkpoint(force=True)
        self.checkpoint_mgr.stop_auto_save()
        logger.info("RobustRunner shutdown complete")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHECKPOINT MANAGER - Test")
    print("=" * 60)

    # Test basique
    test_dir = Path("outputs/_test_checkpoint")
    test_dir.mkdir(parents=True, exist_ok=True)

    manager = CheckpointManager(test_dir, auto_save=False)

    # Simuler progression
    print("\n1. Simulating progress...")
    for trial in range(0, 100, 20):
        manager.update_state(
            seed=101,
            fold=2023,
            phase="2",
            trial=trial,
            total_trials=300,
            best_params={"tenkan": 9, "kijun": 26},
            best_score=0.5 + trial * 0.001,
            completed_phases=["0", "1"],
        )
        manager.save_checkpoint()
        print(f"   Saved checkpoint at trial {trial}")

    # Simuler crash et reprise
    print("\n2. Simulating crash recovery...")
    manager2 = CheckpointManager(test_dir, auto_save=False)
    state = manager2.load_checkpoint()

    if state:
        print(f"   Recovered: seed={state['seed']}, fold={state['fold']}, "
              f"phase={state['phase']}, trial={state['trial']}")

    # Test RobustRunner
    print("\n3. Testing RobustRunner...")
    runner = RobustRunner(test_dir, checkpoint_interval_minutes=1)

    seeds_to_run = runner.get_seeds_to_run([101, 102, 103])
    print(f"   Seeds to run: {seeds_to_run}")

    runner.shutdown()

    # Cleanup test
    print("\n4. Cleanup...")
    shutil.rmtree(test_dir)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
