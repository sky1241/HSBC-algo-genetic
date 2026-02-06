#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTINUOUS LEARNING - Apprentissage continu sur données live

Ce module gère:
1. Mise à jour périodique des données (fetch nouvelles barres)
2. Re-fit du modèle HMM avec données étendues
3. Re-optimisation des params par phase (si nécessaire)
4. Versioning des modèles (historique des params)

Usage:
    # Mode automatique (cron/scheduler)
    python src/continuous_learning.py --auto

    # Mode manuel
    python src/continuous_learning.py --update-data
    python src/continuous_learning.py --refit-hmm
    python src/continuous_learning.py --reoptimize

Fréquence recommandée:
    - update-data: quotidien
    - refit-hmm: hebdomadaire
    - reoptimize: mensuel (si drift détecté)

Version: 1.0
Date: 2025-02-03
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ContinuousLearning")


class ModelVersionManager:
    """
    Gère le versioning des modèles et params.

    Garde un historique des versions pour:
    - Rollback si nouvelle version performe moins bien
    - Analyse de la dérive des params
    - Audit trail
    """

    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.versions_file = self.models_dir / "versions.json"
        self.current_link = self.models_dir / "current"

    def get_versions(self) -> List[Dict[str, Any]]:
        """Retourne la liste des versions."""
        if not self.versions_file.exists():
            return []
        with open(self.versions_file, "r") as f:
            return json.load(f)

    def get_current_version(self) -> Optional[str]:
        """Retourne la version courante."""
        versions = self.get_versions()
        if not versions:
            return None
        return versions[-1].get("version_id")

    def save_new_version(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        reason: str = "scheduled_update"
    ) -> str:
        """
        Sauvegarde une nouvelle version.

        Returns:
            version_id
        """
        # Générer version ID
        version_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = self.models_dir / version_id

        version_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder params
        with open(version_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Sauvegarder métriques
        with open(version_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Mettre à jour versions.json
        versions = self.get_versions()
        versions.append({
            "version_id": version_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "metrics_summary": {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
        })

        with open(self.versions_file, "w") as f:
            json.dump(versions, f, indent=2)

        # Mettre à jour le lien "current"
        if self.current_link.exists():
            self.current_link.unlink()

        # Copier vers current
        shutil.copytree(version_dir, self.current_link, dirs_exist_ok=True)

        logger.info(f"Saved new version: {version_id}")
        return version_id

    def load_version(self, version_id: str = "current") -> Dict[str, Any]:
        """Charge une version spécifique."""
        if version_id == "current":
            version_dir = self.current_link
        else:
            version_dir = self.models_dir / version_id

        if not version_dir.exists():
            raise FileNotFoundError(f"Version not found: {version_id}")

        with open(version_dir / "params.json", "r") as f:
            params = json.load(f)

        return params

    def rollback(self, version_id: str) -> bool:
        """Rollback vers une version précédente."""
        version_dir = self.models_dir / version_id

        if not version_dir.exists():
            logger.error(f"Version not found: {version_id}")
            return False

        # Copier vers current
        if self.current_link.exists():
            shutil.rmtree(self.current_link)

        shutil.copytree(version_dir, self.current_link)

        logger.info(f"Rolled back to version: {version_id}")
        return True

    def cleanup_old_versions(self, keep_last_n: int = 10) -> int:
        """Supprime les anciennes versions."""
        versions = self.get_versions()

        if len(versions) <= keep_last_n:
            return 0

        to_delete = versions[:-keep_last_n]
        deleted = 0

        for v in to_delete:
            version_dir = self.models_dir / v["version_id"]
            if version_dir.exists():
                shutil.rmtree(version_dir)
                deleted += 1

        # Mettre à jour versions.json
        with open(self.versions_file, "w") as f:
            json.dump(versions[-keep_last_n:], f, indent=2)

        logger.info(f"Cleaned up {deleted} old versions")
        return deleted


class DataUpdater:
    """
    Met à jour les données OHLCV depuis l'exchange.
    """

    def __init__(self, data_dir: str | Path, symbol: str = "BTC/USDT", timeframe: str = "2h"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.timeframe = timeframe

        self.data_file = self.data_dir / "BTC_FUSED_2h.csv"

    def fetch_new_data(self, since: datetime) -> pd.DataFrame:
        """Fetch nouvelles données depuis l'exchange."""
        try:
            import ccxt

            exchange = ccxt.binance({
                "enableRateLimit": True,
            })

            # Convertir en milliseconds
            since_ms = int(since.timestamp() * 1000)

            all_ohlcv = []
            while True:
                ohlcv = exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.timeframe,
                    since=since_ms,
                    limit=1000,
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Dernière bougie
                last_ts = ohlcv[-1][0]
                if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000) - 7200000:
                    break

                since_ms = last_ts + 1

            if not all_ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()

    def update_data_file(self) -> Tuple[int, datetime]:
        """
        Met à jour le fichier de données.

        Returns:
            (nombre_nouvelles_barres, derniere_date)
        """
        # Charger données existantes
        if self.data_file.exists():
            existing = pd.read_csv(self.data_file, parse_dates=["timestamp"])
            existing.set_index("timestamp", inplace=True)
            last_date = existing.index.max()
        else:
            existing = pd.DataFrame()
            last_date = datetime(2011, 1, 1, tzinfo=timezone.utc)

        logger.info(f"Last data point: {last_date}")

        # Fetch nouvelles données
        new_data = self.fetch_new_data(last_date + timedelta(hours=2))

        if new_data.empty:
            logger.info("No new data available")
            return 0, last_date

        # Combiner
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Sauvegarder
        combined.to_csv(self.data_file)

        new_count = len(combined) - len(existing)
        new_last = combined.index.max()

        logger.info(f"Added {new_count} new bars, last: {new_last}")
        return new_count, new_last


class HMMRefitter:
    """
    Re-fit le modèle HMM avec nouvelles données.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def refit(self, df: pd.DataFrame) -> Tuple[Any, pd.Series]:
        """
        Re-fit le HMM et retourne le modèle + labels.

        Returns:
            (hmm_model, labels_series)
        """
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler

            # Préparer features
            features = self._compute_features(df)

            # Scaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Fit HMM
            model = hmm.GaussianHMM(
                n_components=self.k,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(features_scaled)

            # Prédire states
            states = model.predict(features_scaled)

            labels = pd.Series(states, index=df.index, name="label")

            logger.info(f"HMM refit complete: K={self.k}, {len(df)} samples")
            return model, labels

        except ImportError:
            logger.warning("hmmlearn not installed, using fallback")
            return self._fallback_clustering(df)

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule les features pour le HMM."""
        features = pd.DataFrame(index=df.index)

        # Returns
        features["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility
        features["volatility"] = features["log_return"].rolling(20).std()

        # Momentum
        features["momentum"] = df["close"].pct_change(20)

        # Volume ratio
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        features = features.dropna()
        return features.values

    def _fallback_clustering(self, df: pd.DataFrame) -> Tuple[None, pd.Series]:
        """Fallback si hmmlearn non disponible."""
        from sklearn.cluster import KMeans

        features = self._compute_features(df)

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        labels_series = pd.Series(
            labels,
            index=df.index[-len(labels):],
            name="label"
        )

        return None, labels_series


class DriftDetector:
    """
    Détecte si les params actuels dérivent (performance dégradée).
    """

    def __init__(self, threshold_sharpe_drop: float = 0.3, window_days: int = 30):
        self.threshold = threshold_sharpe_drop
        self.window_days = window_days

    def detect_drift(
        self,
        recent_trades: List[Dict],
        expected_sharpe: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Détecte si drift significatif.

        Returns:
            (is_drift, details)
        """
        if len(recent_trades) < 10:
            return False, {"reason": "not_enough_trades"}

        # Calculer Sharpe récent
        returns = [t.get("pnl_pct", 0) for t in recent_trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-10

        recent_sharpe = mean_ret / std_ret * np.sqrt(252)  # Annualisé

        # Comparer
        sharpe_drop = expected_sharpe - recent_sharpe

        is_drift = sharpe_drop > self.threshold

        return is_drift, {
            "recent_sharpe": recent_sharpe,
            "expected_sharpe": expected_sharpe,
            "sharpe_drop": sharpe_drop,
            "threshold": self.threshold,
            "n_trades": len(recent_trades),
        }


class ContinuousLearner:
    """
    Orchestrateur principal pour l'apprentissage continu.
    """

    def __init__(
        self,
        data_dir: str | Path,
        models_dir: str | Path,
        k: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.k = k

        # Composants
        self.version_mgr = ModelVersionManager(models_dir)
        self.data_updater = DataUpdater(data_dir)
        self.hmm_refitter = HMMRefitter(k=k)
        self.drift_detector = DriftDetector()

        # Config
        self.config_file = self.models_dir / "learning_config.json"
        self.load_config()

    def load_config(self) -> None:
        """Charge la configuration."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "last_data_update": None,
                "last_hmm_refit": None,
                "last_reoptimize": None,
                "update_frequency_hours": 24,
                "refit_frequency_days": 7,
                "reoptimize_frequency_days": 30,
            }

    def save_config(self) -> None:
        """Sauvegarde la configuration."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def should_update_data(self) -> bool:
        """Vérifie si mise à jour des données nécessaire."""
        last = self.config.get("last_data_update")
        if not last:
            return True

        last_dt = datetime.fromisoformat(last)
        hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600

        return hours_since >= self.config.get("update_frequency_hours", 24)

    def should_refit_hmm(self) -> bool:
        """Vérifie si refit HMM nécessaire."""
        last = self.config.get("last_hmm_refit")
        if not last:
            return True

        last_dt = datetime.fromisoformat(last)
        days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400

        return days_since >= self.config.get("refit_frequency_days", 7)

    def update_data(self) -> Dict[str, Any]:
        """Met à jour les données."""
        logger.info("Updating data...")

        new_bars, last_date = self.data_updater.update_data_file()

        self.config["last_data_update"] = datetime.now(timezone.utc).isoformat()
        self.save_config()

        return {
            "new_bars": new_bars,
            "last_date": last_date.isoformat() if last_date else None,
        }

    def refit_hmm(self) -> Dict[str, Any]:
        """Refit le modèle HMM."""
        logger.info("Refitting HMM...")

        # Charger données
        data_file = self.data_dir / "BTC_FUSED_2h.csv"
        if not data_file.exists():
            return {"error": "Data file not found"}

        df = pd.read_csv(data_file, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Refit
        model, labels = self.hmm_refitter.refit(df)

        # Sauvegarder labels
        labels_dir = self.data_dir.parent / "fourier" / "labels_frozen" / "BTC_FUSED_2h"
        labels_dir.mkdir(parents=True, exist_ok=True)

        labels_file = labels_dir / f"K{self.k}.csv"
        labels.reset_index().to_csv(labels_file, index=False)

        self.config["last_hmm_refit"] = datetime.now(timezone.utc).isoformat()
        self.save_config()

        # Distribution des phases
        dist = labels.value_counts(normalize=True).to_dict()

        return {
            "k": self.k,
            "samples": len(labels),
            "distribution": dist,
            "labels_file": str(labels_file),
        }

    def run_auto(self) -> Dict[str, Any]:
        """
        Mode automatique: effectue les mises à jour nécessaires.
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
        }

        # 1. Mise à jour données si nécessaire
        if self.should_update_data():
            data_result = self.update_data()
            results["actions"].append({"type": "update_data", **data_result})
        else:
            results["actions"].append({"type": "skip_data_update", "reason": "not_due"})

        # 2. Refit HMM si nécessaire
        if self.should_refit_hmm():
            hmm_result = self.refit_hmm()
            results["actions"].append({"type": "refit_hmm", **hmm_result})
        else:
            results["actions"].append({"type": "skip_hmm_refit", "reason": "not_due"})

        return results


def main():
    parser = argparse.ArgumentParser(description="Continuous Learning System")
    parser.add_argument("--auto", action="store_true", help="Mode automatique")
    parser.add_argument("--update-data", action="store_true", help="Mettre à jour les données")
    parser.add_argument("--refit-hmm", action="store_true", help="Refitter le HMM")
    parser.add_argument("--k", type=int, default=5, help="Nombre de phases HMM")
    args = parser.parse_args()

    print("=" * 60)
    print("CONTINUOUS LEARNING SYSTEM")
    print("=" * 60)

    learner = ContinuousLearner(
        data_dir=ROOT / "data",
        models_dir=ROOT / "outputs" / "models",
        k=args.k,
    )

    if args.auto:
        print("\nMode automatique...")
        results = learner.run_auto()
        print(json.dumps(results, indent=2, default=str))

    elif args.update_data:
        print("\nMise à jour des données...")
        results = learner.update_data()
        print(json.dumps(results, indent=2, default=str))

    elif args.refit_hmm:
        print(f"\nRefit HMM K={args.k}...")
        results = learner.refit_hmm()
        print(json.dumps(results, indent=2, default=str))

    else:
        print("\nUsage:")
        print("  python src/continuous_learning.py --auto         # Mode automatique")
        print("  python src/continuous_learning.py --update-data  # MAJ données")
        print("  python src/continuous_learning.py --refit-hmm    # Refit HMM")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
