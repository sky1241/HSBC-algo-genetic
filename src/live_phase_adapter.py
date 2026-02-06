#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIVE PHASE ADAPTER - Auto-apprentissage et adaptation en temps réel

Ce module:
1. Détecte la phase de marché actuelle (HMM K5)
2. Charge les paramètres optimisés pour cette phase
3. Adapte automatiquement la stratégie Ichimoku
4. Gère les transitions entre phases

Usage:
    adapter = LivePhaseAdapter(k=5)
    adapter.load_learned_params("outputs/wfa_phase_k5/aggregated_params.json")

    # En boucle live:
    current_params = adapter.get_current_params(latest_ohlcv_df)
    # -> Utiliser current_params pour le trading

Version: 1.0
Date: 2025-02-03
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LivePhaseAdapter")


@dataclass
class PhaseParams:
    """Paramètres Ichimoku pour une phase spécifique."""
    phase_id: str
    tenkan: int
    kijun: int
    senkou_b: int
    shift: int
    atr_mult: float
    tp_mult: float
    confidence: float = 1.0  # Confiance basée sur le nombre de trades historiques

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "tenkan": self.tenkan,
            "kijun": self.kijun,
            "senkou_b": self.senkou_b,
            "shift": self.shift,
            "atr_mult": self.atr_mult,
            "tp_mult": self.tp_mult,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseParams":
        return cls(
            phase_id=str(d.get("phase_id", "0")),
            tenkan=int(d.get("tenkan", 9)),
            kijun=int(d.get("kijun", 26)),
            senkou_b=int(d.get("senkou_b", 52)),
            shift=int(d.get("shift", 26)),
            atr_mult=float(d.get("atr_mult", 14.0)),
            tp_mult=float(d.get("tp_mult", 10.0)),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class PhaseTransition:
    """Enregistrement d'une transition de phase."""
    timestamp: datetime
    from_phase: str
    to_phase: str
    confidence: float


@dataclass
class AdapterState:
    """État interne de l'adapter."""
    current_phase: str = "0"
    phase_start_time: Optional[datetime] = None
    transition_history: List[PhaseTransition] = field(default_factory=list)
    bars_in_phase: int = 0
    last_update: Optional[datetime] = None


class HMMPhaseDetector:
    """
    Détecteur de phase basé sur HMM pré-entraîné.

    Utilise les features spectrales pour classifier la phase actuelle.
    """

    def __init__(self, k: int = 5, lookback: int = 100):
        """
        Args:
            k: Nombre de phases (K du HMM)
            lookback: Nombre de barres pour le calcul des features
        """
        self.k = k
        self.lookback = lookback
        self.hmm_model = None
        self.scaler = None
        self._feature_cols = [
            "log_return", "volatility", "momentum",
            "rsi_norm", "volume_ratio"
        ]

    def load_model(self, model_path: Path) -> bool:
        """Charge le modèle HMM pré-entraîné."""
        try:
            import joblib
            model_data = joblib.load(model_path)
            self.hmm_model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            logger.info(f"HMM model loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Could not load HMM model: {e}")
            return False

    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule les features pour la détection de phase."""
        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} bars, got {len(df)}")

        # Prendre les dernières barres
        recent = df.tail(self.lookback).copy()

        # Log returns
        recent["log_return"] = np.log(recent["close"] / recent["close"].shift(1))

        # Volatility (rolling std of returns)
        recent["volatility"] = recent["log_return"].rolling(20).std()

        # Momentum (ROC)
        recent["momentum"] = recent["close"].pct_change(20)

        # RSI normalized
        delta = recent["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        recent["rsi_norm"] = (100 - 100 / (1 + rs)) / 100  # Normalize to 0-1

        # Volume ratio
        recent["volume_ratio"] = recent["volume"] / recent["volume"].rolling(20).mean()

        # Prendre la dernière ligne
        features = recent[self._feature_cols].iloc[-1:].values

        return features

    def detect_phase(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Détecte la phase actuelle du marché.

        Returns:
            Tuple[phase_id, confidence]
        """
        features = self.compute_features(df)

        if self.hmm_model is not None and self.scaler is not None:
            # Utiliser le HMM pré-entraîné
            features_scaled = self.scaler.transform(features)
            phase = self.hmm_model.predict(features_scaled)[0]
            # Calculer la probabilité
            proba = self.hmm_model.predict_proba(features_scaled)[0]
            confidence = float(proba[phase])
        else:
            # Fallback: classification simple basée sur volatilité/momentum
            phase, confidence = self._simple_phase_detection(features[0])

        return str(phase), confidence

    def _simple_phase_detection(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Classification simple sans HMM (fallback).
        Basée sur volatilité et momentum.
        """
        log_ret, vol, mom, rsi, vol_ratio = features

        # Règles simples pour K=5
        if vol > 0.03:  # Haute volatilité
            if mom > 0.05:
                return 4, 0.7  # Bull volatile
            elif mom < -0.05:
                return 3, 0.7  # Bear volatile
            else:
                return 2, 0.6  # Choppy
        else:  # Basse volatilité
            if mom > 0.02:
                return 1, 0.8  # Trend up calm
            elif mom < -0.02:
                return 0, 0.8  # Trend down calm
            else:
                return 2, 0.7  # Range/consolidation

        return 2, 0.5  # Default


class LivePhaseAdapter:
    """
    Adaptateur principal pour le trading live avec phases.

    Gère:
    - Détection de phase en temps réel
    - Chargement des params appris
    - Transitions entre phases
    - Logging et monitoring
    """

    def __init__(
        self,
        k: int = 5,
        min_bars_before_switch: int = 3,
        transition_smoothing: bool = True,
    ):
        """
        Args:
            k: Nombre de phases HMM
            min_bars_before_switch: Barres minimum avant changement de phase
            transition_smoothing: Lissage des transitions (évite flip-flop)
        """
        self.k = k
        self.min_bars_before_switch = min_bars_before_switch
        self.transition_smoothing = transition_smoothing

        # Composants
        self.detector = HMMPhaseDetector(k=k)
        self.state = AdapterState()

        # Params par phase (chargés depuis les résultats WFA)
        self.phase_params: Dict[str, PhaseParams] = {}

        # Params par défaut (Ichimoku classique)
        self.default_params = PhaseParams(
            phase_id="default",
            tenkan=9, kijun=26, senkou_b=52, shift=26,
            atr_mult=14.0, tp_mult=10.0, confidence=0.5
        )

        # Historique pour smoothing
        self._recent_phases: List[str] = []
        self._phase_history_size = 5

    def load_learned_params(self, params_path: str | Path) -> bool:
        """
        Charge les paramètres appris depuis les résultats WFA.

        Le fichier doit contenir les params médians par phase, issus
        de l'agrégation des 30 seeds.
        """
        path = Path(params_path)
        if not path.exists():
            logger.error(f"Params file not found: {path}")
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Format attendu: {"phases": {"0": {...}, "1": {...}, ...}}
            phases_data = data.get("phases", data)

            for phase_id, params in phases_data.items():
                self.phase_params[str(phase_id)] = PhaseParams.from_dict({
                    "phase_id": phase_id,
                    **params
                })

            logger.info(f"Loaded params for {len(self.phase_params)} phases from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading params: {e}")
            return False

    def load_hmm_model(self, model_path: str | Path) -> bool:
        """Charge le modèle HMM pour la détection de phase."""
        return self.detector.load_model(Path(model_path))

    def get_current_params(self, df: pd.DataFrame) -> PhaseParams:
        """
        Obtient les paramètres pour la phase actuelle.

        Args:
            df: DataFrame OHLCV récent (au moins 100 barres)

        Returns:
            PhaseParams pour la phase détectée
        """
        # Détecter la phase
        detected_phase, confidence = self.detector.detect_phase(df)

        # Appliquer smoothing si activé
        if self.transition_smoothing:
            smoothed_phase = self._smooth_phase_transition(detected_phase)
        else:
            smoothed_phase = detected_phase

        # Vérifier si on doit changer de phase
        should_switch = self._should_switch_phase(smoothed_phase)

        if should_switch:
            self._record_transition(smoothed_phase, confidence)

        # Mettre à jour l'état
        self.state.last_update = datetime.now(timezone.utc)
        self.state.bars_in_phase += 1

        # Retourner les params pour la phase courante
        current_phase = self.state.current_phase

        if current_phase in self.phase_params:
            params = self.phase_params[current_phase]
            logger.debug(f"Phase {current_phase}: using learned params")
        else:
            params = self.default_params
            logger.warning(f"Phase {current_phase}: no learned params, using default")

        return params

    def _smooth_phase_transition(self, detected_phase: str) -> str:
        """
        Lisse les transitions pour éviter les flip-flops.

        Utilise un vote majoritaire sur les dernières détections.
        """
        self._recent_phases.append(detected_phase)
        if len(self._recent_phases) > self._phase_history_size:
            self._recent_phases.pop(0)

        # Vote majoritaire
        from collections import Counter
        counts = Counter(self._recent_phases)
        majority_phase, count = counts.most_common(1)[0]

        # Besoin d'une majorité claire pour changer
        if count >= (self._phase_history_size // 2 + 1):
            return majority_phase

        # Sinon garder la phase actuelle
        return self.state.current_phase

    def _should_switch_phase(self, new_phase: str) -> bool:
        """Détermine si on doit changer de phase."""
        if new_phase == self.state.current_phase:
            return False

        # Vérifier le nombre minimum de barres
        if self.state.bars_in_phase < self.min_bars_before_switch:
            return False

        return True

    def _record_transition(self, new_phase: str, confidence: float) -> None:
        """Enregistre une transition de phase."""
        now = datetime.now(timezone.utc)

        transition = PhaseTransition(
            timestamp=now,
            from_phase=self.state.current_phase,
            to_phase=new_phase,
            confidence=confidence,
        )

        self.state.transition_history.append(transition)

        logger.info(
            f"PHASE TRANSITION: {self.state.current_phase} -> {new_phase} "
            f"(confidence: {confidence:.2%})"
        )

        # Mettre à jour l'état
        self.state.current_phase = new_phase
        self.state.phase_start_time = now
        self.state.bars_in_phase = 0

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel de l'adapter."""
        current_params = self.phase_params.get(
            self.state.current_phase,
            self.default_params
        )

        return {
            "current_phase": self.state.current_phase,
            "bars_in_phase": self.state.bars_in_phase,
            "phase_start": self.state.phase_start_time.isoformat() if self.state.phase_start_time else None,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
            "total_transitions": len(self.state.transition_history),
            "current_params": current_params.to_dict(),
            "loaded_phases": list(self.phase_params.keys()),
        }

    def export_state(self, path: str | Path) -> None:
        """Exporte l'état pour persistence."""
        state_dict = {
            "current_phase": self.state.current_phase,
            "bars_in_phase": self.state.bars_in_phase,
            "phase_start_time": self.state.phase_start_time.isoformat() if self.state.phase_start_time else None,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
            "transitions": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "from": t.from_phase,
                    "to": t.to_phase,
                    "confidence": t.confidence,
                }
                for t in self.state.transition_history
            ],
            "recent_phases": self._recent_phases,
        }

        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"State exported to {path}")

    def import_state(self, path: str | Path) -> bool:
        """Importe l'état depuis une sauvegarde."""
        try:
            with open(path, "r") as f:
                state_dict = json.load(f)

            self.state.current_phase = state_dict.get("current_phase", "0")
            self.state.bars_in_phase = state_dict.get("bars_in_phase", 0)

            if state_dict.get("phase_start_time"):
                self.state.phase_start_time = datetime.fromisoformat(
                    state_dict["phase_start_time"]
                )

            if state_dict.get("last_update"):
                self.state.last_update = datetime.fromisoformat(
                    state_dict["last_update"]
                )

            self._recent_phases = state_dict.get("recent_phases", [])

            logger.info(f"State imported from {path}")
            return True

        except Exception as e:
            logger.error(f"Error importing state: {e}")
            return False


def aggregate_wfa_results(wfa_results_dir: Path, output_path: Path) -> Dict[str, Any]:
    """
    Agrège les résultats WFA de 30 seeds pour extraire les params médians par phase.

    Args:
        wfa_results_dir: Dossier contenant les résultats (outputs/wfa_phase_k5/)
        output_path: Chemin pour sauvegarder les params agrégés

    Returns:
        Dict avec params médians par phase
    """
    all_params: Dict[str, List[Dict]] = {}

    # Parcourir tous les seeds
    for seed_dir in sorted(wfa_results_dir.glob("seed_*")):
        # Chercher le fichier de résultats
        result_files = list(seed_dir.glob("WFA_*.json"))
        if not result_files:
            continue

        with open(result_files[0], "r") as f:
            result = json.load(f)

        # Extraire params par phase
        params_by_phase = result.get("params_by_phase", {})
        for phase_id, params in params_by_phase.items():
            if phase_id not in all_params:
                all_params[phase_id] = []
            all_params[phase_id].append(params)

    # Calculer médiane par phase
    aggregated = {"phases": {}, "metadata": {
        "n_seeds": len(list(wfa_results_dir.glob("seed_*"))),
        "aggregation_date": datetime.now(timezone.utc).isoformat(),
    }}

    for phase_id, params_list in all_params.items():
        if not params_list:
            continue

        df = pd.DataFrame(params_list)
        median_params = {
            "tenkan": int(df["tenkan"].median()),
            "kijun": int(df["kijun"].median()),
            "senkou_b": int(df["senkou_b"].median()),
            "shift": int(df["shift"].median()),
            "atr_mult": float(df["atr_mult"].median()),
            "tp_mult": float(df["tp_mult"].median()),
            "confidence": float(len(params_list) / aggregated["metadata"]["n_seeds"]),
        }
        aggregated["phases"][phase_id] = median_params

    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Aggregated params for {len(aggregated['phases'])} phases -> {output_path}")
    return aggregated


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LIVE PHASE ADAPTER - Demo")
    print("=" * 60)

    # 1. Créer l'adapter
    adapter = LivePhaseAdapter(k=5, min_bars_before_switch=3)

    # 2. Charger les params appris (si disponibles)
    params_path = Path("outputs/wfa_phase_k5/aggregated_params.json")
    if params_path.exists():
        adapter.load_learned_params(params_path)
    else:
        print(f"\nNo learned params found at {params_path}")
        print("Using default Ichimoku params.")
        print("\nTo generate learned params:")
        print("  1. Run: .\\scripts\\launch_30_seeds_k5.ps1")
        print("  2. Wait for completion (~20-30 hours)")
        print("  3. Run aggregation to create aggregated_params.json")

    # 3. Simuler avec données de test
    print("\n--- Simulation avec données fictives ---")

    # Créer des données fictives pour la démo
    np.random.seed(42)
    n_bars = 150
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="2h")

    # Simuler un prix BTC avec tendance
    price = 40000 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))

    demo_df = pd.DataFrame({
        "open": price * (1 + np.random.randn(n_bars) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
        "low": price * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
        "close": price,
        "volume": np.random.rand(n_bars) * 1000 + 500,
    }, index=dates)

    # 4. Tester la détection de phase
    print("\nDétection de phase sur dernières barres...")
    params = adapter.get_current_params(demo_df)

    print(f"\nPhase détectée: {adapter.state.current_phase}")
    print(f"Paramètres Ichimoku:")
    print(f"  - Tenkan:   {params.tenkan}")
    print(f"  - Kijun:    {params.kijun}")
    print(f"  - Senkou B: {params.senkou_b}")
    print(f"  - Shift:    {params.shift}")
    print(f"  - ATR mult: {params.atr_mult}")
    print(f"  - TP mult:  {params.tp_mult}")
    print(f"  - Confidence: {params.confidence:.2%}")

    # 5. Afficher le statut
    print("\n--- Status ---")
    status = adapter.get_status()
    for key, value in status.items():
        if key != "current_params":
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Pour utiliser en live:")
    print("  1. Lancer les 30 seeds K5")
    print("  2. Agréger les résultats")
    print("  3. Intégrer dans le bot Binance")
    print("=" * 60)
