#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGGREGATE K5 RESULTS - Agrège les résultats des 30 seeds K5

Ce script:
1. Parcourt tous les résultats des 30 seeds
2. Extrait les params optimaux par phase
3. Calcule les médianes pour robustesse
4. Génère aggregated_params.json pour le live trading

Usage:
    python scripts/aggregate_k5_results.py

Après exécution de: .\\scripts\\launch_30_seeds_k5.ps1

Version: 1.0
Date: 2025-02-03
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_seed_results(seed_dir: Path) -> Dict[str, Any] | None:
    """Charge les résultats d'un seed."""
    # Chercher le fichier JSON principal
    json_files = list(seed_dir.glob("WFA_*.json"))

    if not json_files:
        # Essayer de reconstruire depuis les fichiers intermédiaires
        return load_from_intermediate_files(seed_dir)

    try:
        with open(json_files[0], "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {json_files[0]}: {e}")
        return None


def load_from_intermediate_files(seed_dir: Path) -> Dict[str, Any] | None:
    """Reconstruit les résultats depuis fichiers intermédiaires."""
    params_files = list(seed_dir.glob("*_params_*.json"))
    metrics_files = list(seed_dir.glob("*_metrics_*.json"))

    if not params_files:
        return None

    # Reconstruire
    result = {
        "params_by_phase": {},
        "metrics_by_phase": {},
    }

    for pf in params_files:
        try:
            with open(pf, "r") as f:
                data = json.load(f)
            phase = data.get("phase", pf.stem.split("_")[-1])
            result["params_by_phase"][str(phase)] = data
        except Exception:
            continue

    return result if result["params_by_phase"] else None


def aggregate_params(all_results: List[Dict]) -> Dict[str, Dict]:
    """
    Agrège les params de tous les seeds par phase.

    Utilise la médiane pour robustesse.
    """
    # Collecter params par phase
    params_by_phase: Dict[str, List[Dict]] = {}

    for result in all_results:
        phase_params = result.get("params_by_phase", {})
        for phase_id, params in phase_params.items():
            if phase_id not in params_by_phase:
                params_by_phase[phase_id] = []
            params_by_phase[phase_id].append(params)

    # Calculer médianes
    aggregated = {}

    for phase_id, params_list in params_by_phase.items():
        if not params_list:
            continue

        # Convertir en DataFrame pour faciliter le calcul
        df = pd.DataFrame(params_list)

        # Extraire les colonnes de params
        param_cols = ["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "tp_mult"]
        available_cols = [c for c in param_cols if c in df.columns]

        if not available_cols:
            continue

        median_params = {}
        iqr_params = {}

        for col in available_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) > 0:
                median_params[col] = float(values.median())
                q1, q3 = values.quantile([0.25, 0.75])
                iqr_params[col] = {"q1": float(q1), "median": float(values.median()), "q3": float(q3)}

        # Arrondir les entiers
        for col in ["tenkan", "kijun", "senkou_b", "shift"]:
            if col in median_params:
                median_params[col] = int(round(median_params[col]))

        # Ajouter métadonnées
        median_params["n_seeds"] = len(params_list)
        median_params["confidence"] = len(params_list) / len(all_results)
        median_params["iqr"] = iqr_params

        aggregated[phase_id] = median_params

    return aggregated


def aggregate_metrics(all_results: List[Dict]) -> Dict[str, Any]:
    """Agrège les métriques de performance."""
    metrics = {
        "equity_mult": [],
        "max_drawdown": [],
        "trades": [],
        "sharpe": [],
        "cagr": [],
    }

    for result in all_results:
        summary = result.get("summary", result.get("metrics", {}))
        for key in metrics.keys():
            if key in summary:
                metrics[key].append(float(summary[key]))

    # Calculer stats
    aggregated = {}
    for key, values in metrics.items():
        if values:
            arr = np.array(values)
            aggregated[key] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q10": float(np.percentile(arr, 10)),
                "q90": float(np.percentile(arr, 90)),
            }

    return aggregated


def main():
    print("=" * 60)
    print("AGGREGATE K5 RESULTS")
    print("=" * 60)

    # Dossier des résultats
    results_dir = ROOT / "outputs" / "wfa_phase_k5"

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("\nRun first: .\\scripts\\launch_30_seeds_k5.ps1")
        return 1

    # Charger tous les seeds
    print(f"\nScanning: {results_dir}")

    all_results = []
    seed_dirs = sorted(results_dir.glob("seed_*"))

    print(f"Found {len(seed_dirs)} seed directories\n")

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        result = load_seed_results(seed_dir)

        if result:
            all_results.append(result)
            n_phases = len(result.get("params_by_phase", {}))
            print(f"  [OK] {seed_name}: {n_phases} phases")
        else:
            print(f"  [--] {seed_name}: no results found")

    if not all_results:
        print("\nERROR: No valid results found!")
        print("Wait for seeds to complete or check for errors.")
        return 1

    print(f"\n{len(all_results)} seeds loaded successfully")

    # Agréger params
    print("\nAggregating parameters by phase...")
    aggregated_params = aggregate_params(all_results)

    # Agréger métriques
    print("Aggregating performance metrics...")
    aggregated_metrics = aggregate_metrics(all_results)

    # Construire output final
    output = {
        "metadata": {
            "aggregation_date": datetime.now(timezone.utc).isoformat(),
            "n_seeds": len(all_results),
            "k": 5,
            "source_dir": str(results_dir),
        },
        "phases": aggregated_params,
        "performance": aggregated_metrics,
    }

    # Sauvegarder
    output_path = results_dir / "aggregated_params.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nOutput: {output_path}")

    # Afficher résumé
    print(f"\nPhases trouvées: {list(aggregated_params.keys())}")

    print("\nParams médians par phase:")
    for phase_id, params in aggregated_params.items():
        print(f"\n  Phase {phase_id} ({params.get('n_seeds', 0)} seeds, conf={params.get('confidence', 0):.1%}):")
        print(f"    tenkan={params.get('tenkan')}, kijun={params.get('kijun')}, senkou_b={params.get('senkou_b')}")
        print(f"    shift={params.get('shift')}, atr_mult={params.get('atr_mult', 0):.1f}, tp_mult={params.get('tp_mult', 0):.1f}")

    if aggregated_metrics:
        print("\nPerformance agrégée:")
        for key, stats in aggregated_metrics.items():
            print(f"  {key}: median={stats['median']:.4f}, Q10={stats['q10']:.4f}, Q90={stats['q90']:.4f}")

    print(f"\n{'=' * 60}")
    print("Pour utiliser ces params en live:")
    print(f"  python src/live_trader_adaptive.py --testnet --params {output_path}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
