#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION VOLATILITY TARGETING

Simule l'impact du volatility targeting sur les resultats WFA existants.
Permet de voir si augmenter le levier ameliorerait les rendements.

Usage:
    python scripts/simulate_volatility_targeting.py
    python scripts/simulate_volatility_targeting.py --k 5 --leverage 3

Version: 1.0
Date: 2025-02-07
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_best_seed(k=5):
    """Charge le meilleur seed pour K donne."""
    wfa_dir = ROOT / "outputs" / f"wfa_phase_k{k}"

    if not wfa_dir.exists():
        print(f"Dossier {wfa_dir} non trouve")
        return None

    best_data = None
    best_equity = 0

    for seed_dir in wfa_dir.iterdir():
        if not seed_dir.is_dir():
            continue

        json_files = list(seed_dir.glob("WFA_phase_K*.json"))
        if not json_files:
            continue

        try:
            with open(json_files[0]) as f:
                data = json.load(f)

            equity = data.get("overall", {}).get("equity_mult", 0)
            if equity > best_equity:
                best_equity = equity
                best_data = data
        except:
            continue

    return best_data


def extract_phase_returns(data):
    """
    Extrait les returns par phase depuis les donnees WFA.

    Returns:
        dict: {phase: {"returns": [...], "expectancy": float, "trades": int}}
    """
    phase_data = {}

    for fold in data.get("folds", []):
        for segment in fold.get("segments", []):
            phase = segment.get("state", "?")
            metrics = segment.get("metrics", {})

            trades = metrics.get("trades", 0)
            if trades == 0:
                continue

            equity_mult = metrics.get("equity_mult", 1.0)
            expectancy = metrics.get("expectancy", 0)

            if phase not in phase_data:
                phase_data[phase] = {
                    "returns": [],
                    "expectancy_sum": 0,
                    "trades_total": 0,
                    "equity_mults": [],
                }

            # Approximation du return par segment
            segment_return = equity_mult - 1.0
            phase_data[phase]["returns"].append(segment_return)
            phase_data[phase]["expectancy_sum"] += expectancy * trades
            phase_data[phase]["trades_total"] += trades
            phase_data[phase]["equity_mults"].append(equity_mult)

    # Calcul expectancy moyenne ponderee
    for phase, d in phase_data.items():
        if d["trades_total"] > 0:
            d["avg_expectancy"] = d["expectancy_sum"] / d["trades_total"]
        else:
            d["avg_expectancy"] = 0

    return phase_data


def simulate_leverage(phase_data, leverage_by_phase):
    """
    Simule l'application d'un levier par phase.

    Args:
        phase_data: dict des donnees par phase
        leverage_by_phase: dict {phase: leverage}

    Returns:
        dict: resultats simules
    """
    total_return = 0
    max_dd = 0
    equity = 1.0
    peak = 1.0

    # Simuler sur tous les segments
    all_returns = []

    for phase, d in phase_data.items():
        lev = leverage_by_phase.get(phase, 1.0)

        for ret in d["returns"]:
            leveraged_ret = ret * lev
            all_returns.append(leveraged_ret)

            equity *= (1 + leveraged_ret)
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

    return {
        "final_equity": equity,
        "total_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "avg_return_per_segment": np.mean(all_returns) * 100 if all_returns else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Simulation Volatility Targeting")
    parser.add_argument("--k", type=int, default=5, help="K a analyser")
    parser.add_argument("--leverage", type=float, default=1.0, help="Levier uniforme")
    args = parser.parse_args()

    print("=" * 70)
    print(" SIMULATION VOLATILITY TARGETING")
    print("=" * 70)

    # Charger les donnees
    print(f"\nChargement meilleur seed K={args.k}...")
    data = load_best_seed(k=args.k)

    if not data:
        print("Aucune donnee trouvee!")
        return 1

    overall = data.get("overall", {})
    print(f"  Equity original: {overall.get('equity_mult', 0):.4f}")
    print(f"  MDD original:    {overall.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Trades:          {overall.get('trades', 0)}")

    # Extraire returns par phase
    print("\nExtraction des returns par phase...")
    phase_data = extract_phase_returns(data)

    print(f"\n{'Phase':<8} {'Trades':<10} {'Expectancy':<12} {'Segments':<10}")
    print("-" * 40)
    for phase in sorted(phase_data.keys()):
        d = phase_data[phase]
        print(f"{phase:<8} {d['trades_total']:<10} {d['avg_expectancy']:<12.4f} {len(d['returns']):<10}")

    # Identifier phases gagnantes (expectancy > 0)
    winning_phases = [p for p, d in phase_data.items() if d["avg_expectancy"] > 0]
    losing_phases = [p for p, d in phase_data.items() if d["avg_expectancy"] <= 0]

    print(f"\nPhases GAGNANTES: {winning_phases}")
    print(f"Phases PERDANTES: {losing_phases}")

    # Simulations
    print("\n" + "=" * 70)
    print(" SIMULATIONS")
    print("=" * 70)

    scenarios = [
        ("Baseline (L=1)", {p: 1.0 for p in phase_data}),
        ("Levier uniforme L=2", {p: 2.0 for p in phase_data}),
        ("Levier uniforme L=3", {p: 3.0 for p in phase_data}),
        ("Levier uniforme L=5", {p: 5.0 for p in phase_data}),
        ("SMART: L=3 gagnantes, L=0.5 perdantes",
         {p: 3.0 if p in winning_phases else 0.5 for p in phase_data}),
        ("SMART: L=5 gagnantes, L=0 perdantes (no trade)",
         {p: 5.0 if p in winning_phases else 0.0 for p in phase_data}),
        ("SMART: L=10 gagnantes, L=0 perdantes",
         {p: 10.0 if p in winning_phases else 0.0 for p in phase_data}),
    ]

    print(f"\n{'Scenario':<45} {'Equity':<12} {'Return%':<12} {'MDD%':<10}")
    print("-" * 80)

    results = []
    for name, leverage_map in scenarios:
        result = simulate_leverage(phase_data, leverage_map)
        print(f"{name:<45} {result['final_equity']:<12.4f} {result['total_return_pct']:<+12.2f} {result['max_drawdown_pct']:<10.2f}")
        results.append((name, result))

    # Calcul rendement mensuel
    print("\n" + "=" * 70)
    print(" RENDEMENT MENSUEL ESTIME (14 ans = 168 mois)")
    print("=" * 70)

    months = 14 * 12
    print(f"\n{'Scenario':<45} {'Monthly%':<12} {'Annuel%':<12}")
    print("-" * 70)

    for name, result in results:
        equity = result['final_equity']
        if equity > 0:
            monthly = (equity ** (1/months) - 1) * 100
            annual = (equity ** (1/14) - 1) * 100
        else:
            monthly = -100
            annual = -100
        print(f"{name:<45} {monthly:<+12.3f} {annual:<+12.2f}")

    # Conclusion
    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)

    baseline = results[0][1]
    smart_l5 = results[5][1]  # L=5 gagnantes, L=0 perdantes

    improvement = (smart_l5['final_equity'] / baseline['final_equity'] - 1) * 100

    print(f"""
  Baseline (sans levier):
    - Equity: {baseline['final_equity']:.4f}
    - Monthly: {(baseline['final_equity']**(1/months)-1)*100:.3f}%

  SMART (L=5 sur phases gagnantes, L=0 sur perdantes):
    - Equity: {smart_l5['final_equity']:.4f}
    - Monthly: {(smart_l5['final_equity']**(1/months)-1)*100:.3f}%
    - MDD: {smart_l5['max_drawdown_pct']:.1f}%

  Amelioration: {improvement:+.1f}%
""")

    if smart_l5['max_drawdown_pct'] < 50:
        print("  [OK] Le volatility targeting SMART peut ameliorer les resultats!")
        print("  [OK] Le MDD reste acceptable (<50%)")
        print("\n  RECOMMANDATION: Implementer vol targeting avec:")
        print(f"    - Levier 5x sur phases: {winning_phases}")
        print(f"    - Levier 0x sur phases: {losing_phases}")
    else:
        print("  [!] Le MDD devient trop eleve avec ce levier")
        print("  [!] Reduire le levier ou ameliorer l'edge d'abord")

    return 0


if __name__ == "__main__":
    sys.exit(main())
