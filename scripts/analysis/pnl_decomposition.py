#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DÉCOMPOSITION PnL - Diagnostic Edge vs Exposition

Ce script analyse les résultats WFA pour déterminer si le faible rendement
vient d'un manque d'edge (alpha) ou d'une sous-exposition (levier).

Métriques analysées par année et par phase HMM:
- Nombre de trades
- Durée moyenne des trades
- Win rate et payoff ratio
- Profit factor et expectancy
- Distribution des gains (top décile)

Usage:
    python scripts/pnl_decomposition.py
    python scripts/pnl_decomposition.py --k 3  # Seulement K3
    python scripts/pnl_decomposition.py --verbose

Version: 1.0
Date: 2025-02-07
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_wfa_results(k_values=None):
    """
    Charge tous les résultats WFA disponibles.

    Returns:
        dict: {k: {seed: data}}
    """
    if k_values is None:
        k_values = [2, 3, 5, 8]

    results = {}

    for k in k_values:
        results[k] = {}
        wfa_dir = ROOT / "outputs" / f"wfa_phase_k{k}"

        if not wfa_dir.exists():
            continue

        for seed_dir in wfa_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            seed_name = seed_dir.name
            json_files = list(seed_dir.glob("WFA_phase_K*.json"))

            if json_files:
                try:
                    with open(json_files[0]) as f:
                        data = json.load(f)
                    results[k][seed_name] = data
                except Exception as e:
                    print(f"  [WARN] Erreur lecture {json_files[0]}: {e}")

    return results


def analyze_overall_metrics(results):
    """
    Analyse les métriques globales par K.
    """
    print("\n" + "=" * 70)
    print(" MÉTRIQUES GLOBALES PAR K")
    print("=" * 70)

    summary = []

    for k, seeds in sorted(results.items()):
        if not seeds:
            continue

        equity_mults = []
        drawdowns = []
        trades_list = []
        sharpes = []

        for seed_name, data in seeds.items():
            overall = data.get("overall", {})
            equity_mults.append(overall.get("equity_mult", 1.0))
            drawdowns.append(overall.get("max_drawdown", 0))
            trades_list.append(overall.get("trades", 0))
            sharpes.append(overall.get("sharpe_proxy_mean", 0))

        if equity_mults:
            # Calcul rendement mensuel
            years = 14  # 2011-2025
            avg_equity = np.median(equity_mults)
            monthly_return = (avg_equity ** (1 / (years * 12)) - 1) * 100

            summary.append({
                "K": k,
                "seeds": len(seeds),
                "equity_mult_median": np.median(equity_mults),
                "monthly_return_%": monthly_return,
                "mdd_median_%": np.median(drawdowns) * 100,
                "trades_median": np.median(trades_list),
                "trades_per_year": np.median(trades_list) / years,
            })

            print(f"\nK={k} ({len(seeds)} seeds):")
            print(f"  Equity mult (médiane): {np.median(equity_mults):.4f}")
            print(f"  Rendement mensuel:     {monthly_return:.3f}%")
            print(f"  Max Drawdown (médiane):{np.median(drawdowns)*100:.2f}%")
            print(f"  Trades (médiane):      {np.median(trades_list):.0f}")
            print(f"  Trades/an:             {np.median(trades_list)/years:.1f}")

    return summary


def analyze_by_phase(results, k=3):
    """
    Analyse détaillée par phase HMM pour un K donné.
    """
    print("\n" + "=" * 70)
    print(f" ANALYSE PAR PHASE (K={k})")
    print("=" * 70)

    if k not in results or not results[k]:
        print(f"  Pas de données pour K={k}")
        return None

    # Agrégation par phase
    phase_metrics = defaultdict(lambda: {
        "trades": [],
        "win_rate": [],
        "profit_factor": [],
        "expectancy": [],
        "avg_duration_hours": [],
        "equity_mult": [],
    })

    for seed_name, data in results[k].items():
        for fold in data.get("folds", []):
            for segment in fold.get("segments", []):
                state = segment.get("state", "?")
                metrics = segment.get("metrics", {})

                # Skip segments vides
                trades = metrics.get("trades", 0)
                if trades == 0:
                    continue

                phase_metrics[state]["trades"].append(trades)
                phase_metrics[state]["win_rate"].append(metrics.get("win_rate", 0))

                pf = metrics.get("profit_factor", 0)
                if pf != float('inf') and pf > 0:
                    phase_metrics[state]["profit_factor"].append(pf)

                phase_metrics[state]["expectancy"].append(metrics.get("expectancy", 0))

                # Durée moyenne
                avg_long = metrics.get("avg_time_long_hours", 0)
                avg_short = metrics.get("avg_time_short_hours", 0)
                if avg_long > 0 or avg_short > 0:
                    phase_metrics[state]["avg_duration_hours"].append(
                        (avg_long + avg_short) / 2 if avg_long > 0 and avg_short > 0
                        else max(avg_long, avg_short)
                    )

                em = metrics.get("equity_mult", 1.0)
                if em > 0:
                    phase_metrics[state]["equity_mult"].append(em)

    # Affichage
    print(f"\n{'Phase':<8} {'Trades':<10} {'Win%':<10} {'PF':<10} {'Expect':<12} {'Durée(h)':<10}")
    print("-" * 60)

    phase_summary = []

    for state in sorted(phase_metrics.keys()):
        m = phase_metrics[state]

        trades_sum = sum(m["trades"]) if m["trades"] else 0
        win_rate = np.median(m["win_rate"]) * 100 if m["win_rate"] else 0
        pf = np.median(m["profit_factor"]) if m["profit_factor"] else 0
        expect = np.median(m["expectancy"]) if m["expectancy"] else 0
        duration = np.median(m["avg_duration_hours"]) if m["avg_duration_hours"] else 0

        print(f"{state:<8} {trades_sum:<10.0f} {win_rate:<10.1f} {pf:<10.2f} {expect:<12.4f} {duration:<10.1f}")

        phase_summary.append({
            "phase": state,
            "total_trades": trades_sum,
            "win_rate_%": win_rate,
            "profit_factor": pf,
            "expectancy": expect,
            "avg_duration_hours": duration,
        })

    return phase_summary


def analyze_by_year(results, k=3):
    """
    Analyse par année pour K donné.
    """
    print("\n" + "=" * 70)
    print(f" ANALYSE PAR ANNÉE (K={k})")
    print("=" * 70)

    if k not in results or not results[k]:
        print(f"  Pas de données pour K={k}")
        return None

    # Agrégation par année
    year_metrics = defaultdict(lambda: {
        "trades": [],
        "equity_mult": [],
        "drawdown": [],
        "win_rate": [],
    })

    for seed_name, data in results[k].items():
        for fold in data.get("folds", []):
            year = fold.get("period", "?")

            # Agréger les segments du fold
            total_trades = 0
            equity_mults = []
            win_rates = []

            for segment in fold.get("segments", []):
                metrics = segment.get("metrics", {})
                total_trades += metrics.get("trades", 0)

                em = metrics.get("equity_mult", 1.0)
                if em > 0:
                    equity_mults.append(em)

                wr = metrics.get("win_rate", 0)
                if metrics.get("trades", 0) > 0:
                    win_rates.append(wr)

            year_metrics[year]["trades"].append(total_trades)
            if equity_mults:
                # Equity mult du fold = produit des segments
                year_metrics[year]["equity_mult"].append(np.prod(equity_mults))
            if win_rates:
                year_metrics[year]["win_rate"].append(np.mean(win_rates))

    # Affichage
    print(f"\n{'Année':<8} {'Trades':<10} {'Equity×':<12} {'Win%':<10} {'Contrib%':<10}")
    print("-" * 50)

    year_summary = []
    total_pnl = 0

    for year in sorted(year_metrics.keys()):
        m = year_metrics[year]

        trades = np.median(m["trades"]) if m["trades"] else 0
        equity = np.median(m["equity_mult"]) if m["equity_mult"] else 1.0
        win_rate = np.median(m["win_rate"]) * 100 if m["win_rate"] else 0

        # Contribution = log return (approximation)
        contrib = (equity - 1) * 100
        total_pnl += contrib

        print(f"{year:<8} {trades:<10.0f} {equity:<12.4f} {win_rate:<10.1f} {contrib:>+9.2f}%")

        year_summary.append({
            "year": year,
            "trades": trades,
            "equity_mult": equity,
            "win_rate_%": win_rate,
            "contribution_%": contrib,
        })

    print("-" * 50)
    print(f"{'TOTAL':<8} {'':<10} {'':<12} {'':<10} {total_pnl:>+9.2f}%")

    return year_summary


def diagnose_edge_vs_exposure(results, k=3):
    """
    Diagnostic principal : edge ou exposition ?
    """
    print("\n" + "=" * 70)
    print(" DIAGNOSTIC : EDGE vs EXPOSITION")
    print("=" * 70)

    if k not in results or not results[k]:
        print(f"  Pas de données pour K={k}")
        return

    # Collecte des métriques agrégées
    all_win_rates = []
    all_profit_factors = []
    all_expectancies = []
    all_trades = []
    all_durations = []

    for seed_name, data in results[k].items():
        overall = data.get("overall", {})
        all_trades.append(overall.get("trades", 0))

        for fold in data.get("folds", []):
            for segment in fold.get("segments", []):
                metrics = segment.get("metrics", {})

                if metrics.get("trades", 0) > 0:
                    all_win_rates.append(metrics.get("win_rate", 0))

                    pf = metrics.get("profit_factor", 0)
                    if pf != float('inf') and pf > 0:
                        all_profit_factors.append(pf)

                    all_expectancies.append(metrics.get("expectancy", 0))

                    avg_long = metrics.get("avg_time_long_hours", 0)
                    avg_short = metrics.get("avg_time_short_hours", 0)
                    if avg_long > 0 or avg_short > 0:
                        all_durations.append(max(avg_long, avg_short))

    # Calculs
    median_win_rate = np.median(all_win_rates) * 100 if all_win_rates else 0
    median_pf = np.median(all_profit_factors) if all_profit_factors else 0
    median_expect = np.median(all_expectancies) if all_expectancies else 0
    median_trades = np.median(all_trades) if all_trades else 0
    median_duration = np.median(all_durations) if all_durations else 0

    trades_per_year = median_trades / 14

    print("\n1. MÉTRIQUES D'EDGE:")
    print(f"   Win Rate médian:      {median_win_rate:.1f}%")
    print(f"   Profit Factor médian: {median_pf:.2f}")
    print(f"   Expectancy médiane:   {median_expect:.4f}")

    print("\n2. MÉTRIQUES D'EXPOSITION:")
    print(f"   Trades/an:            {trades_per_year:.1f}")
    print(f"   Durée moyenne trade:  {median_duration:.1f} heures")

    # Diagnostic
    print("\n3. DIAGNOSTIC:")

    issues = []

    # Critères d'edge
    if median_win_rate < 50:
        issues.append(f"   [!] Win Rate < 50% ({median_win_rate:.1f}%) -> Edge faible")
    else:
        print(f"   [OK] Win Rate OK ({median_win_rate:.1f}%)")

    if median_pf < 1.2:
        issues.append(f"   [!] Profit Factor < 1.2 ({median_pf:.2f}) -> Edge marginal")
    elif median_pf < 1.5:
        print(f"   [~] Profit Factor modeste ({median_pf:.2f})")
    else:
        print(f"   [OK] Profit Factor bon ({median_pf:.2f})")

    if median_expect < 0.001:
        issues.append(f"   [!] Expectancy tres faible ({median_expect:.4f})")
    else:
        print(f"   [OK] Expectancy positive ({median_expect:.4f})")

    # Critères d'exposition
    if trades_per_year < 50:
        issues.append(f"   [!] Trop peu de trades ({trades_per_year:.0f}/an) -> Sous-expose")
    else:
        print(f"   [OK] Nombre de trades OK ({trades_per_year:.0f}/an)")

    if median_duration > 100:
        issues.append(f"   [!] Trades trop longs ({median_duration:.0f}h) -> Capital immobilise")
    else:
        print(f"   [OK] Duree trades OK ({median_duration:.0f}h)")

    for issue in issues:
        print(issue)

    # Conclusion
    print("\n4. CONCLUSION:")

    edge_issues = sum(1 for i in issues if "Edge" in i or "Expectancy" in i)
    exposure_issues = sum(1 for i in issues if "Sous-exposé" in i or "immobilisé" in i)

    if edge_issues > exposure_issues:
        print("   [EDGE] PROBLEME PRINCIPAL: EDGE")
        print("   -> Le signal n'a pas assez d'alpha")
        print("   -> Augmenter le levier ne resoudra PAS le probleme")
        print("   -> Il faut de NOUVEAUX SIGNAUX (mixture-of-experts, mean-reversion)")
        recommendation = "edge"
    elif exposure_issues > edge_issues:
        print("   [EXPO] PROBLEME PRINCIPAL: EXPOSITION")
        print("   -> L'edge existe mais on est sous-expose")
        print("   -> Le VOLATILITY TARGETING devrait aider")
        print("   -> Tester avec levier plus eleve")
        recommendation = "exposure"
    else:
        print("   [MIX] PROBLEME MIXTE: EDGE + EXPOSITION")
        print("   -> Edge marginal ET sous-exposition")
        print("   -> Volatility targeting en premier (plus simple)")
        print("   -> Puis mixture-of-experts si insuffisant")
        recommendation = "mixed"

    return recommendation


def main():
    parser = argparse.ArgumentParser(description="Décomposition PnL")
    parser.add_argument("--k", type=int, default=None, help="K spécifique à analyser")
    parser.add_argument("--verbose", action="store_true", help="Afficher plus de détails")
    args = parser.parse_args()

    print("=" * 70)
    print(" DÉCOMPOSITION PnL - DIAGNOSTIC EDGE VS EXPOSITION")
    print("=" * 70)

    # Charger les résultats
    print("\nChargement des résultats WFA...")
    k_values = [args.k] if args.k else [2, 3, 5, 8]
    results = load_wfa_results(k_values)

    total_seeds = sum(len(seeds) for seeds in results.values())
    print(f"  {total_seeds} seeds chargés")

    # Analyses
    analyze_overall_metrics(results)

    # Pour K=3 (le plus complet)
    k_analyze = args.k if args.k else 3
    analyze_by_phase(results, k=k_analyze)
    analyze_by_year(results, k=k_analyze)

    # Diagnostic final
    recommendation = diagnose_edge_vs_exposure(results, k=k_analyze)

    print("\n" + "=" * 70)
    print(" PROCHAINE ÉTAPE RECOMMANDÉE")
    print("=" * 70)

    if recommendation == "edge":
        print("""
  Le problème est l'EDGE, pas l'exposition.

  → NE PAS implémenter volatility targeting en priorité
  → Implémenter MIXTURE-OF-EXPERTS:
    - Garder Ichimoku pour phases trend
    - Ajouter mean-reversion pour phases chop
    - Soft-switching via probabilités HMM
""")
    elif recommendation == "exposure":
        print("""
  Le problème est l'EXPOSITION, l'edge existe.

  → Implémenter VOLATILITY TARGETING:
    - leverage = min(L_max, sigma_target / sigma_realized)
    - Drawdown throttle quand DD > seuil

  → Rerun un subset de seeds (5-10) avec vol targeting
  → Vérifier si rendement scale linéairement
""")
    else:
        print("""
  Problème MIXTE: edge marginal + sous-exposition.

  → Étape 1: VOLATILITY TARGETING (plus rapide à implémenter)
    - Tester si le rendement peut scaler

  → Étape 2: Si insuffisant, MIXTURE-OF-EXPERTS
    - Nouveaux signaux pour phases non-trend
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
