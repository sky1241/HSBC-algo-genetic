#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare phase-adapted vs fixed Ichimoku on K3."""
import json
from pathlib import Path
from statistics import median, mean

def load_wfa_metrics(root_paths, pattern="WFA_*.json"):
    """Load all WFA JSON files from given roots and extract metrics."""
    results = []
    for root in root_paths:
        if not root.exists():
            continue
        files = list(root.rglob(pattern))
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                folds = data.get("folds", [])
                if not folds:
                    continue
                eq_final = 1.0
                for fold in folds:
                    eq_final *= fold["metrics"]["equity_mult"]
                mdd_max = max([fold["metrics"]["max_drawdown"] for fold in folds])
                trades_total = sum([fold["metrics"]["trades"] for fold in folds])
                monthly_geo = (eq_final ** (1.0 / (14 * 12))) - 1.0
                results.append({
                    "seed": f.parent.name,
                    "file": str(f),
                    "eq_final": eq_final,
                    "mdd_max": mdd_max,
                    "trades_total": trades_total,
                    "monthly_geo": monthly_geo
                })
            except Exception as e:
                print(f"Error parsing {f}: {e}")
                continue
    return results

# K3 phase-adapted
phase_roots = [Path("E:/ichimoku_runs/wfa_phase_k3"), Path("outputs/wfa_phase_k3")]
phase_results = load_wfa_metrics(phase_roots)
phase_ok = [r for r in phase_results if r["mdd_max"] <= 0.50 and r["trades_total"] >= 280]

# K3 fixed (classic Ichimoku optimized but no phase adaptation)
fixed_roots = [Path("outputs/wfa_fixed_k3")]
fixed_results = load_wfa_metrics(fixed_roots, pattern="WFA_annual_*.json")
fixed_ok = [r for r in fixed_results if r["mdd_max"] <= 0.50 and r["trades_total"] >= 280]

print("="*60)
print("COMPARAISON: K3 Phase-Adapté vs K3 Fixed (Ichimoku classique)")
print("="*60)

print(f"\n📊 K3 PHASE-ADAPTÉ (Fourier):")
print(f"   Seeds terminés: {len(phase_results)} | Pass (MDD<=50% & trades>=280): {len(phase_ok)}/{len(phase_results)}")
if phase_ok:
    med_mon = median([r["monthly_geo"] for r in phase_ok])
    med_mdd = median([r["mdd_max"] for r in phase_ok])
    med_eq = median([r["eq_final"] for r in phase_ok])
    best = max(phase_ok, key=lambda x: x["monthly_geo"])
    print(f"   Médiane: monthly={med_mon*100:.2f}% | MDD={med_mdd*100:.1f}% | equity_14y={med_eq:.2f}x (+{(med_eq-1)*100:.0f}%)")
    print(f"   Meilleur: {best['seed']} => monthly={best['monthly_geo']*100:.2f}% | MDD={best['mdd_max']*100:.1f}% | equity={best['eq_final']:.2f}x (+{(best['eq_final']-1)*100:.0f}%)")

print(f"\n📈 K3 FIXED (Ichimoku classique optimisé):")
print(f"   Seeds terminés: {len(fixed_results)} | Pass (MDD<=50% & trades>=280): {len(fixed_ok)}/{len(fixed_results)}")
if fixed_ok:
    med_mon = median([r["monthly_geo"] for r in fixed_ok])
    med_mdd = median([r["mdd_max"] for r in fixed_ok])
    med_eq = median([r["eq_final"] for r in fixed_ok])
    best = max(fixed_ok, key=lambda x: x["monthly_geo"])
    print(f"   Médiane: monthly={med_mon*100:.2f}% | MDD={med_mdd*100:.1f}% | equity_14y={med_eq:.2f}x (+{(med_eq-1)*100:.0f}%)")
    print(f"   Meilleur: {best['seed']} => monthly={best['monthly_geo']*100:.2f}% | MDD={best['mdd_max']*100:.1f}% | equity={best['eq_final']:.2f}x (+{(best['eq_final']-1)*100:.0f}%)")
else:
    print("   ❌ Aucun seed ne passe les filtres.")

print("\n" + "="*60)
if phase_ok and fixed_ok:
    phase_med = median([r["monthly_geo"] for r in phase_ok])
    fixed_med = median([r["monthly_geo"] for r in fixed_ok])
    gain = ((phase_med / fixed_med) - 1.0) * 100.0 if fixed_med > 0 else 0
    print(f"🎯 VERDICT: Phase-adapté vs Fixed")
    print(f"   Gain médian mensuel: {gain:+.1f}%")
    if gain > 10:
        print(f"   ✅ Les signaux Fourier AMÉLIORENT significativement Ichimoku (+{gain:.0f}%)")
    elif gain > 0:
        print(f"   ⚠️  Les signaux Fourier améliorent légèrement Ichimoku (+{gain:.1f}%), mais peu significatif")
    else:
        print(f"   ❌ Les signaux Fourier N'APPORTENT PAS de gain (voire dégradent: {gain:.1f}%)")
        print(f"   💡 Recommendation: rester sur Ichimoku classique optimisé (9-26-52 ou variant)")
    
    # Objectif 5%/mois
    target = 0.05
    phase_vs_target = (phase_med / target) * 100.0
    fixed_vs_target = (fixed_med / target) * 100.0
    print(f"\n📌 Objectif 5%/mois (réaliste):")
    print(f"   Phase-adapté atteint {phase_vs_target:.0f}% de l'objectif ({phase_med*100:.2f}%/mois)")
    print(f"   Fixed atteint {fixed_vs_target:.0f}% de l'objectif ({fixed_med*100:.2f}%/mois)")
    if phase_med >= 0.05:
        print(f"   ✅ Phase-adapté VALIDE l'objectif 5%/mois!")
    elif fixed_med >= 0.05:
        print(f"   ✅ Fixed VALIDE l'objectif 5%/mois!")
    else:
        print(f"   ❌ Aucune approche n'atteint 5%/mois pour l'instant (attendre plus de seeds)")
else:
    print("⏳ Pas assez de données pour conclure. Attendre plus de seeds terminés.")
print("="*60)

