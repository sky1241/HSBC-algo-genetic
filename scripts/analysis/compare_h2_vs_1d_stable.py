#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare K3 H2 pur vs K3 1D stable + H2 trading."""
import json
from pathlib import Path
from statistics import median

def load_wfa_metrics(root_paths, pattern="WFA_*.json"):
    """Load all WFA JSON files and extract metrics."""
    results = []
    for root in root_paths:
        if isinstance(root, str):
            root = Path(root)
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
                continue
    return results

print("="*70)
print("📊 COMPARAISON: K3 H2 Pur vs K3 1D Stable + H2 Trading")
print("="*70)

# K3 H2 pur (labels changent toutes les 2h)
h2_roots = [Path("E:/ichimoku_runs/wfa_phase_k3"), Path("outputs/wfa_phase_k3")]
h2_results = load_wfa_metrics(h2_roots, pattern="WFA_phase_*.json")
h2_ok = [r for r in h2_results if r["mdd_max"] <= 0.50 and r["trades_total"] >= 280]

# K3 1D stable (labels changent max 1×/jour)
stable_roots = [Path("E:/ichimoku_runs/wfa_phase_k3_1d_stable"), Path("outputs/wfa_phase_k3_1d_stable")]
stable_results = load_wfa_metrics(stable_roots, pattern="WFA_phase_*.json")
stable_ok = [r for r in stable_results if r["mdd_max"] <= 0.50 and r["trades_total"] >= 280]

print(f"\n🔵 K3 H2 PUR (labels 2h natifs):")
print(f"   Seeds terminés: {len(h2_results)} | Pass (MDD<=50% & trades>=280): {len(h2_ok)}/{len(h2_results)}")
if h2_ok:
    med_mon = median([r["monthly_geo"] for r in h2_ok])
    med_mdd = median([r["mdd_max"] for r in h2_ok])
    med_trades = median([r["trades_total"] for r in h2_ok])
    med_eq = median([r["eq_final"] for r in h2_ok])
    best = max(h2_ok, key=lambda x: x["monthly_geo"])
    print(f"   Médiane: monthly={med_mon*100:.2f}% | MDD={med_mdd*100:.1f}% | trades={med_trades:.0f} | equity={med_eq:.2f}x")
    print(f"   Meilleur: {best['seed']} => monthly={best['monthly_geo']*100:.2f}% | equity={best['eq_final']:.2f}x")

print(f"\n🟢 K3 1D STABLE + H2 TRADING (labels journaliers):")
print(f"   Seeds terminés: {len(stable_results)} | Pass (MDD<=50% & trades>=280): {len(stable_ok)}/{len(stable_results)}")
if stable_ok:
    med_mon = median([r["monthly_geo"] for r in stable_ok])
    med_mdd = median([r["mdd_max"] for r in stable_ok])
    med_trades = median([r["trades_total"] for r in stable_ok])
    med_eq = median([r["eq_final"] for r in stable_ok])
    best = max(stable_ok, key=lambda x: x["monthly_geo"])
    print(f"   Médiane: monthly={med_mon*100:.2f}% | MDD={med_mdd*100:.1f}% | trades={med_trades:.0f} | equity={med_eq:.2f}x")
    print(f"   Meilleur: {best['seed']} => monthly={best['monthly_geo']*100:.2f}% | equity={best['eq_final']:.2f}x")
elif len(stable_results) > 0:
    print(f"   ⏳ Seeds en cours, attendre fin pour métriques")
else:
    print(f"   ❌ Aucun seed terminé encore")

print("\n" + "="*70)

if h2_ok and stable_ok:
    h2_mon = median([r["monthly_geo"] for r in h2_ok])
    stable_mon = median([r["monthly_geo"] for r in stable_ok])
    h2_trades = median([r["trades_total"] for r in h2_ok])
    stable_trades = median([r["trades_total"] for r in stable_ok])
    
    gain_mon = ((stable_mon / h2_mon) - 1.0) * 100.0 if h2_mon > 0 else 0
    gain_trades = ((stable_trades / h2_trades) - 1.0) * 100.0 if h2_trades > 0 else 0
    
    print(f"🎯 VERDICT: 1D Stable vs H2 Pur")
    print(f"   Gain rendement mensuel: {gain_mon:+.1f}%")
    print(f"   Gain nombre trades: {gain_trades:+.1f}%")
    print()
    
    if gain_mon > 30 and gain_trades > 50:
        print(f"   ✅✅ SUCCÈS MAJEUR: 1D stable améliore significativement rendement ET trades!")
        print(f"   → RECOMMANDATION: Lancer 30 seeds complets + étendre à K5/K8")
    elif gain_mon > 10:
        print(f"   ✅ SUCCÈS: 1D stable améliore le rendement")
        print(f"   → RECOMMANDATION: Valider avec 10 seeds supplémentaires avant full run")
    elif gain_mon > 0:
        print(f"   ⚠️  AMÉLIORATION LÉGÈRE: Gain faible mais positif")
        print(f"   → RECOMMANDATION: Analyser en détail avant décision")
    else:
        print(f"   ❌ ÉCHEC: 1D stable ne surpasse pas H2 pur")
        print(f"   → RECOMMANDATION: Rester sur H2 pur ou tester méthode rolling")
    
    # Objectif 5%/mois
    target = 0.05
    if stable_mon >= target:
        print(f"\n   🎉 OBJECTIF 5%/MOIS ATTEINT: {stable_mon*100:.2f}%!")
    elif stable_mon >= target * 0.5:
        print(f"\n   📈 Proche objectif: {stable_mon*100:.2f}% (à {(target-stable_mon)*100:.2f}% de 5%)")
    else:
        pct_obj = (stable_mon / target) * 100
        print(f"\n   📊 Objectif 5%/mois: {pct_obj:.0f}% atteint ({stable_mon*100:.2f}%/mois)")

elif h2_ok:
    print(f"⏳ Test 1D stable en cours. Attendre résultats pour comparaison.")
    print(f"   H2 pur baseline établie: {len(h2_ok)} seeds OK")
else:
    print(f"⏳ Pas assez de données pour comparaison. Attendre fin des runs.")

print("="*70)

