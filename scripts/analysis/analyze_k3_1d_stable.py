#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse complète K3 1D stable (30 seeds): médiane, IQR, distribution, survie."""
import json
from pathlib import Path
from statistics import median
import pandas as pd
from datetime import datetime

print("="*80)
print("📊 ANALYSE COMPLÈTE: K3 1D Stable (30 seeds)")
print("="*80)

# Charger tous les résultats K3 1D stable
roots = [
    Path("E:/ichimoku_runs/wfa_phase_k3_1d_stable"),
    Path("outputs/wfa_phase_k3_1d_stable")
]

files = []
for root in roots:
    if root.exists():
        found = list(root.rglob("WFA_phase_*.json"))
        print(f"✓ Root {root}: {len(found)} fichiers JSON")
        files.extend(found)

if not files:
    print("❌ Aucun fichier WFA trouvé!")
    exit(1)

print(f"\n📁 Total: {len(files)} seeds")

# Extraire métriques
results = []
for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        folds = data.get("folds", [])
        if not folds:
            continue
        
        # Calculer métriques agrégées
        eq_final = 1.0
        mdd_max = 0.0
        trades_total = 0
        sharpe_vals = []
        
        for fold in folds:
            eq_fold = fold["metrics"]["equity_mult"]
            eq_final *= eq_fold
            mdd_max = max(mdd_max, fold["metrics"]["max_drawdown"])
            trades_total += fold["metrics"]["trades"]
            sharpe_vals.append(fold["metrics"].get("sharpe_proxy_mean", 0.0))
        
        monthly_geo = (eq_final ** (1.0 / (14 * 12))) - 1.0
        sharpe_mean = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0
        cagr = (eq_final ** (1.0 / 14)) - 1.0
        
        seed_name = f.parent.name.replace("seed_", "")
        results.append({
            "seed": seed_name,
            "eq_final": eq_final,
            "eq_pct": (eq_final - 1.0) * 100.0,
            "mdd_max": mdd_max,
            "mdd_pct": mdd_max * 100.0,
            "trades_total": trades_total,
            "monthly_geo": monthly_geo,
            "monthly_pct": monthly_geo * 100.0,
            "sharpe_mean": sharpe_mean,
            "cagr": cagr,
            "cagr_pct": cagr * 100.0,
            "folds": len(folds),
            "file": str(f)
        })
    except Exception as e:
        print(f"⚠️  Erreur parsing {f}: {e}")
        continue

if not results:
    print("❌ Aucun résultat valide extrait!")
    exit(1)

print(f"\n✅ {len(results)} seeds analysés")

# Créer DataFrame pour calculs statistiques
df = pd.DataFrame(results)

# Filtre: MDD <= 50% & trades >= 280
df_ok = df[(df["mdd_max"] <= 0.50) & (df["trades_total"] >= 280)].copy()

print("\n" + "="*80)
print("📈 SURVIE & ROBUSTESSE")
print("="*80)
print(f"Seeds terminés: {len(df)}")
print(f"Pass (MDD<=50% & trades>=280): {len(df_ok)}/{len(df)} ({len(df_ok)/len(df)*100:.1f}%)")
print(f"Ruine (MDD>50%): {len(df[df['mdd_max'] > 0.50])} ({len(df[df['mdd_max'] > 0.50])/len(df)*100:.1f}%)")
print(f"Trop peu trades (<280): {len(df[df['trades_total'] < 280])}")

if len(df_ok) == 0:
    print("\n❌ AUCUN SEED NE PASSE LES FILTRES!")
    print("\n📊 Tous les seeds (sans filtre):")
    print(f"   Médiane monthly: {median(df['monthly_pct']):.2f}%")
    print(f"   Médiane MDD: {median(df['mdd_pct']):.1f}%")
    print(f"   Médiane trades: {median(df['trades_total']):.0f}")
    exit(1)

# Statistiques sur seeds OK
print("\n" + "="*80)
print("📊 MÉTRIQUES AGRÉGÉES (Seeds Pass)")
print("="*80)

# Calculer statistiques manuellement
def calc_stats(col):
    return {
        "median": df_ok[col].median(),
        "q25": df_ok[col].quantile(0.25),
        "q75": df_ok[col].quantile(0.75)
    }

stats_mon = calc_stats("monthly_pct")
stats_mdd = calc_stats("mdd_pct")
stats_trades = calc_stats("trades_total")
stats_eq = calc_stats("eq_pct")
stats_sharpe = calc_stats("sharpe_mean")
stats_cagr = calc_stats("cagr_pct")

print("\n🎯 RENDEMENT MENSUEL:")
print(f"   Médiane: {stats_mon['median']:.2f}%")
print(f"   Q1: {stats_mon['q25']:.2f}%")
print(f"   Q3: {stats_mon['q75']:.2f}%")
iqr_mon = stats_mon['q75'] - stats_mon['q25']
print(f"   IQR: {iqr_mon:.2f}%")

print("\n📉 MAX DRAWDOWN:")
print(f"   Médiane: {stats_mdd['median']:.1f}%")
print(f"   Q1: {stats_mdd['q25']:.1f}%")
print(f"   Q3: {stats_mdd['q75']:.1f}%")
iqr_mdd = stats_mdd['q75'] - stats_mdd['q25']
print(f"   IQR: {iqr_mdd:.1f}%")

print("\n🔄 TRADES TOTAL (14 ans):")
print(f"   Médiane: {stats_trades['median']:.0f}")
print(f"   Q1: {stats_trades['q25']:.0f}")
print(f"   Q3: {stats_trades['q75']:.0f}")
iqr_trades = stats_trades['q75'] - stats_trades['q25']
print(f"   IQR: {iqr_trades:.0f}")

print("\n💰 EQUITY FINALE (multiplicateur):")
print(f"   Médiane: {stats_eq['median']:.1f}% (+{(stats_eq['median']/100+1):.2f}x)")
print(f"   Q1: {stats_eq['q25']:.1f}%")
print(f"   Q3: {stats_eq['q75']:.1f}%")

print("\n📊 SHARPE PROXY (moyenne):")
print(f"   Médiane: {stats_sharpe['median']:.2f}")
print(f"   Q1: {stats_sharpe['q25']:.2f}")
print(f"   Q3: {stats_sharpe['q75']:.2f}")

print("\n🚀 CAGR (14 ans):")
print(f"   Médiane: {stats_cagr['median']:.2f}%")
print(f"   Q1: {stats_cagr['q25']:.2f}%")
print(f"   Q3: {stats_cagr['q75']:.2f}%")

# Top seeds
print("\n" + "="*80)
print("🏆 TOP 10 SEEDS (par monthly_geo)")
print("="*80)
top10 = df_ok.nlargest(10, "monthly_pct")[["seed", "monthly_pct", "mdd_pct", "trades_total", "eq_pct", "sharpe_mean"]]
for idx, row in top10.iterrows():
    print(f"   {row['seed']:>6s}: monthly={row['monthly_pct']:6.2f}% | MDD={row['mdd_pct']:5.1f}% | trades={row['trades_total']:4.0f} | equity={row['eq_pct']:7.1f}% | Sharpe={row['sharpe_mean']:5.2f}")

# Objectif 5%/mois
print("\n" + "="*80)
print("🎯 OBJECTIF: 5% MENSUEL")
print("="*80)
target = 5.0
med_monthly = stats_mon['median']
pct_ok = (df_ok['monthly_pct'] >= target).sum()
print(f"   Médiane actuelle: {med_monthly:.2f}%")
print(f"   Seeds >= 5%/mois: {pct_ok}/{len(df_ok)} ({pct_ok/len(df_ok)*100:.1f}%)")
if med_monthly >= target:
    print(f"   ✅✅ OBJECTIF ATTEINT!")
elif med_monthly >= target * 0.8:
    print(f"   ✅ Proche objectif ({(target-med_monthly)/target*100:.0f}% à atteindre)")
else:
    pct_obj = (med_monthly / target) * 100
    print(f"   📊 {pct_obj:.0f}% de l'objectif atteint")

# Export CSV
out_csv = Path("docs/K3_1D_STABLE_ANALYSIS.csv")
df_ok.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\n💾 Résultats exportés: {out_csv}")

print("\n" + "="*80)

