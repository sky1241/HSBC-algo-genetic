#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick K3 metrics extraction from WFA JSON files."""
import json
from pathlib import Path
from statistics import median

roots = [Path("E:/ichimoku_runs/wfa_phase_k3"), Path("outputs/wfa_phase_k3")]
files = []
for r in roots:
    if r.exists():
        found = list(r.rglob("WFA_*.json"))
        print(f"Root {r}: {len(found)} WFA JSON files")
        files.extend(found)
    else:
        print(f"Root {r}: does not exist")

results = []
for f in files:
    if not f.exists():
        continue
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        folds = data.get("folds", [])
        if not folds:
            continue
        results.append({
            "seed": f.parent.name,
            "file": str(f),
            "folds": [(fold["period"], fold["metrics"]["equity_mult"], fold["metrics"]["max_drawdown"], fold["metrics"]["trades"]) for fold in folds]
        })
    except Exception as e:
        print(f"Error parsing {f}: {e}")
        continue

print(f"K3 seeds terminés: {len(results)}")

seeds_metrics = []
for r in results:
    eq_final = 1.0
    for fold in r["folds"]:
        eq_final *= fold[1]
    mdd_max = max([fold[2] for fold in r["folds"]])
    trades_total = sum([fold[3] for fold in r["folds"]])
    monthly_geo = (eq_final ** (1.0 / (14 * 12))) - 1.0
    seeds_metrics.append({
        "seed": r["seed"],
        "eq_final": eq_final,
        "mdd_max": mdd_max,
        "trades_total": trades_total,
        "monthly_geo": monthly_geo
    })

ok = [s for s in seeds_metrics if s["mdd_max"] <= 0.50 and s["trades_total"] >= 280]
print(f"Pass (MDD<=50% & trades>=280): {len(ok)}/{len(seeds_metrics)}")

# Stats sur ceux qui passent
if ok:
    med_mon = median([s["monthly_geo"] for s in ok])
    med_mdd = median([s["mdd_max"] for s in ok])
    med_eq = median([s["eq_final"] for s in ok])
    print(f"Médiane (seeds OK): monthly_geo={med_mon*100:.2f}% | MDD={med_mdd*100:.1f}% | eq_final={med_eq:.2f} ({(med_eq-1)*100:.1f}%)")
else:
    print("Aucun seed ne passe les filtres.")

print("\nTop 10 seeds (par monthly_geo):")
for s in sorted(seeds_metrics, key=lambda x: x["monthly_geo"], reverse=True)[:10]:
    print(f"{s['seed']}: eq_final={s['eq_final']:.2f} ({(s['eq_final']-1)*100:.1f}%) | MDD_max={s['mdd_max']*100:.1f}% | monthly_geo={s['monthly_geo']*100:.2f}% | trades={s['trades_total']}")

