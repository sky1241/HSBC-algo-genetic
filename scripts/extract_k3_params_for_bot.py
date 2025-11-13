#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extrait paramètres médians par phase depuis résultats K3 1D stable pour bot Binance."""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

print("="*80)
print("📊 EXTRACTION PARAMÈTRES K3 1D STABLE (Médianes par phase)")
print("="*80)

# Chercher fichiers K3 1D stable
roots = [
    Path("E:/ichimoku_runs/wfa_phase_k3_1d_stable"),
    Path("outputs/wfa_phase_k3_1d_stable")
]

files = []
for root in roots:
    if root.exists():
        found = list(root.rglob("WFA_phase_K3*.json"))
        print(f"✓ {root}: {len(found)} fichiers")
        files.extend(found)

if not files:
    print("❌ Aucun fichier WFA K3 trouvé!")
    exit(1)

print(f"\n📁 Total: {len(files)} seeds analysés")

# Collecter paramètres par phase
params_by_phase = defaultdict(lambda: {
    "tenkan": [],
    "kijun": [],
    "senkou_b": [],
    "shift": [],
    "atr_mult": [],
    "tp_mult": []  # Ajouter tp_mult si présent
})

for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        folds = data.get("folds", [])
        for fold in folds:
            params_by_state = fold.get("params_by_state", {})
            for phase_str, params in params_by_state.items():
                if phase_str == "nan" or phase_str is None:
                    continue
                try:
                    phase = int(float(phase_str))
                    if phase not in [0, 1, 2]:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Extraire tous les paramètres
                for param_name in ["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "tp_mult"]:
                    val = params.get(param_name)
                    if val is not None:
                        params_by_phase[phase][param_name].append(float(val))
    except Exception as e:
        print(f"⚠️ Erreur lecture {f.name}: {e}")
        continue

if not params_by_phase:
    print("❌ Aucun paramètre trouvé!")
    exit(1)

# Calculer médianes par phase
print("\n" + "="*80)
print("📊 PARAMÈTRES MÉDIANS PAR PHASE")
print("="*80)

phase_params = {}
for phase in sorted(params_by_phase.keys()):
    stats = {}
    for param_name in ["tenkan", "kijun", "senkou_b", "shift", "atr_mult", "tp_mult"]:
        vals = params_by_phase[phase][param_name]
        if vals:
            median_val = pd.Series(vals).median()
            stats[param_name] = float(median_val)
        else:
            # Valeur par défaut si absent
            if param_name == "tp_mult":
                stats[param_name] = 20.0  # Défaut TP
            else:
                stats[param_name] = None
    
    phase_params[str(phase)] = stats
    
    print(f"\n🎯 Phase {phase}:")
    print(f"   tenkan: {stats.get('tenkan', 'N/A'):.0f}" if stats.get('tenkan') else f"   tenkan: N/A")
    print(f"   kijun: {stats.get('kijun', 'N/A'):.0f}" if stats.get('kijun') else f"   kijun: N/A")
    print(f"   senkou_b: {stats.get('senkou_b', 'N/A'):.0f}" if stats.get('senkou_b') else f"   senkou_b: N/A")
    print(f"   shift: {stats.get('shift', 'N/A'):.0f}" if stats.get('shift') else f"   shift: N/A")
    print(f"   atr_mult: {stats.get('atr_mult', 'N/A'):.2f}" if stats.get('atr_mult') else f"   atr_mult: N/A")
    print(f"   tp_mult: {stats.get('tp_mult', 'N/A'):.2f}" if stats.get('tp_mult') else f"   tp_mult: N/A")
    print(f"   (échantillon: {len(params_by_phase[phase]['tenkan'])} valeurs)")

# Sauvegarder JSON pour bot
output_path = Path("binance_bot/configs/phase_params_K3.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(phase_params, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print(f"✅ Paramètres sauvegardés: {output_path}")
print("="*80)

# Vérifier que tp_mult est présent
missing_tp = []
for phase, params in phase_params.items():
    if params.get('tp_mult') is None:
        missing_tp.append(phase)

if missing_tp:
    print(f"\n⚠️ ATTENTION: tp_mult manquant pour phases {missing_tp}")
    print("   Valeur par défaut 20.0 utilisée. Vérifier vos analyses WFA.")

print("\n✅ Fichier prêt pour le bot Binance!")

