#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze Fourier signal stability and parameter consistency across phases."""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 1. Analyse des labels K3 (stabilité Fourier année par année)
labels_path = Path("data/BTC_FUSED_2h_K3.csv")
if not labels_path.exists():
    print(f"❌ Labels K3 introuvables: {labels_path}")
else:
    df = pd.read_csv(labels_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    
    print("="*60)
    print("📊 STABILITÉ DES SIGNAUX FOURIER K3 (par année)")
    print("="*60)
    
    by_year = df.groupby('year')['label'].value_counts(normalize=True).unstack(fill_value=0)
    print("\nDistribution des phases (labels) par année:")
    print(by_year.round(3))
    
    # Variance de distribution
    phase_variance = by_year.var(axis=0)
    print(f"\n📈 Variance de distribution par phase (0={phase_variance[0]:.3f}, 1={phase_variance[1]:.3f}, 2={phase_variance[2]:.3f})")
    if phase_variance.max() < 0.02:
        print("✅ Distribution très stable année par année (variance < 2%)")
    elif phase_variance.max() < 0.05:
        print("⚠️  Distribution moyennement stable (variance 2-5%)")
    else:
        print("❌ Distribution instable (variance > 5%), signaux Fourier varient beaucoup")

# 2. Analyse des paramètres optimaux par phase (cohérence)
print("\n" + "="*60)
print("🔧 COHÉRENCE DES PARAMÈTRES ICHIMOKU PAR PHASE")
print("="*60)

phase_roots = [Path("E:/ichimoku_runs/wfa_phase_k3"), Path("outputs/wfa_phase_k3")]
files = []
for root in phase_roots:
    if root.exists():
        files.extend(list(root.rglob("WFA_phase_*.json")))

if not files:
    print("❌ Aucun fichier WFA K3 phase trouvé")
else:
    # Collecter tous les paramètres par phase et par année
    params_by_phase = defaultdict(lambda: defaultdict(list))
    
    for f in files[:5]:  # Limiter à 5 seeds pour vitesse
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            folds = data.get("folds", [])
            for fold in folds:
                period = fold.get("period", "unknown")
                params_by_state = fold.get("params_by_state", {})
                for state_str, params in params_by_state.items():
                    if state_str == "nan":
                        continue
                    state = int(state_str)
                    for param_name in ["tenkan", "kijun", "senkou_b", "shift", "atr_mult"]:
                        val = params.get(param_name)
                        if val is not None:
                            params_by_phase[state][param_name].append(float(val))
        except Exception as e:
            continue
    
    if params_by_phase:
        print("\nParamètres moyens et écart-type par phase (K3):")
        for state in sorted(params_by_phase.keys()):
            print(f"\n🔹 Phase {state}:")
            for param_name in ["tenkan", "kijun", "shift", "atr_mult"]:
                vals = params_by_phase[state].get(param_name, [])
                if vals:
                    mean_val = sum(vals) / len(vals)
                    std_val = (sum((v - mean_val)**2 for v in vals) / len(vals))**0.5
                    cv = (std_val / mean_val) if mean_val > 0 else 0
                    consistency = "✅ stable" if cv < 0.3 else ("⚠️  variable" if cv < 0.6 else "❌ très variable")
                    print(f"   {param_name:12s}: {mean_val:6.1f} ± {std_val:5.1f}  (CV={cv:.2f}) {consistency}")
        
        # Variance inter-phase
        print("\n" + "-"*60)
        print("📊 Comparaison inter-phases (variance des moyennes):")
        for param_name in ["tenkan", "kijun", "shift", "atr_mult"]:
            means = []
            for state in sorted(params_by_phase.keys()):
                vals = params_by_phase[state].get(param_name, [])
                if vals:
                    means.append(sum(vals) / len(vals))
            if means:
                mean_of_means = sum(means) / len(means)
                var_means = sum((m - mean_of_means)**2 for m in means) / len(means)
                cv_inter = (var_means**0.5 / mean_of_means) if mean_of_means > 0 else 0
                print(f"   {param_name:12s}: CV inter-phase = {cv_inter:.2f}  ", end="")
                if cv_inter > 0.5:
                    print("✅ Forte différenciation entre phases (Fourier guide bien)")
                elif cv_inter > 0.2:
                    print("⚠️  Différenciation moyenne")
                else:
                    print("❌ Peu de différence entre phases (Fourier n'apporte pas grand-chose)")

print("\n" + "="*60)
print("💡 RECOMMANDATIONS POUR AMÉLIORER L'APPRENTISSAGE")
print("="*60)
print("""
1️⃣  Si les signaux Fourier varient peu année par année (variance < 5%):
    ✅ Les labels sont fiables, on peut faire confiance à l'optimisation

2️⃣  Si les paramètres optimaux varient beaucoup au sein d'une phase (CV > 0.6):
    ⚠️  Élargir les ranges Optuna ou ajouter des contraintes (ex: tenkan/kijun ratios)

3️⃣  Si la différenciation inter-phase est faible (CV < 0.2):
    ❌ Fourier n'aide pas assez → tester K5/K8 ou ajouter d'autres features (volatilité, volume)

4️⃣  Pour améliorer l'apprentissage:
    a) Augmenter le nombre de trials Optuna (300 → 500+)
    b) Utiliser un loss function plus agressif (favoriser high Sharpe/Calmar)
    c) Ajouter des features secondaires (ATR ratio, momentum) pour mieux discriminer les phases
    d) Tester un modèle de régime plus fin (GMM au lieu de HMM, ou K=5/8)
    e) Optimiser par sous-phase (si phase 2 dure 6 mois, découper en 2×3 mois)
""")
print("="*60)

