#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse complète: stabilité Fourier et capacité d'apprentissage."""
import json
from pathlib import Path
import pandas as pd

print("="*70)
print("🔬 ANALYSE: Les signaux Fourier varient-ils? Peut-on mieux apprendre?")
print("="*70)

# 1. Lire les paramètres médians par phase K3
settings_path = Path("docs/PHASE_ICHIMOKU_SETTINGS_K3_MEDIAN_IQR_20250924_063232.csv")
if settings_path.exists():
    df = pd.read_csv(settings_path)
    df = df[df['phase'] != 'nan']  # Exclure les NaN states
    
    print("\n📊 PARAMÈTRES MÉDIANS PAR PHASE K3:")
    print(df[['phase', 'tenkan_med', 'kijun_med', 'shift_med', 'atr_med']].to_string(index=False))
    
    print("\n🔍 IQR (Variabilité intra-phase):")
    print(df[['phase', 'tenkan_iqr', 'kijun_iqr', 'shift_iqr', 'atr_iqr']].to_string(index=False))
    
    # Calcul de la différenciation inter-phases
    print("\n" + "-"*70)
    print("📈 DIFFÉRENCIATION INTER-PHASES (range / moyenne):")
    for param in ['tenkan', 'kijun', 'shift', 'atr']:
        med_col = f"{param}_med"
        vals = df[med_col].values
        range_val = vals.max() - vals.min()
        mean_val = vals.mean()
        ratio = range_val / mean_val if mean_val > 0 else 0
        
        verdict = ""
        if ratio > 1.0:
            verdict = "✅ Forte différenciation (Fourier guide bien)"
        elif ratio > 0.5:
            verdict = "⚠️  Différenciation moyenne"
        else:
            verdict = "❌ Faible différenciation (phases similaires)"
        
        print(f"   {param:10s}: range={range_val:6.1f} / mean={mean_val:6.1f} = {ratio:.2f}  {verdict}")
    
    # Variabilité intra-phase (IQR)
    print("\n" + "-"*70)
    print("🎯 VARIABILITÉ INTRA-PHASE (IQR / médiane):")
    for param in ['tenkan', 'kijun', 'shift', 'atr']:
        med_col = f"{param}_med"
        iqr_col = f"{param}_iqr"
        ratios = []
        for _, row in df.iterrows():
            med = row[med_col]
            iqr = row[iqr_col]
            if med > 0:
                ratios.append(iqr / med)
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        
        verdict = ""
        if avg_ratio < 0.3:
            verdict = "✅ Paramètres stables (IQR < 30% de la médiane)"
        elif avg_ratio < 0.6:
            verdict = "⚠️  Variabilité modérée"
        else:
            verdict = "❌ Forte variabilité (optimisation difficile)"
        
        print(f"   {param:10s}: IQR/median moyen = {avg_ratio:.2f}  {verdict}")

# 2. Analyser la stabilité temporelle des labels (si WFA JSON disponibles)
print("\n" + "="*70)
print("📅 STABILITÉ TEMPORELLE DES SIGNAUX FOURIER")
print("="*70)

phase_roots = [Path("E:/ichimoku_runs/wfa_phase_k3"), Path("outputs/wfa_phase_k3")]
files = []
for root in phase_roots:
    if root.exists():
        files.extend(list(root.rglob("WFA_phase_*.json")))

if files:
    # Compter distribution des phases par période
    phase_dist_by_period = {}
    for f in files[:3]:  # Limiter à 3 seeds pour rapidité
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            folds = data.get("folds", [])
            for fold in folds:
                period = fold.get("period", "unknown")
                segments = fold.get("segments", [])
                if not segments:
                    continue
                
                phase_counts = {0: 0, 1: 0, 2: 0}
                for seg in segments:
                    state_str = seg.get("state", "nan")
                    if state_str != "nan":
                        try:
                            state = int(state_str)
                            phase_counts[state] += 1
                        except:
                            pass
                
                total = sum(phase_counts.values())
                if total > 0:
                    if period not in phase_dist_by_period:
                        phase_dist_by_period[period] = {0: [], 1: [], 2: []}
                    for phase in [0, 1, 2]:
                        phase_dist_by_period[period][phase].append(phase_counts[phase] / total)
        except Exception:
            continue
    
    if phase_dist_by_period:
        print("\n📊 Distribution des phases par année (moyenne sur seeds):")
        for period in sorted(phase_dist_by_period.keys()):
            dists = phase_dist_by_period[period]
            print(f"   {period}: ", end="")
            for phase in [0, 1, 2]:
                vals = dists[phase]
                avg = sum(vals) / len(vals) if vals else 0
                print(f"phase_{phase}={avg*100:5.1f}%  ", end="")
            print()
        
        # Variance inter-années
        print("\n🔍 Variance inter-années (stabilité des signaux):")
        for phase in [0, 1, 2]:
            all_vals = []
            for period_dists in phase_dist_by_period.values():
                all_vals.extend(period_dists[phase])
            if all_vals:
                mean_val = sum(all_vals) / len(all_vals)
                var_val = sum((v - mean_val)**2 for v in all_vals) / len(all_vals)
                std_val = var_val ** 0.5
                
                verdict = ""
                if std_val < 0.05:
                    verdict = "✅ Très stable (écart-type < 5%)"
                elif std_val < 0.10:
                    verdict = "⚠️  Stabilité moyenne"
                else:
                    verdict = "❌ Instable (varie beaucoup d'année en année)"
                
                print(f"   Phase {phase}: moyenne={mean_val*100:5.1f}%, écart-type={std_val*100:5.1f}%  {verdict}")
    else:
        print("⏳ Pas assez de données pour analyser la stabilité temporelle")
else:
    print("❌ Aucun fichier WFA K3 trouvé")

# 3. Recommandations
print("\n" + "="*70)
print("💡 RECOMMANDATIONS POUR AMÉLIORER L'APPRENTISSAGE")
print("="*70)
print("""
✅ CE QUI FONCTIONNE DÉJÀ:
   - Fourier évite la ruine (MDD 13% vs 100% en fixed)
   - Les paramètres ont une cohérence par phase (IQR raisonnables)

⚠️  CE QUI PEUT ÊTRE AMÉLIORÉ:

1️⃣  AUGMENTER LA DIFFÉRENCIATION INTER-PHASES:
    Si ratio < 0.5 pour certains paramètres:
    → Tester K5 ou K8 (plus de phases = discrimination plus fine)
    → Ajouter features: volatilité ATR, momentum, volume
    → Utiliser GMM (Gaussian Mixture Model) au lieu de HMM

2️⃣  RÉDUIRE LA VARIABILITÉ INTRA-PHASE:
    Si IQR/median > 0.6:
    → Augmenter trials Optuna (300 → 500-1000)
    → Ajouter contraintes: ratios kijun/tenkan fixés (2.0-3.0)
    → Optimiser sur des sous-périodes plus courtes (6 mois au lieu d'1 an)

3️⃣  AMÉLIORER LA LOSS FUNCTION:
    Actuellement Optuna optimise sur equity_mult ou Sharpe.
    → Essayer: Calmar ratio (return/MDD) pour favoriser robustesse
    → Ou: Sortino ratio (pénaliser seulement downside volatility)
    → Ou: Custom loss = 0.5*Sharpe + 0.3*Calmar + 0.2*win_rate

4️⃣  TESTER DES STRATÉGIES HYBRIDES:
    → Phase 0: conservative (ATR élevé, peu de trades)
    → Phase 1: aggressive (ATR bas, plus de trades)
    → Phase 2: neutre
    → Ajouter un "confidence score" pour chaque prédiction Fourier

5️⃣  OBJECTIF RÉALISTE:
    0.3-0.5%/mois est déjà bon pour une stratégie robuste (MDD<15%).
    Pour atteindre 5%/mois:
    → Leverage x10 (mais MDD x10 aussi → 130% = ruine probable)
    → OU combiner 10+ stratégies décorrélées (diversification)
    → OU accepter MDD 30-40% (vs actuel 13%)
""")
print("="*70)

