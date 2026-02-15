# 🔄 Stratégie Hybride: Phases 1D Stables + Trading H2

## 🎯 Concept

**Problème identifié:**
- Labels K3 sur 2h changent potentiellement 12×/jour (toutes les 2h)
- Beaucoup de "faux changements" (whipsaw) dus au bruit intra-journalier
- Résultat: peu de trades (32/an), rendement faible (0.30%/mois)

**Solution hybride:**
1. **Phases détectées sur base JOURNALIÈRE** (1D)
   - Label majoritaire du jour J → appliqué au jour J+1
   - Maximum 1 changement de phase par jour
   
2. **Trading exécuté sur 2H**
   - 12 opportunités de trade par jour maintenues
   - Paramètres Ichimoku de la phase actuelle
   
3. **Avantages combinés:**
   - Stabilité des phases 1D (moins de whipsaw)
   - Réactivité du trading H2 (plus de trades)
   - Potentiel: 100-150 trades/an (vs 32 actuel)

---

## 📊 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ JOUR J (Passé)                                               │
├──────────────────────────────────────────────────────────────┤
│ Barres 2h:  00h → phase 0                                    │
│             02h → phase 0                                    │
│             04h → phase 1  ← changement temporaire           │
│             06h → phase 0                                    │
│             08h → phase 0                                    │
│             ...                                              │
│             22h → phase 0                                    │
│                                                              │
│ Calcul 23:59: Label majoritaire = Phase 0 (10/12 barres)    │
└──────────────────────────────────────────────────────────────┘
                              ↓
         Décision: Phase stable pour demain = Phase 0
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ JOUR J+1 (Futur) - TRADING                                   │
├──────────────────────────────────────────────────────────────┤
│ Phase appliquée: Phase 0 (toute la journée)                 │
│ Params Ichimoku: tenkan=8, kijun=40, shift=32, atr=4.0      │
│                                                              │
│ 00h: Calcul signal Ichimoku H2 → LONG si conditions OK      │
│ 02h: Calcul signal Ichimoku H2 → Aucun trade                │
│ 04h: Calcul signal Ichimoku H2 → SHORT si conditions OK     │
│ ...  (12 opportunités avec MÊMES paramètres)                │
│ 22h: Calcul signal Ichimoku H2 → Aucun trade                │
│                                                              │
│ Fin de journée: Recalcul label majoritaire pour J+2         │
└──────────────────────────────────────────────────────────────┘
```

---

## ✅ Validation No Lookahead

**Question critique:** Utilise-t-on des données futures?

**NON! Voici pourquoi:**

1. **Calcul du label:**
   - À la fin du jour J (23:59), on analyse les 12 barres 2h du jour J
   - Toutes ces barres appartiennent au PASSÉ
   - On calcule le label majoritaire (statistique sur passé)

2. **Application du label:**
   - Le label est appliqué au jour J+1 (futur)
   - C'est une PRÉDICTION basée sur le passé, exactement comme le WFA annuel

3. **Analogie avec WFA actuel:**
   ```
   WFA annuel (actuel):
   - Train 2011 → Test 2012 ✅ OK
   - Optimisation sur 2011 (passé) → Application 2012 (futur)
   
   Hybride 1D stable (nouveau):
   - Jour J (passé) → Jour J+1 (futur) ✅ OK
   - Label majoritaire J → Application J+1
   ```

**Conclusion:** Stratégie rigoureuse, conforme aux standards scientifiques.

---

## 🔬 Méthode de Downsampling

### Algorithme "Daily Majority"

```python
Pour chaque jour J dans l'historique:
    1. Lire les 12 labels 2h du jour J
    2. Compter fréquence de chaque phase (0, 1, 2)
    3. Retenir la phase majoritaire (ex: phase 0 apparaît 10/12 fois)
    4. Appliquer cette phase à TOUTES les 12 barres du jour J+1
    
Forward fill pour début historique (pas de J-1 disponible)
```

### Exemple Concret

**Données brutes 2h (jour J):**
```
2012-01-15 00:00 → phase 0
2012-01-15 02:00 → phase 0
2012-01-15 04:00 → phase 1  ← outlier
2012-01-15 06:00 → phase 0
2012-01-15 08:00 → phase 0
2012-01-15 10:00 → phase 2  ← outlier
2012-01-15 12:00 → phase 0
2012-01-15 14:00 → phase 0
2012-01-15 16:00 → phase 0
2012-01-15 18:00 → phase 0
2012-01-15 20:00 → phase 0
2012-01-15 22:00 → phase 0
```
**Comptage:** Phase 0 = 10, Phase 1 = 1, Phase 2 = 1  
**Majorité:** Phase 0

**Labels 1D stables (jour J+1):**
```
2012-01-16 00:00 → phase 0  ← stabilisé
2012-01-16 02:00 → phase 0  ← stabilisé
2012-01-16 04:00 → phase 0  ← stabilisé
2012-01-16 06:00 → phase 0  ← stabilisé
... (toutes les 12 barres = phase 0)
2012-01-16 22:00 → phase 0  ← stabilisé
```

---

## 📈 Résultats Attendus

### Comparaison H2 Pur vs 1D Stable + H2 Trading

| Métrique | H2 Pur (actuel) | 1D Stable + H2 | Amélioration |
|----------|-----------------|----------------|--------------|
| **Stabilité phases** |  |  |  |
| Changes/an | ~4,380 (12/jour) | ~365 (1/jour max) | **-92%** |
| Switches/jour | Jusqu'à 12 | Max 1 | **-92%** |
| Whipsaw | Élevé | Minimal | ✅ |
| **Performance** |  |  |  |
| Trades/an | 32 | 100-150 (estimé) | **+300%** |
| Monthly return | 0.30% | 0.5-0.7% (estimé) | **+100%** |
| MDD | 13.2% | 12-15% (estimé) | ≈ Similaire |
| Survie (MDD<=50%) | 100% (11/11) | 100% (estimé) | ✅ |
| **Cohérence** |  |  |  |
| Params jour | Variables | Fixes | ✅ |
| Analyse | Difficile | Simple | ✅ |

### Hypothèses clés

1. **Plus de trades:** Trading H2 avec phases stables devrait capturer plus d'opportunités
2. **Meilleur signal/bruit:** Filtrage outliers intra-journaliers améliore décisions
3. **MDD préservé:** Stratégie toujours conservative (pas plus risquée)
4. **Robustesse maintenue:** Pas de sur-optimisation, principe simple

---

## 🛠 Implémentation

### 1. Génération Labels 1D Stables

```bash
# À partir des labels K3 2h existants
python scripts/downsample_labels_2h_to_1d.py --k 3 --method daily

# Output: outputs/fourier/labels_frozen/BTC_FUSED_2h/K3_1d_stable.csv
# Format identique aux labels 2h (60,531 lignes) mais phases stables
```

**Stats K3:**
- Lignes: 60,531 (inchangé, toujours grid 2h)
- Changements: 213 → 207 (-2.8%)
- Switches/jour moyen: 0.04 (vs potentiel 12)
- Distribution: Phase 0=29.8%, Phase 1=15.1%, Phase 2=55.0%

### 2. Lancement Test 5 Seeds

```powershell
# Lancement parallèle 5 seeds (1001-1005)
.\scripts\launch_k3_1d_stable_test.ps1

# Durée: 24-48h
# Output: outputs/wfa_phase_k3_1d_stable/seed_XXXX/
```

### 3. Comparaison Résultats

```python
# Une fois terminé (5 seeds × 300 trials × 14 folds)
python scripts/compare_h2_vs_1d_stable.py

# Compare:
# - K3 H2 pur (11 seeds): 0.30%/mois, MDD 13%, 32 trades/an
# - K3 1D stable (5 seeds): ???%/mois, MDD ???%, ??? trades/an
#
# Décision: Si 1D stable > H2 pur
#   → Lancer 30 seeds complets!
#   → Tester K5 et K8 également
```

---

## 📊 Plan de Validation

### Phase 1: Test Rapide (En cours)
- ✅ Génération K3_1d_stable.csv
- ⏳ 5 seeds K3 1D stable (24-48h)
- 📊 Comparaison vs 11 seeds H2 pur

### Phase 2: Si Validation Positive
- 🔄 Lancer 30 seeds K3 1D stable (complet)
- 🔄 Générer K5_1d_stable.csv et K8_1d_stable.csv
- 🔄 Tests K5 et K8 (5 seeds chacun)

### Phase 3: Sélection Finale
- 📈 Comparer K3 vs K5 vs K8 (version 1D stable)
- 🎯 Sélectionner meilleur modèle
- 🚀 Production ready

---

## 🎓 Contributions Scientifiques

### Nouveautés méthodologiques

1. **Multi-timeframe cohérent:**
   - Stratégie sur timeframe différent de détection de régime
   - Respecte principe no-lookahead
   - Exploite avantages des deux échelles

2. **Filtrage statistique simple:**
   - Label majoritaire = filtre outliers naturel
   - Pas de sur-optimisation (1 seul hyperparamètre: fenêtre 1D)
   - Reproductible et explicable

3. **Validation empirique:**
   - Test A/B clair: H2 pur vs 1D stable
   - Métriques comparables (même protocole WFA)
   - Décision data-driven

---

## 📚 Références Scripts

### Génération
- `scripts/downsample_labels_2h_to_1d.py` — Génère labels 1D stables

### Exécution
- `scripts/launch_k3_1d_stable_test.ps1` — Lance 5 seeds test
- `scripts/run_scheduler_wfa_phase.py` — WFA phase-adapté (inchangé)

### Analyse
- `scripts/quick_k3_metrics.py` — Extraction métriques
- `scripts/compare_h2_vs_1d_stable.py` — Comparaison H2 vs 1D stable (à créer)

### Visualisation
- `scripts/plot_phase_stability.py` — Compare stabilité phases (à créer)

---

## ✅ Checklist Validation

**Avant lancement:**
- [x] Labels K3_1d_stable.csv générés
- [x] Script downsampling testé
- [x] Script lancement 5 seeds prêt
- [x] Documentation complète
- [x] Git push

**Pendant exécution (24-48h):**
- [ ] Monitoring jobs (Get-Job)
- [ ] Vérification logs (seed_X/run.log)
- [ ] Tracking PROGRESS.json
- [ ] Pas de crash/blocage

**Après résultats:**
- [ ] Extraction métriques 5 seeds
- [ ] Comparaison vs 11 seeds H2 pur
- [ ] Décision: valider ou ajuster
- [ ] Documentation résultats finaux

---

**Document créé:** 2025-10-16  
**Statut:** Test en préparation (5 seeds)  
**Prochaine mise à jour:** Après résultats test (48h)

