# PROMPT NOUVELLE SESSION - COPIE TOUT ÇA

---

## MISSION

Implémenter les améliorations identifiées pour le système de trading algo BTC, puis lancer les 30 seeds K5 (2-3 semaines de calcul).

---

## DIAGNOSTIC (ChatGPT Deep Research)

**Problème** : Système robuste (100% survie, 12% MDD) mais rendement faible (0.30%/mois vs 5% objectif).

**Causes identifiées** :
1. On reparamétrise du trend-following → pas d'alpha dans les phases non-trend
2. Fourier/PSD est DESCRIPTIF, pas PRÉDICTIF
3. HMM Gaussien inadapté aux heavy tails crypto
4. 300 trials Optuna = possible overfitting

**Analyse complète** : `docs/CHATGPT_DEEP_RESEARCH_ANALYSIS.md`

---

## À CODER (dans l'ordre)

### 1. CHECKPOINT SYSTEM ⭐⭐⭐ (OBLIGATOIRE)
**Fichiers** : `src/checkpoint_manager.py` → intégrer dans `scripts/run_scheduler_wfa_phase.py`

**Ce que ça fait** :
- Sauvegarde progression toutes les 10 min
- Reprend après crash au bon fold/phase/trial
- Détecte seeds déjà terminés

**Code à intégrer** :
```python
from src.checkpoint_manager import RobustRunner

runner = RobustRunner("outputs/wfa_phase_k5", checkpoint_interval_minutes=10)

for seed in runner.get_seeds_to_run(all_seeds):
    resume_state = runner.get_resume_state()
    if resume_state:
        start_fold = resume_state["fold"]
        start_phase = resume_state["phase"]

    for fold in folds:
        for phase in phases:
            # Optimisation Optuna...
            runner.save_progress(seed, fold, phase, trial, total_trials, best_params, best_score, completed_phases)

    runner.complete_seed(seed, final_results)

runner.shutdown()
```

### 2. VOLATILITY TARGETING ⭐⭐⭐ (HIGH IMPACT)
**But** : Scaler l'exposition selon la volatilité réalisée

**Formule** :
```python
leverage = min(L_max, sigma_target / sigma_realized)
# + drawdown throttle : réduire si drawdown > seuil
```

**Où l'implémenter** : Dans le backtest (`ichimoku_pipeline_web_v4_8_fixed.py`) ou nouveau module `src/volatility_targeting.py`

### 3. MIXTURE-OF-EXPERTS ⭐⭐ (MEDIUM-HIGH IMPACT)
**But** : Stratégie différente par phase au lieu de juste reparamétrer Ichimoku

| Phase HMM | Stratégie |
|-----------|-----------|
| Trend (LFP élevé) | Ichimoku (actuel) |
| Chop/Range | Mean-reversion (RSI, Bollinger) |
| Breakout | Donchian, range expansion |
| Carry | Funding/basis arbitrage |

**Soft-switching** : Utiliser probabilités HMM, pas hard switch

### 4. HMM HEAVY-TAIL ⭐ (STABILITÉ)
**But** : Remplacer Gaussian → Student-t emissions

**Fichier** : `src/regime_hmm.py` ou nouveau `src/hmm_student_t.py`

---

## WORKFLOW

### Étape 1 : Vérifier que tout marche (2 min)
```powershell
py scripts/test_quick_validation.py
```

### Étape 2 : Coder le checkpoint
- Modifier `scripts/run_scheduler_wfa_phase.py`
- Tester : `py scripts/test_quick_validation.py`

### Étape 3 : Coder le volatility targeting
- Nouveau module ou intégrer dans pipeline
- Tester

### Étape 4 : Test intégration complet (1h)
```powershell
py scripts/test_integration_1h.py
```

### Étape 5 : Si tout OK → Lancer 30 seeds
```powershell
.\scripts\launch_30_seeds_k5.ps1
# Tourne 2-3 semaines jour/nuit
# Checkpoint = peut reprendre après crash
```

---

## FICHIERS CLÉS

| Fichier | Rôle |
|---------|------|
| `docs/CHATGPT_DEEP_RESEARCH_ANALYSIS.md` | Diagnostic + solutions détaillées |
| `src/checkpoint_manager.py` | Code checkpoint PRÊT |
| `src/continuous_learning.py` | Auto-learning PRÊT |
| `src/spectral/` | Module Fourier P2+P3 PRÊT |
| `scripts/run_scheduler_wfa_phase.py` | **À MODIFIER** (ajouter checkpoint) |
| `scripts/test_quick_validation.py` | Test rapide 2 min |
| `scripts/test_integration_1h.py` | Test complet 1h |
| `scripts/launch_30_seeds_k5.ps1` | Lancement 30 seeds parallèles |

---

## CONTRAINTES

- **Calculs = 2-3 semaines** pour 30 seeds → checkpoint OBLIGATOIRE
- **Test 1h avant de lancer** → valider que tout fonctionne
- **PC tourne jour/nuit** → checkpoint permet de reprendre

---

## POUR DÉMARRER

Dis :

```
Lis ces fichiers dans l'ordre :
1. docs/CHATGPT_DEEP_RESEARCH_ANALYSIS.md (diagnostic)
2. src/checkpoint_manager.py (code à intégrer)
3. scripts/run_scheduler_wfa_phase.py (script à modifier)

Ensuite :
1. Intègre le checkpoint dans run_scheduler_wfa_phase.py
2. Crée un module volatility_targeting.py
3. Lance py scripts/test_quick_validation.py après chaque modif
4. Quand tout est OK, lance py scripts/test_integration_1h.py
5. Si le test 1h passe, on lance les 30 seeds
```

---

## RÉSUMÉ EN UNE PHRASE

"Code le checkpoint + volatility targeting, teste en 1h, puis lance 30 seeds K5 qui tourneront 2-3 semaines."

---
