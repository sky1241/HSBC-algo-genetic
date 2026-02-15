# Guide de Validation Automatique - Generateur Alpha HSBC

**Objectif:** Determiner automatiquement seeds/trials selon ce qu'on implemente

---

## CONTEXTE DU PROJET

Ce projet est un **generateur d'alpha** base sur:
- **Fourier**: Extraction features spectrales (P1_period, LFP_ratio, volatility)
- **HMM/NHHM**: Classification regimes de marche (K phases)
- **Ichimoku + ATR**: Strategie de trading
- **WFA**: Walk-Forward Analysis (validation sans lookahead)

---

## ETAPE 1: Identifier le type de changement

**Question a poser au debut de chaque test:**

| Type | Ce que tu changes | Exemples concrets |
|------|-------------------|-------------------|
| **A** | Parametre Ichimoku/ATR | atr_mult 10→15, tenkan range |
| **B** | Filtre/Condition | CYCLE filter, entry condition |
| **C** | Nombre de phases K | K3→K5, K5→K8 |
| **D** | Modele de regime | HMM→NHHM, ajouter ML |
| **E** | Refonte features | Nouvelles features Fourier, Deep Learning |

---

## ETAPE 2: Parametres par type

### Type A - Tweak Ichimoku/ATR
```
Seeds:       3
Trials:      30
Temps reel:  3-6 heures
```
**Critere OK:** 2/3 seeds survivent, pas de regression

---

### Type B - Nouveau filtre (CYCLE, condition)
```
Seeds:       5
Trials:      50
Temps reel:  8-15 heures
```
**Critere OK:** 4/5 survivent, MDD pas pire que baseline

---

### Type C - Changement K (K3→K5, K5→K8)
```
Seeds:       10
Trials:      100
Temps reel:  15-30 heures
```
**Critere OK:** 8/10 survivent, Sharpe median > 0.5

**C'est le cas K5 actuel!** On aurait du faire 10 seeds × 100 trials.

---

### Type D - Nouveau modele regime (HMM→NHHM, +ML)
```
Seeds:       15
Trials:      150
Temps reel:  30-50 heures
```
**Critere OK:** 12/15 survivent, alpha positif vs baseline

---

### Type E - Refonte majeure (Deep Learning, nouvelles features)
```
Seeds:       20
Trials:      200
Temps reel:  50-80 heures
```
**Critere OK:** 15/20 survivent, amelioration significative

---

### Validation FINALE (avant production)
```
Seeds:       30
Trials:      300
Temps reel:  80-120 heures (3-5 jours)
```
**Critere OK:** 25/30 survivent, medianes stables

---

## ETAPE 3: Prompt automatique

**Copie ce template au debut de chaque session:**

```
Je vais tester: [DESCRIPTION]

## Identification
- Type: [A/B/C/D/E] (voir GUIDE_VALIDATION_AUTO.md)
- Fichier labels: [data/XXX.csv]
- Baseline comparaison: [K3/CYCLE/autre]

## Parametres recommandes (selon guide)
- Seeds: X
- Trials: Y
- Temps estime: Z heures

## Critere de succes
- Survie: X/Y seeds
- Sharpe median: > 0.5
- MDD median: < 20%

## Commande
py -3 scripts/run_scheduler_wfa_phase.py \
  --labels-csv data/[FICHIER].csv \
  --trials [Y] --seed 101 \
  --out-dir outputs/[NOM_TEST] \
  --use-fused

## SI test rapide insuffisant → lancer plus de seeds
```

---

## TABLEAU RECAPITULATIF

| Type | Description | Seeds | Trials | Heures | Survie OK |
|------|-------------|-------|--------|--------|-----------|
| A | Tweak param | 3 | 30 | 3-6h | 2/3 |
| B | Filtre | 5 | 50 | 8-15h | 4/5 |
| C | Changer K | 10 | 100 | 15-30h | 8/10 |
| D | Nouveau modele | 15 | 150 | 30-50h | 12/15 |
| E | Refonte | 20 | 200 | 50-80h | 15/20 |
| PROD | Final | 30 | 300 | 80-120h | 25/30 |

---

## TEMPS REELS (ta machine)

Base sur experience K5:
- 12 seeds a 80% = ~10 heures
- 1 seed complet (300t × 14 folds) = ~5-8 heures
- 6 jobs paralleles ≠ 6× plus rapide (CPU sature ~2-3×)

**Regle: Multiplier estimations par 5-10**

---

## ERREURS A EVITER

| Erreur | Impact |
|--------|--------|
| 30 seeds pour valider K5 | +48h inutiles |
| 300 trials pour test | ×3 le temps |
| Pas de critere defini | On sait pas si OK |
| Comparer au mauvais baseline | Conclusions fausses |

---

## EXEMPLE K5 (ce qu'on aurait du faire)

**Situation:** Tester K5 au lieu de K3

1. **Type = C** (changement nombre de phases)
2. **Parametres:** 10 seeds × 100 trials
3. **Temps:** ~15-30h (pas 80h+)
4. **Critere:** 8/10 survivent, Sharpe > 0.5

**Apres validation rapide SI OK:**
- Lancer 30 seeds × 300 trials pour production

---

## DECISION TREE

```
Tu changes quoi?
│
├─ Parametre numerique (ATR, tenkan...) → Type A (3 seeds)
├─ Condition/Filtre (CYCLE, entry) → Type B (5 seeds)
├─ Nombre de phases (K3→K5) → Type C (10 seeds)
├─ Modele regime (HMM→NHHM, ML) → Type D (15 seeds)
└─ Refonte complete (DL, features) → Type E (20 seeds)

Resultat OK?
├─ Oui → Lancer validation finale (30 seeds × 300 trials)
└─ Non → Abandonner ou modifier approche
```

---

---

## AUTO-DETECTION PAR FICHIER

**Quel fichier tu modifies = quel type de test**

| Fichier modifie | Type | Seeds | Trials | Ce que tu fais |
|-----------------|------|-------|--------|----------------|
| `src/optimizer.py` (params) | A | 3 | 30 | Tweak Ichimoku/ATR |
| `src/risk_sizing.py` | A | 3 | 30 | Tweak sizing |
| `scripts/generate_cycle_labels.py` | B | 5 | 50 | Filtre CYCLE |
| `data/CYCLE_*.csv` | B | 5 | 50 | Nouveau filtre |
| `scripts/freeze_hmm_labels.py` (K) | C | 10 | 100 | Changer K3→K5→K8 |
| `data/K*.csv` labels | C | 10 | 100 | Nouvelles phases |
| **`src/regime_nhhm.py`** | **D** | **15** | **150** | **NHHM (toi maintenant!)** |
| `src/ml_directional.py` | D | 15 | 150 | ML directionnel |
| `src/regime_hmm.py` (modele) | D | 15 | 150 | Nouveau HMM |
| `src/features_fourier.py` | E | 20 | 200 | Nouvelles features |
| `src/continuous_learning.py` | E | 20 | 200 | Online learning |

---

## TON CAS ACTUEL: NHHM

Tu travailles sur `src/regime_nhhm.py` = **Type D**

**Parametres recommandes:**
```
Seeds:       15
Trials:      150
Temps reel:  30-50 heures
```

**Critere de succes:**
- 12/15 seeds survivent
- Alpha positif vs baseline HMM
- Sharpe median > baseline

**PAS 30 seeds × 300 trials pour valider NHHM!**

---

## PROMPT SPECIFIQUE NHHM

```
Je teste le NHHM (regime_nhhm.py)

## Identification
- Type: D (nouveau modele regime)
- Fichier: src/regime_nhhm.py
- Baseline: HMM classique (K3 ou K5)

## Parametres (Type D)
- Seeds: 15
- Trials: 150
- Temps: 30-50h

## Critere de succes
- Survie: 12/15 seeds
- Sharpe > baseline HMM
- P(direction) fonctionne vraiment

## Commande
py -3 scripts/run_scheduler_wfa_phase.py \
  --labels-csv data/NHHM_labels.csv \
  --trials 150 --seed 101 \
  --out-dir outputs/wfa_nhhm_test \
  --use-fused
```

---

Cree: 2026-02-09
Mis a jour: 2026-02-09
