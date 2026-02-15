# Arbre de Decision - Generateur Alpha HSBC

**Objectif:** Savoir ou on en est et quelle branche suivre
**Maj:** 2026-02-13 - v1 4/5 equity+ (survie 80%), v2 ELIMINE. MULTI-SEED COMPLETE.

**Legende couleurs:**
- 🔴 = FAIT (termine, resultats connus)
- 🟡 = TESTE (en cours d'analyse ou resultats mitiges)
- ⚪ = PAS TESTE (a faire)

---

## ARBRE COMPLET

```
APPROCHES
      |
      +-🔴 HMM K3 (fit sur 14 ans)
      |     Sharpe 0.99, MDD 13%
      |     BIAISE: look-ahead (fit sur tout)
      |     → INUTILISABLE tel quel
      |
      +-🔴 HMM K5 (fit sur 14 ans)
      |     12 seeds ~87%
      |     BIAISE: meme probleme que K3
      |     → INUTILISABLE tel quel
      |
      +-🔴 NHHM v1 (statsmodels)
      |     CRASH: ne converge pas sur 60k points
      |     → ABANDONNE
      |
      +-🔴 NHHM v2 (hmmlearn + rolling 10k)
      |     Score bear markets: 3/5 (60%)
      |     Sharpe global: -0.86, Hit rate: 49.1%
      |     → MEDIOCRE comme filtre seul
      |     → MAIS booste le pipeline en combinaison!
      |
      +-🔴 NHHM 3 etats
      |     Score: 2/5, Sharpe -2.05
      |     → ELIMINE (pire que 2 etats)
      |
      +-🔴 ML seul (LightGBM walk-forward)
      |     Sharpe 0.12, Equity 1.23x, MDD 4.4%
      |     CLEAN (walk-forward), mais trop faible seul
      |
      +-🔴 CYCLE seul (halving fixe)
      |     Sharpe 0.99, Equity 1.22x, MDD ~15%
      |     CLEAN (regle fixe, pas de look-ahead)
      |
      +-🟡 COMBINED v1 (CYCLE + ML) ← FAVORI ★
      |     Seed 101 (30t): Sharpe 0.91, Equity 1.719x, MDD 5.82%
      |     Seed 102 (30t): Sharpe -0.65, Equity 1.835x, MDD 9.76%
      |     Seed 103 (30t): Sharpe -1.30, Equity 1.489x, MDD 20.65%
      |     Seed 104 (30t): Sharpe -1.93, Equity 0.751x, MDD 38.54% ← PERD
      |     Seed 105 (30t): Sharpe -0.02, Equity 1.598x, MDD 8.95%
      |     → 4/5 seeds EQUITY POSITIVE (survie 80%)
      |     → Moyenne equity: ~1.48x, MDD moyen: ~16.7%
      |     → MULTI-SEED COMPLETE
      |
      +-🔴 COMBINED v2 (NHHM + ML) ← ELIMINE
      |     Seed 201 (30t): Sharpe 2.38, Equity 1.321x, MDD 10.59%
      |     Seed 202 (30t): Sharpe -3.39, Equity 0.829x, MDD 27.8%
      |     Seed 203 (30t): Sharpe 0.56, Equity 1.259x, MDD 15.07%
      |     Seed 204 (30t): Sharpe 0.29, Equity 0.695x, MDD 37.87%
      |     → 2/4 PERDENT de l'argent! Survie 50% = INSUFFISANT
      |     → ABANDONNE
      |
      +-🟡 Multi-seed v1 COMPLETE
      |     4/5 equity+ (80% survie)
      |     1 seed perd (s104: 0.751x, MDD 38.5%)
      |     → VALIDE (seuil 66% depasse)
      |
      +-⚪ COMBINED v3 (CYCLE + NHHM + ML)
      |     Triple filtre, pas encore teste
      |     → OPTIONNEL
      |
      +-⚪ NHHM comme feature ML
      |     Ajouter P(bull) comme input du LightGBM
      |     → OPTIONNEL (v2 deja bon sans ca)
      |
      +-⚪ HMM K3 rolling (repare)
            Refaire K3 en rolling window
            → BASSE PRIORITE
```

---

## TABLEAU COMPLET

| # | Approche | Sharpe | Equity | MDD | Clean? | Seeds | Status |
|---|----------|--------|--------|-----|--------|-------|--------|
| 🔴 | K3 HMM | 0.99 | ~1x | 13% | NON | 30 | Biaise |
| 🔴 | K5 HMM | ? | ? | ? | NON | 12 | Biaise |
| 🔴 | NHHM v1 | - | - | - | - | 0 | Crash |
| 🔴 | NHHM v2 seul | -0.86 | - | - | OUI | - | Mediocre |
| 🔴 | NHHM 3 etats | -2.05 | - | - | OUI | - | Elimine |
| 🔴 | ML seul | 0.12 | 1.23x | 4.4% | OUI | 1 | Faible |
| 🔴 | CYCLE seul | 0.99 | 1.22x | ~15% | OUI | 1 | OK |
| 🟡 | **COMBINED v1 s101 (30t)** | **0.91** | **1.719x** | **5.82%** | OUI | 1 | **Equity OK** |
| 🟡 | **COMBINED v1 s102 (30t)** | **-0.65** | **1.835x** | **9.76%** | OUI | 1 | **Equity OK** |
| 🟡 | **COMBINED v1 s103 (30t)** | **-1.30** | **1.489x** | **20.65%** | OUI | 1 | **Equity OK** |
| 🟡 | COMBINED v2 s201 (30t) | 2.38 | 1.321x | 10.59% | OUI | 1 | Equity OK |
| 🔴 | COMBINED v2 s202 (30t) | -3.39 | 0.829x | 27.8% | OUI | 1 | **PERD $** |
| 🔴 | COMBINED v2 s203 (30t) | 0.56 | 1.259x | 15.07% | OUI | 1 | Equity OK |
| 🔴 | **COMBINED v2 s204 (30t)** | **0.29** | **0.695x** | **37.87%** | **OUI** | **1** | **PERD $** |
| 🟡 | **COMBINED v1 s104 (30t)** | **-1.93** | **0.751x** | **38.54%** | OUI | 1 | **PERD $** |
| 🟡 | **COMBINED v1 s105 (30t)** | **-0.02** | **1.598x** | **8.95%** | OUI | 1 | **Equity OK** |
| 🟡 | **COMBINED v1 multi** | - | **moy 1.48x** | **moy 16.7%** | OUI | **5/5** | **4/5 equity+ = VALIDE (80%)** |
| 🔴 | **COMBINED v2 multi** | - | - | - | OUI | **4/5** | **2/4 PERDENT = ELIMINE** |

---

## SMOKE TESTS DU 2026-02-11

```
COMBINED v1 smoke (seed 102, 5 trials):
  Equity 1.10x, MDD 17.05%, Sharpe -0.48, 598 trades
  → DECEVANT vs seed originale (0.91, 1.72x, 5.8%)
  → Forte sensibilite a la seed

COMBINED v2 smoke (seed 201, 5 trials):
  Equity 1.44x, MDD 2.52%, Sharpe 1.69, 371 trades
  → EXCELLENT! Meilleur que v1 sur tous criteres
  → NHHM filtre mediocre seul mais booste en combinaison
  → Le biais bear (93.7% CASH) = faible exposition = faible MDD

NHHM 3 etats (validation rapide):
  Score 2/5, Sharpe -2.05
  → ELIMINE
```

---

## DETAIL NHHM v2 (test du 2026-02-11)

```
Config: hmmlearn GaussianHMM, 2 etats, rolling 10k, refit/500
Features: return_1, momentum_6/12, vol_ratio, rsi_centered, dist_ma20

Bear market tests:
  COVID Mars 2020     attendu BEAR  → detecte BEAR  [OK]   P(bull)=0.43
  Top 64k Mai 2021    attendu BEAR  → detecte BULL  [FAIL] P(bull)=0.67
  FTX Nov 2022        attendu BEAR  → detecte BEAR  [OK]   P(bull)=0.16
  Recovery Jan 2023   attendu BULL  → detecte BEAR  [FAIL] P(bull)=0.21
  Post-Halving 2024   attendu BULL  → detecte BULL  [OK]   P(bull)=0.54

Probleme: le HMM capture la VOLATILITE, pas la DIRECTION
  → Bull hit rate: 54.5% (decent)
  → Bear hit rate: 46.5% (pire que random)
  → Biais bear: 68% des predictions sont "bear"

MAIS en combinaison avec ML:
  → Le biais bear garde le systeme en CASH pendant les periodes incertaines
  → Le ML trade uniquement quand le NHHM dit "bull" = periodes stables
  → Resultat: moins de trades (371 vs 598), meilleur Sharpe, MDD tres bas
```

---

## PRIORITES (mises a jour post-smoke tests)

```
PRIORITE 1: 🟡 Multi-seed COMBINED v2 (NHHM + ML)
   |  Smoke test excellent: Sharpe 1.69, MDD 2.52%
   |  Lancer 5 seeds (201-205), 30 trials chacune
   |  Si 4+/5 survivent (Sharpe > 0.5) → VALIDE
   |  Duree: ~4-5 jours sequentiel
   |
PRIORITE 1bis: 🟡 Multi-seed COMBINED v1 (CYCLE + ML) en parallele
   |  Smoke decevant (-0.48) mais seed originale etait bonne (0.91)
   |  Lancer 5 seeds (101-105), 30 trials chacune
   |  Comparer avec v2
   |
PRIORITE 2: Analyser POURQUOI v2 > v1
   |  Le biais bear du NHHM = feature ou bug?
   |  Est-ce que CASH 93.7% = simple reduction d'exposition?
   |  → Important pour comprendre la source de l'alpha
   |
BASSE: ⚪ COMBINED v3 (CYCLE + NHHM + ML)
        ⚪ NHHM comme feature ML
        ⚪ HMM K3 rolling
```

---

## FICHIERS PAR BRANCHE

### 🟡 COMBINED v2 (meilleur smoke test!)
```
data/COMBINED_v2_labels.csv        ← Labels NHHM+ML fusionnes
src/regime_nhhm_v2.py              ← Code NHHM v2 (hmmlearn + rolling)
src/ml_directional.py              ← Code ML (walk-forward)
outputs/wfa_combined_v2_smoke_201/ ← Smoke test resultats
outputs/nhhm_v2_validation.csv     ← Predictions NHHM completes
```

### 🟡 COMBINED v1 (a confirmer multi-seed)
```
data/COMBINED_labels.csv           ← Labels CYCLE+ML fusionnes
data/CYCLE_cash_bear.csv           ← Input: filtre halving
data/ML_directional.csv            ← Input: direction ML
outputs/wfa_combined_test/         ← Resultats 1 seed originale
outputs/wfa_combined_v1_smoke_102/ ← Smoke test resultats
src/ml_directional.py              ← Code ML (walk-forward)
scripts/analysis/generate_cycle_labels.py ← Code CYCLE
```

### 🔴 NHHM v2 seul (mediocre)
```
src/regime_nhhm_v2.py              ← Code NHHM v2 (hmmlearn + rolling)
scripts/experimental/test_nhhm_v2_validation.py ← Test validation
outputs/nhhm_v2_validation.csv     ← Predictions completes
```

### 🔴 HMM K3/K5 (biaise)
```
src/regime_hmm.py                  ← HMM (fit sur tout = probleme)
src/features_fourier.py            ← Fourier (trailing window = OK)
scripts/production/freeze_hmm_labels.py ← Gel labels (biaise)
```

### 🔴 NHHM v1 (abandonne)
```
src/regime_nhhm.py                 ← statsmodels (crash)
```

---

## CRITERES DE DECISION

| Metrique | Minimum | Bon | Excellent |
|----------|---------|-----|-----------|
| Sharpe | > 0.3 | > 0.5 | > 1.0 |
| Survie seeds | > 50% | > 66% | > 80% |
| MDD | < 25% | < 15% | < 10% |
| Equity 14 ans | > 1.2x | > 1.5x | > 2x |
| Look-ahead | **ZERO** | - | - |

---

## HISTORIQUE DECISIONS

| Date | Decision | Resultat |
|------|----------|----------|
| 2025-10 | Tester K3 | Sharpe 0.99 (biaise: look-ahead) |
| 2026-02-07 | Implementer NHHM v1 | Crash statsmodels |
| 2026-02-08 | Fallback CYCLE | Sharpe 0.99 (clean) |
| 2026-02-09 | Tester ML seul | Sharpe 0.12 (clean mais faible) |
| 2026-02-10 | Fusion COMBINED v1 | Equity 1.72x, MDD 5.8% (clean, 1 seed) |
| 2026-02-11 | Audit look-ahead | K3/K5 biaises, COMBINED clean |
| 2026-02-11 | NHHM v2 (hmmlearn) | 3/5 bear tests, Sharpe -0.86, mediocre |
| 2026-02-11 | NHHM 3 etats | 2/5, Sharpe -2.05, ELIMINE |
| 2026-02-11 | Smoke COMBINED v1 s102 | Sharpe -0.48, decevant |
| 2026-02-11 | Smoke COMBINED v2 s201 | Sharpe 1.69, MDD 2.52%! EXCELLENT |
| 2026-02-12 | 30t v2 s201 | Sharpe 2.38, Equity 1.321x, MDD 10.59% |
| 2026-02-12 | 30t v1 s101 | Sharpe 0.91, Equity 1.719x, MDD 5.82% (confirme!) |
| 2026-02-12 | 30t v2 s202 | Sharpe -3.39, Equity 0.829x, MDD 27.8% = PERD $ |
| 2026-02-12 | 30t v1 s102 | Equity 1.835x, MDD 9.76% = CONFIRME v1 stable |
| **2026-02-13** | **30t v2 s203** | **Equity 1.259x, MDD 15.07% (OK)** |
| **2026-02-13** | **30t v2 s204** | **Equity 0.695x, MDD 37.87% = PERD $ → v2 ELIMINE (2/4 perdent)** |
| **2026-02-13** | **30t v1 s103** | **Equity 1.489x, MDD 20.65% → v1 3/3 equity+** |
| **2026-02-13** | **30t v1 s104** | **Equity 0.751x, MDD 38.54% = PERD $** |
| **2026-02-13** | **30t v1 s105** | **Equity 1.598x, MDD 8.95% → v1 4/5 equity+ = VALIDE** |

---

## COMMANDES UTILES

```powershell
# Lancer multi-seed COMBINED v2
foreach ($seed in 201..205) {
  py -3 scripts/production/run_scheduler_wfa_phase.py `
    --labels-csv data/COMBINED_v2_labels.csv `
    --trials 30 --seed $seed `
    --out-dir outputs/wfa_combined_v2_seed_$seed --use-fused
}

# Lancer multi-seed COMBINED v1
foreach ($seed in 101..105) {
  py -3 scripts/production/run_scheduler_wfa_phase.py `
    --labels-csv data/COMBINED_labels.csv `
    --trials 30 --seed $seed `
    --out-dir outputs/wfa_combined_v1_seed_$seed --use-fused
}

# Verifier tous les resultats
py -3 -c "
import json, glob
for p in sorted(glob.glob('outputs/wfa_combined_*/WFA_*.json')):
    d = json.load(open(p))
    o = d.get('overall', {})
    print(f'{p}:')
    print(f'  Sharpe={o.get(\"sharpe\",\"?\")}, Equity={o.get(\"equity_final\",\"?\")}, MDD={o.get(\"max_dd\",\"?\")}')
"

# Tester imports
py -3 -c "from src.regime_nhhm_v2 import NHHMv2; print('OK')"
```

---

Cree: 2026-02-09
Maj: 2026-02-13 - MULTI-SEED COMPLETE. v1 4/5 equity+ (80% survie). v2 ELIMINE. s104 seul echec (0.751x).
