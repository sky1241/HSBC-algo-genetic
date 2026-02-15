# POST-MORTEM: Echec Implementation NHHM

**Date:** 2026-02-08
**Projet:** HSBC-algo-genetic
**Auteur:** Claude (Session 3-4)

---

## RESUME EXECUTIF

On a **abandonne trop vite** le NHHM (Non-Homogeneous HMM) qui etait cense etre le **moteur d'alpha** du systeme. A la place, on a implemente un simple filtre temporel (CYCLE halving) qui **ne genere pas d'alpha**.

**Resultat:** Sharpe OOS de 0.99 (survie) au lieu de l'alpha espere (Sharpe > 2.0).

---

## CE QU'ON VOULAIT FAIRE

### Architecture Prevue
```
NHHM (prediction direction) --> P(bull), P(bear)
         |
         v
   Labels dynamiques (1=LONG, -1=SHORT, 0=CASH)
         |
         v
   Ichimoku+ATR (signaux entree/sortie)
         |
         v
   WFA Optuna (optimisation parametres)
         |
         v
   ALPHA $$$
```

### Le Role du NHHM
Le NHHM devait:
1. Prendre des features (momentum, vol, halving, funding, Fourier)
2. Modeliser les transitions entre regimes bull/bear
3. **PREDIRE** P(bull demain) et P(bear demain)
4. Generer des signaux directionnels AVANT que le mouvement arrive

---

## CE QUI A MERDE

### Erreur #1: Mauvais choix de librairie
```python
# regime_nhhm.py utilise:
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# PROBLEME: Cette librairie ne converge PAS bien avec:
# - Beaucoup de donnees (>10k points)
# - Plusieurs covariables TVTP
# - Donnees financieres bruitees
```

### Erreur #2: Fallback silencieux
```python
# Lignes 295-327 de regime_nhhm.py:
except Exception as e1:
    # ... au lieu de CORRIGER, on tombe en mode degrade:
    self._fallback_mode = True  # <-- MODE BIDON ACTIVE
```

On aurait du:
- Logger l'erreur clairement
- Lever une exception pour forcer la correction
- PAS avoir de fallback silencieux

### Erreur #3: Abandon premature
Au lieu de chercher une solution (autre librairie, autre approche), on a:
1. Abandonne le NHHM
2. Implemente le CYCLE halving (simple filtre temporel)
3. Cru que ca "marchait" car Sharpe ~1.0

### Erreur #4: Confusion filtre vs predicteur
```
CYCLE = Filtre temporel passif
        "Ne trade pas pendant les bear markets"
        = Evite les pertes mais ne genere pas d'alpha

NHHM = Predicteur actif
       "Demain sera bullish avec P=0.73"
       = Genere de l'alpha si la prediction est bonne
```

On a confondu les deux et cru que le CYCLE etait suffisant.

---

## RESULTATS OBTENUS (vs ATTENDUS)

| Metrique | Attendu (NHHM) | Obtenu (CYCLE) | Ecart |
|----------|----------------|----------------|-------|
| Sharpe OOS | > 2.0 | 0.99 | -50% |
| Equity 14 ans | > x5 | x1.217 | -76% |
| Alpha | OUI | NON | ECHEC |
| Survie | OUI | OUI | OK |

Le systeme CYCLE **survit** mais **ne performe pas**.

---

## SOLUTIONS POSSIBLES

### Solution A: Reparer NHHM avec hmmlearn
```python
from hmmlearn import GaussianHMM

# Plus robuste que statsmodels
# Converge bien sur donnees financieres
# Supporte les features externes via preprocessing
```

### Solution B: Remplacer par ML supervisee (RECOMMANDE)
```python
import lightgbm as lgb

# Predire directement: y = 1 si return_forward > 0, else 0
# Features: momentum, vol, halving, funding, Fourier
# Avantages: simple, interpretable, robuste
```

### Solution C: Ensemble NHHM + ML
Combiner les deux approches pour robustesse.

---

## LECONS APPRISES

1. **Pas de fallback silencieux** - Si ca crash, ca doit CRIER
2. **Tester le coeur du systeme d'abord** - Le NHHM etait le coeur, on aurait du le valider avant tout
3. **Filtre != Predicteur** - Un filtre temporel n'est pas un generateur d'alpha
4. **Ne pas abandonner trop vite** - Chercher des alternatives avant de pivoter
5. **Documenter les echecs** - Ce post-mortem aurait du etre fait plus tot

---

## FICHIERS CONCERNES

| Fichier | Probleme | Action |
|---------|----------|--------|
| `src/regime_nhhm.py` | Fallback silencieux, mauvaise librairie | A REFAIRE |
| `scripts/generate_nhhm_labels.py` | Utilise NHHM casse | A REFAIRE |
| `data/CYCLE_*.csv` | Filtre passif, pas d'alpha | GARDER comme backup |

---

## PROCHAINES ETAPES

1. [ ] Implementer predicteur ML (LightGBM) pour direction
2. [ ] Tester sur donnees historiques (hit rate, Sharpe backtest)
3. [ ] Generer labels `ML_directional.csv`
4. [ ] Lancer WFA avec nouveaux labels
5. [ ] Comparer Sharpe OOS vs CYCLE (0.99)

---

## METRIQUES DE SUCCES

Le nouveau systeme sera considere comme REUSSI si:
- [ ] Hit rate prediction > 53%
- [ ] Sharpe OOS > 1.5
- [ ] Equity 14 ans > x2.0
- [ ] Max Drawdown < 25%

---

*Document genere le 2026-02-08 pour passation de contexte.*
