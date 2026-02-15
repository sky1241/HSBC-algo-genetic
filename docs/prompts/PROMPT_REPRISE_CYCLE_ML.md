# PROMPT DE REPRISE - Fusion CYCLE + ML

**Date:** 2026-02-09
**Objectif:** Combiner CYCLE (filtre) + ML (direction) pour meilleur Sharpe

---

## CONTEXTE RAPIDE

On a teste 2 approches separees:
- **CYCLE seul:** Sharpe 0.99, Equity 1.22x, MDD ~15%
- **ML seul:** Sharpe 0.12, Equity 1.23x, MDD 4.4%

ML a un excellent MDD mais mauvais Sharpe. CYCLE a bon Sharpe mais MDD eleve.
L'idee: combiner les deux.

---

## PROMPT A COPIER-COLLER

```
Je reprends le projet HSBC-algo-genetic.

## CHECKS DE CONFORMITE (FAIRE EN PREMIER)

Avant de coder, verifier que tout est en place:

1. Fichiers existent:
   - data/ML_directional.csv
   - data/CYCLE_cash_bear.csv
   - src/ml_directional.py

2. Tester import ML:
   py -3 -c "from src.ml_directional import MLDirectional; print('OK')"

3. Verifier format labels:
   py -3 -c "import pandas as pd; print(pd.read_csv('data/ML_directional.csv').head())"

## RESULTATS PRECEDENTS

| Approche | Sharpe | Equity | MDD |
|----------|--------|--------|-----|
| CYCLE | 0.99 | 1.22x | ~15% |
| ML | 0.12 | 1.23x | 4.4% |

## CE QUE JE VEUX

Creer un script qui fusionne CYCLE + ML:
- Si CYCLE = 0 (bear) → CASH force
- Si CYCLE = 1 (bull) → utiliser ML

Output: data/COMBINED_labels.csv

## CONTRAINTES IMPORTANTES

1. Tests RAPIDES: --trials 30 (pas 50!)
2. Un seul seed pour valider l'hypothese
3. Si ca marche, on lance plus de seeds apres

## FICHIERS A LIRE

1. data/CYCLE_cash_bear.csv - Format labels CYCLE
2. data/ML_directional.csv - Format labels ML
3. src/ml_directional.py - Voir comment ML genere les labels

## COMMANDE WFA TEST

py -3 scripts/run_scheduler_wfa_phase.py \
  --labels-csv data/COMBINED_labels.csv \
  --trials 30 --seed 101 \
  --out-dir outputs/wfa_combined_test \
  --use-fused

## CRITERES DE SUCCES

- Sharpe OOS > 1.0 (mieux que ML seul)
- MDD < 10% (garder avantage ML)
- Test termine en < 30 min
```

---

## LOGIQUE DE FUSION

```python
import pandas as pd

# Charger les deux
cycle = pd.read_csv('data/CYCLE_cash_bear.csv', parse_dates=['timestamp'])
ml = pd.read_csv('data/ML_directional.csv', parse_dates=['timestamp'])

# Merger sur timestamp
combined = cycle.merge(ml, on='timestamp', suffixes=('_cycle', '_ml'))

# Logique: CYCLE filtre, ML decide
combined['label'] = combined.apply(
    lambda r: 0 if r['label_cycle'] == 0 else r['label_ml'],
    axis=1
)

# Sauver
combined[['timestamp', 'label']].to_csv('data/COMBINED_labels.csv', index=False)
```

---

## RESULTATS ATTENDUS

| Metrique | Objectif | Raison |
|----------|----------|--------|
| Sharpe | > 1.0 | CYCLE filtre les mauvais trades |
| MDD | < 10% | ML garde son avantage |
| Equity | > 1.3x | Synergie des deux |
