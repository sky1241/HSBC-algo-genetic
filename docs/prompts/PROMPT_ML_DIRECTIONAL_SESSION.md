# SESSION: Implementation ML Directional (LightGBM)

**Date debut:** 2026-02-08
**Objectif:** Remplacer NHHM casse par LightGBM pour generer de l'alpha

---

## ETAT ACTUEL

| Etape | Status | Resultat |
|-------|--------|----------|
| 1. Creer src/ml_directional.py | OK | Module cree |
| 2. Test unitaire features | OK | 11 features, 61210 bars valides |
| 3. Walk-forward training | OK | 12 annees (2014-2025) |
| 4. Generer ML_directional.csv | OK | 61465 bars, LONG=10757, CASH=50169, SHORT=539 |
| 5. Valider hit rate > 53% | PROCHE | **52.69%** (objectif 53%) |
| 6. Lancer WFA test | EN COURS | outputs/wfa_ml_v1 (50 trials) |
| 7. Comparer Sharpe vs CYCLE | PENDING | baseline: 0.99 |

## RESULTATS PAR ANNEE (Hit Rate)

| Annee | Hit Rate | Notes |
|-------|----------|-------|
| 2014 | 45.8% | Bear market post-ATH |
| 2015 | 50.6% | Accumulation |
| 2016 | 57.0% | Pre-halving |
| 2017 | 57.7% | Bull run |
| 2018 | 51.1% | Bear market |
| 2019 | 52.3% | Recovery |
| 2020 | 58.8% | Halving + bull |
| 2021 | 52.3% | Double top |
| 2022 | 48.3% | Bear market |
| 2023 | 52.7% | Recovery |
| 2024 | 54.0% | ETF + halving |
| 2025 | 51.2% | Partiel |

## FEATURE IMPORTANCE (Top 5)

| Feature | Importance |
|---------|------------|
| LFP_ratio | 21.8% |
| vol_ratio | 12.7% |
| dist_ma50 | 12.3% |
| halving_phase | 11.4% |
| halving_direction | 10.1% |

---

## CHECKPOINTS AUTO-CONTROLE

### Checkpoint 1: Features OK
- [ ] `build_nhhm_features()` retourne DataFrame sans NaN excessifs
- [ ] Toutes les features disponibles depuis 2011

### Checkpoint 2: Model OK
- [ ] LightGBM s'entraine sans erreur
- [ ] Pas de lookahead bias (train < test temporellement)

### Checkpoint 3: Predictions OK
- [ ] Hit rate calcule sur OOS > 50%
- [ ] Labels generes au bon format (timestamp, label)

### Checkpoint 4: WFA OK
- [ ] WFA tourne sans crash
- [ ] Sharpe OOS calcule

### Checkpoint 5: SUCCES
- [ ] Hit rate > 53%
- [ ] Sharpe OOS > 1.5
- [ ] Equity > x2.0

---

## METRIQUES BASELINE (CYCLE)

| Metrique | Valeur CYCLE |
|----------|--------------|
| Sharpe OOS | 0.99 |
| Equity 14 ans | x1.217 |
| Alpha | NON |

---

## FICHIERS CREES

| Fichier | Description |
|---------|-------------|
| src/ml_directional.py | Module LightGBM principal |
| scripts/generate_ml_labels.py | Generateur de labels |
| data/ML_directional.csv | Labels predits |

---

## COMMANDES DE TEST

```powershell
# Test module ML
py -3 -c "from src.ml_directional import MLDirectional; print('OK')"

# Generer labels
py -3 scripts/generate_ml_labels.py

# WFA test rapide
py -3 scripts/run_scheduler_wfa_phase.py --labels-csv data/ML_directional.csv --trials 50 --seed 101 --out-dir outputs/wfa_ml_test
```

---

## NOTES DE SESSION

### Session 2026-02-08

**21:30** - Debut implementation ML Directional
- Cree src/ml_directional.py avec LightGBM
- Walk-forward training: 2014-2025
- Hit rate global: 52.69% (proche de 53%)
- Labels generes: data/ML_directional.csv

**21:45** - WFA lance en background
- Commande: `py -3 scripts/run_scheduler_wfa_phase.py --labels-csv data/ML_directional.csv --trials 50 --seed 101 --out-dir outputs/wfa_ml_v1 --use-fused`
- Progress: 16.57% (2/14 folds)
- Estimation: ~30-60 min pour completion

### Observations
1. Les meilleurs hit rates sont en annees post-halving (2016, 2017, 2020)
2. Les pires sont en bear markets (2014, 2022)
3. LFP_ratio (Fourier) est la feature la plus importante (21.8%)
4. Le modele genere beaucoup de CASH (82%) vs LONG (17%) vs SHORT (0.9%)

### Pistes d'amelioration
- Baisser long_threshold de 0.55 a 0.52 pour plus de LONG
- Ajouter funding_rate pour 2019+ (dispo dans les donnees)
- Essayer horizon plus long (24 bars = 48h au lieu de 12)

