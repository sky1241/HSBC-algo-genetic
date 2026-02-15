# PROMPT REPRISE - HSBC-algo-genetic

**Date:** 2026-02-10
**Copie-colle ce prompt pour reprendre le projet**

---

## PROMPT

```
Je reprends le projet HSBC-algo-genetic (trading algo BTC).

## FICHIERS A LIRE EN PREMIER
1. INDEX.md - carte complete du projet
2. ARBRE_PROJET.md - structure visuelle
3. docs/README_ETAT.md - etat actuel

## ETAT ACTUEL (2026-02-10)

### Resultats valides
| Approche | Sharpe | Equity | MDD | Status |
|----------|--------|--------|-----|--------|
| K3 | 0.99 | - | 13% | VALIDE |
| CYCLE | 0.99 | 1.22x | ~15% | VALIDE |
| ML seul | 0.12 | 1.23x | 4.4% | Teste |
| COMBINED | 0.91 | 1.72x | 5.8% | PROMETTEUR |
| K5 | ? | ? | ? | En cours (~87%) |

### COMBINED = meilleur resultat
- CYCLE protege en bear (MDD faible)
- ML filtre en bull (plus de trades)
- Equity 1.72x vs 1.22x CYCLE seul

### Structure projet (arbre simplifie)
```
CIME:     [C1 Quality] [C2 Docs] [C3 Agreg]
BRANCHES: [B1 Labels] [B2 HMM] [B3 ML] [B4 Analysis] [B5 Live]
TRONC:    T5 Metrics → T4 Scheduler → T3 Labels → T2 Pipeline → T1 Data
RACINES:  [R1 BTC] [R2 Fourier] [R3 HMM] [R4 Seeds] [R5 WFA]
```

### Fichier COEUR
ichimoku_pipeline_web_v4_8_fixed.py (3212 lignes)
- Backtest Ichimoku + ATR
- Optimisation Optuna
- Calcul metriques

### Scripts production
- scripts/production/run_scheduler_wfa_phase.py → Lance WFA
- scripts/production/freeze_hmm_labels.py → Gele labels
- scripts/production/aggregate_k5_results.py → Agregge resultats

### Labels disponibles
- data/CYCLE_cash_bear.csv → CYCLE (VALIDE)
- data/ML_directional.csv → ML
- data/COMBINED_labels.csv → CYCLE+ML (PROMETTEUR)
- outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv → K3 (VALIDE)
- outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv → K5 (EN TEST)

## CE QUE JE VEUX FAIRE
[COMPLETE ICI]

## REGLES
- NE PAS toucher a outputs/wfa_phase_k5/ (K5 en cours)
- NE PAS supprimer de fichiers sans demander
- Consulter INDEX.md avant de chercher du code
```

---

## VARIANTES RAPIDES

### Pour continuer les tests
```
Je reprends HSBC-algo-genetic.
Lire: INDEX.md, docs/README_ETAT.md
Etat: K5 a ~87%, COMBINED = 1.72x equity (meilleur resultat)
Je veux: [lancer test / voir resultats / comparer approches]
```

### Pour comprendre le code
```
Je reprends HSBC-algo-genetic.
Lire: INDEX.md, ARBRE_PROJET.md
Structure: Pipeline (3212 lignes) → Scheduler → Labels → WFA
Je veux comprendre: [backtest / labels / WFA / ML]
```

### Pour ajouter une feature
```
Je reprends HSBC-algo-genetic.
Lire: INDEX.md, ARBRE_PROJET.md section REFERENCES
Coeur: ichimoku_pipeline_web_v4_8_fixed.py
Je veux ajouter: [feature]
Zone concernee: [R/T/B/C + numero]
```

---

## CHECKS RAPIDES

```powershell
# Processus K5 actifs
Get-Process python* | Measure-Object

# Progression K5
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
  $j = Get-Content $_.FullName | ConvertFrom-Json
  "$($_.Directory.Name): $($j.percent)%"
}

# Tester imports
py -3 -c "from src.regime_hmm import fit_regime_model; print('OK')"

# Aide scheduler
py -3 scripts/production/run_scheduler_wfa_phase.py --help
```

---

## ZONES DE L'ARBRE (pour reference)

| Zone | ID | Fichiers cles |
|------|-----|---------------|
| Racine | R1 | data/BTC_FUSED_2h.csv |
| Racine | R2 | src/features_fourier.py |
| Racine | R3 | src/regime_hmm.py |
| Racine | R4 | --seed dans scheduler |
| Racine | R5 | WFA 14 folds annuels |
| Tronc | T1 | data/BTC_FUSED_2h.csv |
| Tronc | T2 | ichimoku_pipeline_web_v4_8_fixed.py |
| Tronc | T3 | data/*.csv, outputs/fourier/labels_frozen/ |
| Tronc | T4 | scripts/production/run_scheduler_wfa_phase.py |
| Tronc | T5 | Sharpe/MDD/CAGR calcules dans T2 |
| Branche | B1 | scripts/production/freeze_hmm_labels.py |
| Branche | B2 | src/regime_hmm.py |
| Branche | B3 | src/ml_directional.py |
| Branche | B4 | scripts/analysis/ (63 scripts) |
| Branche | B5 | src/live_trader_adaptive.py |
| Cime | C1 | src/checkpoint_manager.py |
| Cime | C2 | docs/guides/, INDEX.md |
| Cime | C3 | scripts/production/aggregate_k5_results.py |

---

*Cree le 2026-02-10*
