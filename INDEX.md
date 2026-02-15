# INDEX - Carte Complete du Projet

**Derniere maj:** 2026-02-10
**But:** Savoir ou chercher quoi, comprendre l'architecture

---

## ARCHITECTURE EN 1 IMAGE

```
                    DONNEES
                       |
              data/BTC_FUSED_2h.csv
                       |
                       v
    +------------------------------------------+
    |   ichimoku_pipeline_web_v4_8_fixed.py    |  <-- COEUR (3200 lignes)
    |   - Backtest Ichimoku + ATR              |
    |   - Optimisation Optuna                  |
    |   - Calcul metriques (Sharpe, MDD)       |
    +------------------------------------------+
                       |
                       v
    +------------------------------------------+
    |   scripts/production/                     |
    |   run_scheduler_wfa_phase.py             |  <-- ORCHESTRATEUR WFA
    |   - Charge labels (K3, K5, CYCLE)        |
    |   - Lance optimisation par phase         |
    |   - Sauvegarde resultats JSON            |
    +------------------------------------------+
                       |
                       v
              outputs/wfa_phase_*/
              (resultats JSON par seed)
```

---

## FICHIERS CRITIQUES (NE PAS TOUCHER)

### Moteur Principal
| Fichier | Lignes | Role |
|---------|--------|------|
| `ichimoku_pipeline_web_v4_8_fixed.py` | 3212 | Backtest + Optuna + Ichimoku |

### Orchestration WFA
| Fichier | Role |
|---------|------|
| `scripts/production/run_scheduler_wfa_phase.py` | Lance WFA avec labels |
| `scripts/production/freeze_hmm_labels.py` | Gele les labels HMM |
| `scripts/production/aggregate_k5_results.py` | Agregge resultats multi-seeds |

### Labels Valides
| Fichier | Description |
|---------|-------------|
| `outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv` | 3 regimes HMM (VALIDE) |
| `outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv` | 5 regimes (EN TEST) |
| `data/CYCLE_cash_bear.csv` | Cycle halving (VALIDE) |
| `data/ML_directional.csv` | Labels ML directionnel |

### Donnees
| Fichier | Description |
|---------|-------------|
| `data/BTC_FUSED_2h.csv` | Prix BTC 2h (Bitstamp+Binance fusionne) |

---

## MODULES src/ - ACTIFS vs EXPERIMENTAUX

### ACTIFS (utilises en production)
| Module | Role | Utilise par |
|--------|------|-------------|
| `src/checkpoint_manager.py` | Reprise apres crash | run_scheduler_wfa_phase.py |
| `src/ichimoku/risk.py` | Seuil perte journalier | ichimoku_pipeline_web_v4_8_fixed.py |

### EXPERIMENTAUX (pas en production)
| Module | Role | Status |
|--------|------|--------|
| `src/regime_hmm.py` | HMM detection regimes | Remplace par labels pre-calcules |
| `src/features_fourier.py` | Features Welch PSD | Utilise pour generer labels |
| `src/wfa.py` | WFA alternatif | Non utilise (scheduler custom) |
| `src/optimizer.py` | Optimisation random | Remplace par Optuna |
| `src/risk_sizing.py` | Simulation strategie | Dans pipeline principal |
| `src/stats_eval.py` | Calcul metriques | Dans pipeline principal |
| `src/regime_nhhm.py` | NHHM Markov-switching | ECHEC - abandonne |
| `src/ml_directional.py` | LightGBM directionnel | Teste, Sharpe faible |
| `src/funding_rate.py` | Funding rate Binance | Features pour NHHM |
| `src/volatility_targeting.py` | Leverage dynamique | Pas encore integre |
| `src/live_trader_adaptive.py` | Bot trading live | Pas teste en prod |
| `src/continuous_learning.py` | Refit auto HMM | Pas teste |
| `src/spectral/*` | Modules spectraux P2/P3 | Partiellement utilises |

---

## OU CHERCHER PAR SUJET

### "Je veux comprendre le backtest Ichimoku"
→ `ichimoku_pipeline_web_v4_8_fixed.py` lignes 800-1500

### "Je veux comprendre le WFA"
→ `scripts/production/run_scheduler_wfa_phase.py`
→ `docs/guides/METHODOLOGIE_COMPLETE.md`

### "Je veux voir les resultats K3/K5"
→ `outputs/wfa_phase_k3/` et `outputs/wfa_phase_k5/`
→ Fichiers JSON avec metriques par fold

### "Je veux comprendre les labels/regimes"
→ Labels HMM: `outputs/fourier/labels_frozen/`
→ Labels CYCLE: `data/CYCLE_cash_bear.csv`
→ Generation: `scripts/production/freeze_hmm_labels.py`

### "Je veux comprendre l'approche CYCLE (halving)"
→ `data/CYCLE_cash_bear.csv` - labels
→ `docs/prompts/PROMPT_IMPLEMENTATION_CYCLE.md` - contexte
→ Logique: CYCLE=0 → CASH, CYCLE=1 → TRADE

### "Je veux comprendre l'approche ML"
→ `src/ml_directional.py` - code LightGBM
→ `data/ML_directional.csv` - labels generes
→ `docs/reports/POST_MORTEM_NHHM_ECHEC.md` - pourquoi NHHM abandonne

### "Je veux lancer un test"
→ `scripts/production/run_scheduler_wfa_phase.py --help`
→ Exemple: `py -3 scripts/production/run_scheduler_wfa_phase.py --labels-csv data/CYCLE_cash_bear.csv --trials 30 --seed 42`

### "Je veux voir l'historique des decisions"
→ `docs/journals/` - journaux par date
→ `docs/guides/ARBRE_DECISION_ALPHA.md` - arbre de decision

---

## FLUX DE DONNEES

```
1. DONNEES BRUTES
   data/BTC_FUSED_2h.csv (OHLCV 2h)
          |
          v
2. GENERATION LABELS (one-shot, deja fait)
   scripts/production/freeze_hmm_labels.py
          |
          v
3. LABELS GELES
   outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv
   outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv
   data/CYCLE_cash_bear.csv
          |
          v
4. WFA PHASE-AWARE
   scripts/production/run_scheduler_wfa_phase.py
   (optimise Ichimoku par regime sur train, teste sur test)
          |
          v
5. RESULTATS
   outputs/wfa_phase_k3/seed_*/WFA_*.json
   outputs/wfa_phase_k5/seed_*/WFA_*.json
          |
          v
6. AGREGATION
   scripts/production/aggregate_k5_results.py
```

---

## RESULTATS VALIDES

| Approche | Sharpe | MDD | Fichiers |
|----------|--------|-----|----------|
| K3 | 0.99 | 13% | `outputs/wfa_phase_k3/` |
| CYCLE | 0.99 | ~15% | Tests avec `CYCLE_cash_bear.csv` |
| ML seul | 0.12 | 4.4% | `data/ML_directional.csv` |
| K5 | ? | ? | `outputs/wfa_phase_k5/` (en cours) |

---

## APPROCHES ABANDONNEES

| Approche | Pourquoi | Doc |
|----------|----------|-----|
| NHHM (Markov-switching) | Instable, predictions nulles | `docs/reports/POST_MORTEM_NHHM_ECHEC.md` |
| Wavelets | Pas mieux que Fourier | `docs/archive/FOURIER_WAVELETS_*.md` |
| Elliott Waves | Trop subjectif | `docs/archive/ELLIOTT/` |

---

## COMMANDES RAPIDES

```powershell
# Verifier que le projet marche
py -3 ichimoku_pipeline_web_v4_8_fixed.py --help

# Lancer WFA avec K3
py -3 scripts/production/run_scheduler_wfa_phase.py \
    --labels-csv outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv \
    --trials 30 --seed 42

# Voir progression K5
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
  $j = Get-Content $_.FullName | ConvertFrom-Json
  "$($_.Directory.Name): $($j.percent)%"
}

# Agreger resultats
py -3 scripts/production/aggregate_k5_results.py
```

---

## POUR CLAUDE: OU CHERCHER

Quand l'utilisateur demande quelque chose, voici ou regarder:

| Sujet | Fichiers a lire |
|-------|-----------------|
| Backtest/Strategie | `ichimoku_pipeline_web_v4_8_fixed.py` |
| WFA/Validation | `scripts/production/run_scheduler_wfa_phase.py` |
| Labels/Regimes | `outputs/fourier/labels_frozen/`, `data/*.csv` |
| Resultats | `outputs/wfa_phase_*/seed_*/` |
| Methodologie | `docs/guides/METHODOLOGIE_COMPLETE.md` |
| Historique decisions | `docs/journals/`, `docs/guides/ARBRE_DECISION_ALPHA.md` |
| Echecs/Post-mortems | `docs/reports/POST_MORTEM_*.md` |
| ML/NHHM | `src/ml_directional.py`, `src/regime_nhhm.py` |
| Config Ichimoku | `ichimoku_pipeline_web_v4_8_fixed.py` lignes 100-300 |

---

## FICHIERS A LA RACINE

| Fichier | Role | Critique? |
|---------|------|-----------|
| `ichimoku_pipeline_web_v4_8_fixed.py` | Moteur principal | OUI |
| `ichimoku_pipeline_web_v4_8.py` | Ancienne version | Non |
| `run_oos.py` | Script WFA alternatif | Non |
| `requirements.txt` | Dependances Python | OUI |
| `INDEX.md` | CE FICHIER | Reference |
| `README.md` | Intro projet | Reference |

---

*Derniere mise a jour: 2026-02-10*
