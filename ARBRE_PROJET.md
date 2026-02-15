# ARBRE HIVER - HSBC-algo-genetic

**Date:** 2026-02-10
**Version:** 1.0

---

## (1) TREE_SILHOUETTE

```
                                        [C1  Quality/Ship]
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                   [C2  Docs]                                  [C3  Agregation]
                                               │
          ═══════════════════════════════════════════════════════════════════
                                               │
               ┌───────────────┬───────────────┼───────────────┬─────────────┐
               │               │               │               │             │
         [B1  Labels]    [B2  HMM]      [B3  ML/NHHM]    [B4  Analysis]  [B5  Live]
         freeze_hmm      regime_hmm     ml_directional   scripts/        trader
               │               │               │          analysis/      adaptive
               │               │               │               │             │
               └───────────────┴───────────────┴───────────────┴─────────────┘
                                               │
                                        [T5  Metrics]
                                        Sharpe/MDD/CAGR
                                               │
                                        [T4  Scheduler]
                                        run_scheduler_wfa_phase.py
                                               │
                                        [T3  Labels System]
                                        K3/K5/CYCLE/ML/COMBINED
                                               │
                                        [T2  Pipeline Engine]
                                        ichimoku_pipeline_v4_8_fixed.py
                                        (3212 lignes - COEUR)
                                               │
                                        [T1  Data Layer]
                                        BTC_FUSED_2h.csv
                                               │
          ═════════════════════════════════════════════════════════════════════
          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  SOL  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          ═════════════════════════════════════════════════════════════════════
                                               │
               ┌───────────────┬───────────────┼───────────────┬─────────────┐
               │               │               │               │             │
         [R1  BTC]       [R2  Fourier]   [R3  HMM]       [R4  Seeds]    [R5  WFA]
         Crypto          Welch PSD       Regimes K       Determinisme   14 folds
         Only            Spectral        Detection       Reproductible  annuels
```

---

## (2) NODE_REGISTRY

```yaml
# RACINES (Fondations / Contraintes)
- id: R1
  label: "BTC Crypto Only"
  level: R
  parent: null
  desc: "Asset unique BTC, donnees fusionnees Bitstamp+Binance 2h"

- id: R2
  label: "Fourier/Spectral"
  level: R
  parent: null
  desc: "Welch PSD pour detection cycles, features spectrales"

- id: R3
  label: "HMM Regimes"
  level: R
  parent: null
  desc: "K regimes (K3 valide, K5 en test), labels pre-calcules"

- id: R4
  label: "Determinisme Seeds"
  level: R
  parent: null
  desc: "Reproductibilite via seeds Optuna, 30 seeds par config"

- id: R5
  label: "WFA Validation"
  level: R
  parent: null
  desc: "Walk-Forward Analysis 14 folds annuels, expanding window"

# TRONC (Core Architecture)
- id: T1
  label: "Data Layer"
  level: T
  parent: null
  desc: "OHLCV BTC 2h fusionne, ~60k lignes, 2010-2024"

- id: T2
  label: "Pipeline Engine"
  level: T
  parent: T1
  desc: "ichimoku_pipeline_web_v4_8_fixed.py - backtest + Optuna"

- id: T3
  label: "Labels System"
  level: T
  parent: T2
  desc: "K3, K5_1d_stable, CYCLE, ML, COMBINED - fichiers CSV"

- id: T4
  label: "WFA Scheduler"
  level: T
  parent: T3
  desc: "run_scheduler_wfa_phase.py - orchestration multi-seeds"

- id: T5
  label: "Metrics Engine"
  level: T
  parent: T4
  desc: "Sharpe, MDD, CAGR, equity curve - calcul et agregation"

# BRANCHES (Modules)
- id: B1
  label: "Label Generation"
  level: B
  parent: T3
  desc: "freeze_hmm_labels.py, generation labels HMM"

- id: B2
  label: "HMM Detection"
  level: B
  parent: T3
  desc: "src/regime_hmm.py, src/features_fourier.py"

- id: B3
  label: "ML/NHHM"
  level: B
  parent: T3
  desc: "src/ml_directional.py (actif), src/regime_nhhm.py (abandonne)"

- id: B4
  label: "Analysis Scripts"
  level: B
  parent: T5
  desc: "scripts/analysis/ - 63 scripts d'analyse et visualisation"

- id: B5
  label: "Live Trading"
  level: B
  parent: T2
  desc: "src/live_trader_adaptive.py, src/live_phase_adapter.py"

# CIME (Quality / Ship)
- id: C1
  label: "Quality & Ship"
  level: C
  parent: T5
  desc: "Validation multi-seeds, checkpoint, crash recovery"

- id: C2
  label: "Documentation"
  level: C
  parent: C1
  desc: "docs/guides/, journals/, INDEX.md"

- id: C3
  label: "Agregation"
  level: C
  parent: C1
  desc: "aggregate_k5_results.py, comparaison approches"
```

---

## (3) REFERENCES

### RACINES

**[R1  BTC Crypto Only]**
- (code) `data/BTC_FUSED_2h.csv` — donnees principales ~60k lignes
- (code) `scripts/analysis/fuse_btc_h2.py` — fusion Bitstamp+Binance
- (doc) `INDEX.md` lignes 63-64 — "Prix BTC 2h fusionne"
Notes: Asset unique, pas de multi-asset

**[R2  Fourier/Spectral]**
- (code) `src/features_fourier.py` — compute_fourier_features(), Welch PSD
- (code) `src/spectral/fourier_features.py` — module spectral P2
- (doc) `docs/archive/FOURIER_GUIDE.md` — methodologie Fourier
Notes: Base de la detection de cycles

**[R3  HMM Regimes]**
- (code) `src/regime_hmm.py` — fit_regime_model(), GaussianHMM
- (code) `outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv` — labels valides
- (code) `outputs/fourier/labels_frozen/BTC_FUSED_2h/K5_1d_stable.csv` — K5 en test
Notes: K3 valide (Sharpe 0.99), K5 en cours

**[R4  Determinisme Seeds]**
- (code) `scripts/production/run_scheduler_wfa_phase.py` lignes ~50-60 — --seed arg
- (code) `ichimoku_pipeline_web_v4_8_fixed.py` — random.seed(), np.random.seed()
Notes: 30 seeds minimum pour validation robuste

**[R5  WFA Validation]**
- (code) `scripts/production/run_scheduler_wfa_phase.py` — WFA annuel 14 folds
- (code) `src/wfa.py` — WFA alternatif (non utilise)
- (doc) `docs/guides/METHODOLOGIE_COMPLETE.md` — methodologie WFA
Notes: Expanding window, train sur historique, test sur annee suivante

---

### TRONC

**[T1  Data Layer]**
- (code) `data/BTC_FUSED_2h.csv` — OHLCV 2h, 2010-2024
- (code) `src/io_loader.py` — chargement et validation CSV
- (code) `data/README.md` — description donnees
Notes: Source unique de verite pour le projet

**[T2  Pipeline Engine]**
- (code) `ichimoku_pipeline_web_v4_8_fixed.py` — 3212 lignes, COEUR
- (code) lignes ~800-1500 — backtest Ichimoku
- (code) lignes ~100-300 — configuration parametres
- (code) `src/ichimoku/risk.py` — daily_loss_threshold()
Notes: Tout le backtest + optimisation Optuna dans un fichier

**[T3  Labels System]**
- (code) `data/CYCLE_cash_bear.csv` — labels CYCLE (valide)
- (code) `data/ML_directional.csv` — labels ML
- (code) `data/COMBINED_labels.csv` — fusion CYCLE+ML
- (code) `outputs/fourier/labels_frozen/` — labels HMM geles
Notes: 4 systemes de labels testables

**[T4  WFA Scheduler]**
- (code) `scripts/production/run_scheduler_wfa_phase.py` — orchestrateur principal
- (code) `src/checkpoint_manager.py` — crash recovery
- (code) `scripts/production/launch_30_seeds_k5.ps1` — lancement multi-seeds
Notes: Point d'entree pour tous les tests WFA

**[T5  Metrics Engine]**
- (code) `ichimoku_pipeline_web_v4_8_fixed.py` — calcul Sharpe/MDD/CAGR
- (code) `src/stats_eval.py` — metriques additionnelles
- (code) `scripts/production/aggregate_k5_results.py` — agregation seeds
Notes: Metriques cles: Sharpe > 0.8, MDD < 15%, Survie 100%

---

### BRANCHES

**[B1  Label Generation]**
- (code) `scripts/production/freeze_hmm_labels.py` — gel labels HMM
- (code) `scripts/analysis/downsample_labels_2h_to_1d.py` — stabilisation 1D
Notes: Labels pre-calcules, pas de refit en production

**[B2  HMM Detection]**
- (code) `src/regime_hmm.py` — GaussianHMM, fit_regime_model()
- (code) `src/features_fourier.py` — features Welch PSD
- (code) `src/spectral/hmm_features.py` — features HMM avancees
Notes: Utilise pour generation labels, pas en runtime

**[B3  ML/NHHM]**
- (code) `src/ml_directional.py` — LightGBM directionnel (actif)
- (code) `src/regime_nhhm.py` — NHHM Markov-switching (ABANDONNE)
- (doc) `docs/reports/POST_MORTEM_NHHM_ECHEC.md` — pourquoi NHHM abandonne
Notes: ML Sharpe 0.12, COMBINED Sharpe 0.91

**[B4  Analysis Scripts]**
- (code) `scripts/analysis/` — 63 scripts
- (code) `scripts/analysis/analyze_k5_1d_stable.py` — analyse K5
- (code) `scripts/analysis/plot_*.py` — visualisations
Notes: Scripts one-shot, pas critiques

**[B5  Live Trading]**
- (code) `src/live_trader_adaptive.py` — bot Binance adaptatif
- (code) `src/live_phase_adapter.py` — detection phase live
- (code) `src/volatility_targeting.py` — leverage dynamique
Notes: NON TESTE en production, experimental

---

### CIME

**[C1  Quality & Ship]**
- (code) `src/checkpoint_manager.py` — RobustRunner, crash recovery
- (code) `outputs/wfa_phase_k5/seed_*/PROGRESS.json` — tracking progression
- (code) `outputs/wfa_phase_k5/seed_*/CHECKPOINT.json` — reprise apres crash
Notes: 12 seeds K5 a ~87% completion

**[C2  Documentation]**
- (doc) `INDEX.md` — carte du projet
- (doc) `docs/README_ETAT.md` — etat actuel
- (doc) `docs/guides/METHODOLOGIE_COMPLETE.md` — methodologie
- (doc) `docs/journals/JOURNAL_2026-02-10.md` — journal du jour
Notes: Documentation maintenue a jour

**[C3  Agregation]**
- (code) `scripts/production/aggregate_k5_results.py` — agregation multi-seeds
- (code) `outputs/wfa_combined_test/seed_101/RESULTS_FINAL.json` — resultats
Notes: Comparaison K3 vs K5 vs CYCLE vs COMBINED

---

## (4) QUICK SUMMARY

- **Ce projet est surtout:** Un systeme de trading algorithmique BTC utilisant detection de regimes HMM + optimisation Ichimoku
- **Le tronc est:** `ichimoku_pipeline_web_v4_8_fixed.py` (3212 lignes) + WFA scheduler
- **Les branches dominantes sont:** Generation labels (B1), ML/NHHM (B3), Analysis (B4)
- **La contrainte racine la plus forte est:** WFA validation (R5) - 14 folds, 30 seeds, reproductibilite
- **Le risque structurel principal est:** Code monolithique (3212 lignes dans un fichier) + modules experimentaux non nettoyes

---

## LEGENDE RAPIDE

| Zone | Couleur mentale | Description |
|------|-----------------|-------------|
| R* | Marron | Fondations, contraintes non-negociables |
| T* | Vert fonce | Tronc, architecture core |
| B* | Vert clair | Branches, modules evolutifs |
| C* | Jaune/Or | Cime, qualite et livraison |

---

*Genere le 2026-02-10*
