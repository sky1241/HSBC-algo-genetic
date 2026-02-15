# Log de Reorganisation - 2026-02-10

## Resume

Reorganisation du projet sans suppression de fichiers. Structure clarifiee.

## Sous-dossiers crees

### docs/
- `guides/` - Documentation technique et guides
- `prompts/` - Prompts de session Claude/ChatGPT
- `reports/` - Rapports d'analyse et resultats
- `journals/` - Journaux de session
- `archive/` - Ancienne documentation

### scripts/
- `production/` - Scripts de production valides
- `analysis/` - Scripts d'analyse one-shot
- `experimental/` - Scripts en test
- `archived/` - Anciens scripts

## Fichiers deplaces

### docs/guides/
- GUIDE_VALIDATION_AUTO.md
- ARBRE_DECISION_ALPHA.md
- FOURIER_GUIDE.md
- USAGE.md
- METHODOLOGIE_COMPLETE.md
- FORMULAS_AND_EXAMPLES.md
- FORMULES_ET_EXEMPLES.md

### docs/prompts/
- PROMPT_*.md (tous les fichiers PROMPT_)

### docs/journals/
- JOURNAL_*.md (tous les fichiers JOURNAL_)

### docs/reports/
- ANALYSE_*.md
- REPORT_*.md
- CSV_AGG_REPORT_*.md
- FOURIER_RAPPORTS_*.md
- HMM_*.md
- TOP_BEST_RESULTS.md
- WFA_SUMMARY_COMPUTED.md
- QUALITY_REPORTS.md
- POST_MORTEM_*.md
- DEEP_RESEARCH_*.md
- CHATGPT_*.md

### docs/archive/
- ALGORITHME_ET_SEED_FR.md
- ETAT_PROJET_20251016.md
- FOURIER_*.md (divers)
- HSBC_*.md
- ROADMAP*.md
- Et autres fichiers anciens

### scripts/production/
- run_scheduler_wfa_phase.py
- run_scheduler_wfa.py
- run_walkforward.py
- freeze_hmm_labels.py
- aggregate_k5_results.py
- auto_aggregate_wfa.py
- launch_*.ps1 (tous)
- monitor_k5.ps1
- run_wfa_seed.ps1
- refresh_*.ps1

### scripts/analysis/
- analyze_*.py
- build_*.py
- compare_*.py
- summarize_*.py
- plot_*.py
- quick_*.py
- fourier_*.py
- wavelet_*.py
- Et autres scripts d'analyse

### scripts/experimental/
- test_*.py
- debug_*.py
- smoke_optuna.py

### scripts/archived/
- run_btc_baseline_fixed.py
- run_btc_optuna_seeds*.py
- run_hmm_kgrid_seeds.py
- watch_*.py
- push_now.bat

## Fichiers crees
- docs/README_ETAT.md - Etat actuel du projet
- docs/REORGANISATION_LOG_2026-02-10.md - Ce fichier

## Fichiers modifies
- README.md - Mis a jour avec nouvelle structure

## Validation
- Imports Python: OK
- Processus K5: 13 actifs (non interrompus)
- Structure: OK

## Note
L'erreur `ModuleNotFoundError: ichimoku_pipeline_web_v4_8_fixed` dans run_scheduler_wfa_phase.py est pre-existante (non causee par cette reorganisation).
