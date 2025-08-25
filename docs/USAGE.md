Usage — pipeline_web6
=====================

Exécuter une optimisation complète
----------------------------------

pwsh -NoProfile -File .\run_full_analysis.ps1 -ProfileName pipeline_web6 -Label mylabel -Trials 1000 -Seed 999 -BaselineJson .\outputs\BEST_BASELINE.json -PreArchive:$true -OpenStatus -OpenReport

Paramètres utiles
-----------------
- ProfileName: toujours pipeline_web6 dans ce repo
- Trials: nombre de trials Optuna (ex: 1000)
- Seed: seed de reproductibilité (ex: 42, 123, 777, 999)
- BaselineJson: JSON {symbol: {tenkan,kijun,senkou_b,shift,atr_mult}}
- OutDir: dossier de sortie (défaut: outputs)
- OpenStatus: ouvre un Notepad avec le compteur « pruned »
- OpenReport: ouvre le MASTER_REPORT à la fin
- Foreground: exécute dans la fenêtre courante (utile pour le debug)

Baselines
---------
- Générer depuis les snapshots: outputs/select_best_by_symbol.py
  - Produit: BEST_PER_SYMBOL.json, BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json, BEST_BASELINE.json
- Baseline custom: créer un JSON dans outputs/ puis passer son chemin à -BaselineJson

Rapports
--------
- MASTER_REPORT: outputs/MASTER_REPORT.html (copie en archives/*_MASTER_REPORT.html)
- Résumé Top3: outputs/MASTER_SUMMARY.csv
- Graphes timeline: outputs/build_graphs_from_snapshots.py

Export des meilleurs résultats
------------------------------
- `python outputs/export_top_results.py`
  - Lit les rapports `shared_portfolio_*` et produit `outputs/top_results.json`
- Serveur FastAPI optionnel pour consulter ces données :
  - `uvicorn src.ichimoku.api:app --reload`
  - Endpoint: http://localhost:8000/top-results

Environnement de backtest
-------------------------
- POSITION_SIZE, LEVERAGE, MAX_POS_PER_SIDE lus via variables d’environnement par le backtester


