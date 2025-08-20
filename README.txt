Ichimoku Pipeline (pipeline_web6)
==================================

1) Double-clique `run_pipeline.bat`
   - crée un venv, installe ccxt/pandas/numpy
   - lance: python ichimoku_pipeline_web_v4_8.py pipeline_web6 --trials 1000000 --seed 42

2) OU via Cursor/VS Code:
   - Ouvre le dossier
   - Terminal:
       py -3 -m venv .venv
       .\.venv\Scripts\Activate.ps1
       pip install --upgrade pip
       pip install -r requirements.txt
       py -3 .\ichimoku_pipeline_web_v4_8.py pipeline_web6 --trials 1000000 --seed 42
   - Option “Run Task…”: `Run Ichimoku pipeline`

3) Résultats:
   - outputs\runs_pipeline_web6_*.csv
   - outputs\top_pipeline_web6_*.csv
   - outputs\best_params_per_symbol_*.json (Optuna par paire)
   - outputs\runs_best_per_symbol_*.csv (Backtests finaux par paire)

Notes:
- Le filtre 'au-dessus du nuage' tient compte du décalage `shift` (1..99) **sans look-ahead**.
- Les données OHLCV sont récupérées depuis Binance via ccxt et mises en cache dans `data/`.

Mode optimisation Optuna (walk-forward annuel + ASHA):
- CLI Python:
  - Global (tous symboles mêmes paramètres): `python -c "from ichimoku_pipeline_web_v4_8_fixed import optuna_optimize_profile; optuna_optimize_profile('pipeline_web6', n_trials=50, jobs=2, fast_ratio=0.25, start_year=2019, end_year=2024)"`
  - Par paire (paramètres indépendants): `python -c "from ichimoku_pipeline_web_v4_8_fixed import optuna_optimize_profile_per_symbol; optuna_optimize_profile_per_symbol('pipeline_web6', n_trials=5000, jobs=4, fast_ratio=0.5, start_year=2019, end_year=2024)"`
- Détails:
  - Folds annuels automatiques
  - Paramètres échantillonnés avec contraintes (Tenkan ≤ Kijun ≤ SenkouB)
  - Pruning ASHA, score combiné Sharpe/CAGR/MaxDD/Stabilité
  - Sauvegarde `best_params_*.json` dans `outputs/`

---

Mises à jour clés (13 août 2025):
- Position sizing actuel: 1% du capital par trade; 3 entrées max par côté/symbole
- Résultats 5 ans (params fixes 9-26-52-26, ATR=3.0, 2h):
  - BTC/USDT ≈ -1.277, ETH/USDT ≈ 0.108, DOGE/USDT ≈ 0.041
  - Portefeuille unique 1000€ (réparti 1/3; négatifs plafonnés à 0€): ≈ 49.52€
- Correctifs réalisme: equity pondérée par `position_size`, funding/rollover sur notional ouvert
- Barre de progression restaurée pour les runs fixes/optimisation
- Optimisation par paire (Optuna ASHA, folds annuels) en cours: 5000 essais/pair (jobs=4, fast_ratio=0.5)

Annexes:
- Hypothèses/Limites, Reproductibilité, Glossaire: voir `LOGIQUE_PROGRAMME.md`
