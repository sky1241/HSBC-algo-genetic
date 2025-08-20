Tests et résultats
==================

Contexte
--------
- Données: profils pipeline_web6 (H2), Ichimoku + ATR trailing
- Exécutions: Optuna 1000 trials multiples (seeds 42/123/777/999) avec baselines successives
- Paramètres de risque de référence: POSITION_SIZE=1%, LEVERAGE=10, MAX_POS_PER_SIDE=3

Top résultats (extraits)
------------------------
- n2 (1% / x10 / 3 pos): Equity 59 103 €, Max DD 4.8%, Min 952 €, Sharpe* 3.05, Trades 1023
- n1 (1% / x10 / 3 pos): Equity 24 536 €, Max DD 1.6%, Min 984 €, Sharpe* 3.67, Trades 884

Monte Carlo (1%)
----------------
- n1: Final × p5=4.92, p50=22.32, p95=131.66; Max DD médiane ≈21.5%; prob_loss=0.00%
- n2: Final × p5=9.26, p50=50.18, p95=406.27; Max DD médiane ≈22.0%; prob_loss=0.00%

Comparatif
----------
- Risque similaire (DD médian ~22%) mais potentiel et médiane de performance beaucoup plus élevés pour n2.
- Recommandation: retenir n2 comme base pour la suite (robustesse > rendement attendu).

Fichiers utiles
---------------
- Archives HTML: archives/*_MASTER_REPORT.html
- Résumés: outputs/MASTER_SUMMARY.csv
- Monte Carlo: outputs/MC_REPORT_*.txt et outputs/MC_DIST_COMPARE_*.png


