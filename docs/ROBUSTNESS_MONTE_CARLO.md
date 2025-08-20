Robustesse — IS/OOS & Monte Carlo
=================================

IS/OOS (80/20)
--------------
Exécution:

pwsh
$env:POSITION_SIZE='0.01'; $env:LEVERAGE='10'; $env:MAX_POS_PER_SIDE='3'
.\.venv\Scripts\python.exe -u .\outputs\validate_robustness.py

Sortie: outputs/ROBUSTNESS_REPORT_*.txt

Monte Carlo (block bootstrap)
-----------------------------
Exécution rapide (baseline auto depuis outputs/):

pwsh
$env:POSITION_SIZE='0.01'; $env:LEVERAGE='10'
.\.venv\Scripts\python.exe -u .\outputs\monte_carlo_resample.py

Distribution comparée (2 baselines):

pwsh
.\.venv\Scripts\python.exe -u .\outputs\plot_mc_distribution.py --baseline .\outputs\baseline_n1.json --label n1 --baseline .\outputs\baseline_n2.json --label n2

Résultats typiques (1%):
- n1: Final × p5=4.92, p50=22.32, p95=131.66; DD médian ≈21.5%
- n2: Final × p5=9.26, p50=50.18, p95=406.27; DD médian ≈22.0%

Interprétation
--------------
- Préférer la configuration dont la distribution MC offre un meilleur p50/p5 pour une DD médiane comparable.
- Éviter les réglages qui augmentent fortement le DD (ex: 3% par trade → DD > 50% médian).


