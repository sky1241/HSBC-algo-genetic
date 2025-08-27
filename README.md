HSBC algo genetic
=================

(basic optimisation — Monte Carlo sim — backtest Ichimoku BK + Lyanapoulov)

Pipeline de backtest/optimisation (Ichimoku + ATR) avec génération de rapports, validations de robustesse (IS/OOS, Monte Carlo) et script de trading Testnet Binance.

Points clés
-----------
- Optimisation par essaim (Optuna) avec gestion des seeds et baseline paramétrables
- Rapports HTML « MASTER_REPORT » avec graphiques intégrés (base64), Top résultats et résumé CSV
- Sélection automatique des meilleurs paramètres par paire et baseline dérivée
- Validations de robustesse: In-Sample/Out-of-Sample et Monte Carlo (block bootstrap) avec synthèse p5/p50/p95
- Orchestration Windows via PowerShell (run_full_analysis.ps1) avec suivi « pruned trials » en direct
- Live trading (Testnet) avec trailing stop serveur Binance basé sur l’ATR

Démarrage rapide (Windows / PowerShell)
--------------------------------------
1) Installer Python 3.10+ et Git (facultatif pour versionnement)
2) Dans le dossier du projet:

pwsh -NoProfile -File .\run_full_analysis.ps1 -ProfileName pipeline_web6 -Label bestpairs_eq_s999 -Trials 1000 -Seed 999 -BaselineJson .\outputs\BEST_BASELINE.json -PreArchive:$true -OpenStatus -OpenReport

- OpenStatus ouvre un Notepad qui affiche en direct le nombre de « pruned trials »
- OpenReport ouvre automatiquement le rapport final

Voir docs/USAGE.md pour toutes les options (baseline personnalisée, exécution en foreground, etc.).

Structure
---------
- ichimoku_pipeline_web_v4_8_fixed.py: coeur du pipeline (backtest, Optuna, enregistrement courbes)
- outputs/: scripts auxiliaires (génération rapports, Monte Carlo, robustesse, etc.) et résultats
- run_full_analysis.ps1: orchestration bout‑en‑bout (venv, deps, run, post‑traitements, archivage)
- docs/: documentation produit (roadmap, mode d’emploi, résultats, robustesse, live)
 - docs/FORMULES_ET_EXEMPLES.md: formules (LaTeX), définitions (pourquoi/comment) et exemples chiffrés

Résultats récents (extraits)
----------------------------
- Jeu n2 (1% par trade, levier 10, max 3 pos/side)
  - Equity: 59 103 €; Max DD: 4.8%; Min: 952 €; Sharpe*: 3.05; Trades: 1023
  - Monte Carlo (1%): final × p5=9.26, p50=50.18, p95=406.27; DD médiane ≈22%; prob_loss=0.00%
- Jeu n1 (1% par trade)
  - Equity: 24 536 €; Max DD: 1.6%; Min: 984 €; Sharpe*: 3.67; Trades: 884
  - Monte Carlo (1%): final × p5=4.92, p50=22.32, p95=131.66; DD médiane ≈21.5%; prob_loss=0.00%

Comparatif: n2 domine en rendement attendu (MC) à DD similaire. Voir docs/TESTS_AND_RESULTS.md.

Robustesse
----------
- IS/OOS (80/20) et Monte Carlo: scripts et mode d’emploi dans docs/ROBUSTNESS_MONTE_CARLO.md.
- Recommandation actuelle: 1% par trade, levier 10, max 3 positions par côté.

English docs
------------
- Executive report: `docs/HSBC_REPORT_EN.md`
- Formulas and examples: `docs/FORMULAS_AND_EXAMPLES.md`

Live (Testnet)
--------------
- Script: outputs/live_trader_testnet.py
- Trailing stop serveur Binance (TRAILING_STOP_MARKET) avec taux de callback dérivé de l’ATR
- Variables dans outputs/TESTNET_ENV.txt (API keys Testnet, position_size, leverage, garde‑fous)
- Voir docs/LIVE_TESTNET.md.

Roadmap et historique
---------------------
- Roadmap: docs/ROADMAP.md
- Historique (daté): HISTORIQUE_MODIFICATIONS_HIER.md

Licence
-------
Ce projet est distribué sous licence MIT — voir [LICENSE](LICENSE). Usage à des fins éducatives et expérimentales. Le trading comporte des risques.


