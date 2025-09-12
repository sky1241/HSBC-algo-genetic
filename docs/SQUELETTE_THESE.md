# Squelette de thèse – Optimisation d’Ichimoku via Fourier et cycles de Halving BTC

## 🏛 Partie I – Introduction & Contexte
- **Introduction générale**
  - Présentation du contexte financier et des spécificités des marchés crypto.
  - Objectif: Stratégie systématique Ichimoku + ATR, robuste (phase-aware halving) avec seeds contrôlés.
  - Périmètre: BTC (H2 prioritaire), extension daily si utile (plus d’historique).
  - Mise en avant de l’importance des indicateurs techniques et des cycles dans l’analyse de marché.
- **Problématique & Objectifs**
  - Faut-il recourir à des réglages universels ou adaptés aux phases de halving ?
  - Hypothèse : l’Ichimoku peut être optimisé via l’analyse fréquentielle (Fourier) pour mieux capter ces cycles.
- **Méthodologie globale**
  - Source des données (historique BTC depuis 2011), indicateurs étudiés (Ichimoku), méthode d’analyse (Fourier) et protocole de backtesting.

## 📚 Partie II – Revue de Littérature
1. **Cycles de marché**
   - Théories cycliques classiques (Kondratieff, Elliott, structures harmoniques).
   - Littérature spécifique aux cryptomonnaies (cycles de halving, volatilité).
2. **Indicateurs techniques et optimisation**
   - Positionnement traditionnel de l’Ichimoku.
   - Comparaison avec d’autres indicateurs (SAR, MA, RSI…).
3. **Méthodes fréquentielles**
   - Applications de la transformée de Fourier à la finance.
   - Détection de patterns cycliques dans les séries temporelles.

## 🛠 Partie III – Méthodologie
1. **Description des données**
   - Sources & fréquences:
     - Binance: `BTC/USDT` en 2h et 1d, historique exploité depuis 2017‑08‑17 jusqu’à aujourd’hui; CSV: `data/BTC_USDT_2h.csv`, `data/BTC_USDT_1d.csv`.
     - Bitstamp: `BTC/USD` en 1d (2011‑08‑18 → 2024‑08‑30) et 2h obtenu par resampling de 1h (agrégations OHLCV conservatrices); CSV: `data/BTC_USD_1d.csv`, `data/BTC_USD_2h.csv`.
   - Contrôle qualité (QC):
     - Rapports automatiques écrits dans `outputs/quality_reports/` et synthèse `docs/QUALITY_REPORTS.md` (export HTML/PDF dans `outputs/reports/`).
     - Vérifications: monotonie/duplicates des timestamps, gaps vs fréquence attendue, NaN OHLCV, cohérence OHLC (`low ≤ {open,close} ≤ high`), volumes nuls/négatifs, outliers de retours (|z| > 4).
     - État: aucune anomalie bloquante; données exploitables pour Fourier et cartographie Ichimoku.
   - Découpage par périodes de halving (avant/après).
2. **Ichimoku Kinko Hyo**
   - Formules des composantes (Tenkan, Kijun, SSA/SSB, Chikou).
   - Paramètres standards et paramétrisation libre.
3. **Analyse fréquentielle**
   - Rappel mathématique de la transformée de Fourier.
   - Estimation PSD via Welch sur fenêtres roulantes (annuelle et mensuelle).
   - Extraction des trois cycles dominants \(P1,P2,P3\) et du ratio basse fréquence \(\mathrm{LFP}\); extension volume: \(P1_{vol},P2_{vol},P3_{vol},\mathrm{LFP}_{vol}\).
   - Objectif : cartographier \(P_k\) vers des plages Ichimoku (heuristique: `kijun ≈ P/2`, `tenkan ≈ P/8–P/6`, `senkou_b ≈ P`, `shift ≈ kijun/2`).
4. **Cadre expérimental**
   - Stratégie de backtest (temps, signaux, gestion du risque).
   - Variables d’évaluation (winrate, drawdown, profit factor).

## 📊 Partie IV – Résultats
1. **Résultats bruts**
   - Tableaux comparatifs des performances par phase de marché.
   - Indicateurs statistiques (taux de réussite, ratios de rendement/risque).
2. **Comparaison universel vs spécifique**
   - Réglages constants sur toute la période.
   - Réglages ajustés par cycle de halving.
3. **Apport de Fourier**
   - Impact de l’analyse fréquentielle sur la calibration des paramètres Ichimoku.
   - Visualisation des cycles et corrélation avec les périodes de performance.
   - H2 vs D1 (depuis 2020, rolling monthly): écart faible sur les périodes dominantes (ex.: \(\Delta P1\) ≈ +0.24 jour H2–D1; \(\Delta P2\) ≈ +0.05 jour; \(\Delta P3\) ≈ +0.31 jour). LFP: \(\Delta\) ≈ −0.005. Rolling annual: \(\Delta\mathrm{LFP}\) ≈ −3.6e−4.
4. **Analyse comparative**
   - Synthèse des résultats, bénéfices et limites de chaque approche.
   - Implications pratiques pour le trading algorithmique.

## 🔎 Partie V – Discussion
- **Interprétation des résultats**
  - Validité de l’hypothèse initiale.
  - Robustesse statistique et limites du backtest.
  - Les écarts H2/D1 étant modestes, une stratégie « phase-aware » peut s’appuyer sur D1 pour la robustesse structurelle et sur H2 pour l’ajustement fin/scheduler sans divergence majeure.
- **Risques et possibilités d’amélioration**
  - Risque d’overfitting, limites des données historiques.
  - Potentiel d’extension à d’autres actifs (ETH, altcoins).
  - Vers une auto-optimisation via IA ou apprentissage automatique.

## 🏁 Partie VI – Conclusion
- **Résumé des contributions**
  - Optimisation de l’Ichimoku via cycles de halving et Fourier.
  - Validité (ou non) de réglages universels.
- **Perspectives**
  - Intégration dans un framework auto-adaptatif.
  - Applications possibles en temps réel et dans l’IA (paramétrage automatisé).

## 📚 Bibliographie
- Articles académiques sur les cycles de marché, l’analyse fréquentielle et les indicateurs techniques.
- Sources de données (Binance, Yahoo Finance, Coin Metrics).
- Ouvrages théoriques sur l’Ichimoku, les cycles boursiers et la transformée de Fourier.

## 📅 Mise à jour du 2025-08-26
- Réalisé aujourd’hui
  - Comparatif des phases 3/5/6 (durées en jours et en barres H2, distance au halving médiane, P1..P6 médians, LFP moyen) sur BTC pour Binance (USDT) et Bitstamp (USD) en 2h/1d.
  - Production des tableaux par label set: `outputs/fourier/compare/<SYM>_<TF>/SUMMARY_{regime3,phase5,phase6}.csv` et segments détaillés `SEGMENTS_*.csv`.
  - Consolidation dans `outputs/fourier/compare/ALL_SUMMARIES_BY_LABELSET.csv` et page dédiée `docs/PHASE_LABELS/COMPARATIF_DUREES_FOURIER_HALVING.md` (PDF exporté dans `outputs/reports/COMPARATIF_DUREES_FOURIER_HALVING.pdf`).
  - Ajout au master index `docs/INDEX_THESE_FOURIER_PHASES.md`.
  - Enrichissement méthodologique: ajout des métriques de distance au halving (début/milieu/fin de segment) et agrégat `dsh_mid_median` pour lecture temporelle.

- ## 📅 Mise à jour du 2025-09-05
- - Réalisé aujourd’hui
-   - HMM (BTC_FUSED 2h) K ∈ [2..10], 60 seeds (30×2): agrégat `outputs/fourier/hmm/BTC_FUSED_2h/HMM_K_SELECTION_AGG.csv` + rapport `docs/HMM_BTC_FUSED_2h.md`.
-   - Résultat data‑driven: BIC_median et LL_OOS_median s’améliorent en montant K; reco K=10.
- - Prochaines actions
-   - Figer K=3 et K=5 (labels) pour comparaison thèse; lancer optimisation Ichimoku “fixe vs par phase” (30×2 seeds) par fenêtres halving.

## 📅 Mise à jour du 2025-09-06
- Réalisé aujourd’hui
  - Finalisation HMM K‑grid (BTC_FUSED 2h), exports par K et agrégat; exécutions Optuna BTC (batches), génération Top/Best et rapport `docs/TOP_BEST_RESULTS.md` avec conclusions.
- Conclusions clés
  - K=10 optimal au sens BIC/LL_OOS; labels thèse figés K=3/5.
  - Plages Ichimoku récurrentes BTC/USDT: Tenkan 42–45, Kijun 85–111, Senkou_B 215–225, Shift 23–29, ATR× ≈ 4.8–5.3.
- À surveiller
  - Snapshot equity ≈ 7 470 € à re‑vérifier (horizon/scope) avant inclusion; nettoyage des NaN dans “Best par symbole”; QC volumes nuls (1 cas) à corriger.
 - Avancement complémentaire
   - Activation du mode “full history” pour Optuna (`USE_FUSED_H2=1`) afin d’utiliser `data/BTC_FUSED_2h.csv`.
   - Exports Top/Best filtrés (multi‑symbole/NaN exclus). Prochaine étape: baseline 9‑26‑52, optimisation “par phase” (K=3/5), et comparaison mensuelle vs annuelle.

## 📅 Mise à jour du 2025-09-07
- Réalisé aujourd’hui
  - Consolidation “best per symbol” et Top‑5 global à partir des JSON de portefeuille propres (gating MDD/min_equity/liquidations/margin). Sorties:
    - `outputs/BEST_PER_SYMBOL.{csv,json,txt}` et `outputs/BEST_PER_SYMBOL_TOP_DECILE_DDMIN.{csv,json,txt}`
    - `outputs/top_results.json`
    - `docs/TOP_BEST_RESULTS.md` (Top‑5 global + tableau Best par symbole + conclusions)
- Faits saillants
  - BTC/USDT baseline robuste: equity ≈ 1 948 €, DD ≈ 4.60%, 109 trades, params ≈ (tenkan 42, kijun 83, senkou_b 215, shift 27, ATR× 5.2).
  - Top‑5 global extrait des snapshots récents; base de comparaison “fixe vs par phase”.

- Reste à faire (priorités)
  - Intégrer les features Fourier au scheduler runtime (cadence et seeds) et brancher les JSON: `outputs/fourier/phase/<SYM>_<TF>/SCHEDULER_FOURIER_*.json`.
  - Valider les mappages \(P\rightarrow\) Ichimoku par phase en walk‑forward IS/OOS.
  - Monter des stress‑tests de robustesse (Monte Carlo, coûts, latence; sensibilité hyperparamètres).
  - Ajouter STFT/ondelettes pour la localisation temporelle des régimes.
  - Automatiser le recalcul quotidien et la mise à jour des rapports (MD/PDF) et conclusions datées.
  - Intégrer des widgets Fourier dans `MASTER_REPORT.html` (lecture rapide P1..P6/LFP).
  - Exporter les agrégats vers Google Sheets; alertes QC (gaps) automatisées; tests unitaires des fonctions/rapports; optimisation perf de Welch; enrichissement doc; extension à ETH/altcoins; finaliser TOP5/TOP6.

## 📅 Mise à jour du 2025-09-08
- Réalisé aujourd’hui
  - Baselines BTC/USDT 2h (full history, fused):
    - 9-26-52-26, ATR×3: equity 26 618€, CAGR 26.37%, MDD 18.06%, trades 1 681
    - 9-26-52-26, ATR×5: equity 46 217€, CAGR 31.44%, MDD 18.65%, trades 1 408
  - Nettoyage des volumes nuls/négatifs dans `data/BTC_FUSED_2h.csv` (2 429 lignes corrigées)
  - Lancement walk-forward mensuel (2017-08 → 2025-08) sans Optuna
- À suivre
  - Comparer baselines vs Optuna full history (batch 1), puis lancer batch 2 (seeds alternatifs)

## 📅 Mise à jour du 2025-09-12
- Réalisé aujourd’hui
  - Correction de la baseline: exécutions BTC‑only sur l’historique fused (2h) avec paramètres fixes 9‑26‑52‑26 et ATR×3/×5. Lancements en cours, logs dans `outputs/baseline_btc_only/`.
  - Lancement d’un walk‑forward annuel (scheduler) BTC‑only fused avec Optuna (≈200 trials, jobs=1). Script dédié: `scripts/run_scheduler_wfa.py`; logs `outputs/scheduler_annual_btc/`.
  - Création d’un runner baseline explicite BTC‑only: `scripts/run_btc_baseline_fixed.py`.
- En cours
  - Optuna Batch‑1 segment 3 (seeds 20–30, 6h, fused) et passe d’exploitation (5 seeds, 1200 trials) tournent; attente de snapshots consolidés.
  - Baselines BTC‑only ATR×3/×5 et WFA annuel: attente des métriques finales avant comparaison.
- Prochaines actions
  - Lancer le WFA mensuel (roulant 12 mois) et comparer « mensuel vs annuel » sur la baseline BTC‑only.
  - Stratégie par phase (labels HMM gelés K=3/5/8 disponibles, K=10 pour référence): optimisation + walk‑forward par phase, puis comparaison vs baseline fixe.
  - Reporting: mise à jour `docs/TOP_BEST_RESULTS.md` et export PDF; ajout d’un tableau comparatif baseline vs phase‑aware.
- Points de vigilance
  - Les baselines précédentes incluaient ETH/DOGE par erreur pour la comparaison BTC‑only; elles sont écartées des comparatifs. Les nouvelles baselines seront strictement BTC‑only sur `BTC_FUSED_2h.csv`.
  - Normalisation temporelle UTC (tz‑naive) validée pour éviter les erreurs Pandas; chargement fused activé via `USE_FUSED_H2=1`.
