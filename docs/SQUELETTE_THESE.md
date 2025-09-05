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

- Reste à faire (priorités)
  - Intégrer les features Fourier au scheduler runtime (cadence et seeds) et brancher les JSON: `outputs/fourier/phase/<SYM>_<TF>/SCHEDULER_FOURIER_*.json`.
  - Valider les mappages \(P\rightarrow\) Ichimoku par phase en walk‑forward IS/OOS.
  - Monter des stress‑tests de robustesse (Monte Carlo, coûts, latence; sensibilité hyperparamètres).
  - Ajouter STFT/ondelettes pour la localisation temporelle des régimes.
  - Automatiser le recalcul quotidien et la mise à jour des rapports (MD/PDF) et conclusions datées.
  - Intégrer des widgets Fourier dans `MASTER_REPORT.html` (lecture rapide P1..P6/LFP).
  - Exporter les agrégats vers Google Sheets; alertes QC (gaps) automatisées; tests unitaires des fonctions/rapports; optimisation perf de Welch; enrichissement doc; extension à ETH/altcoins; finaliser TOP5/TOP6.
