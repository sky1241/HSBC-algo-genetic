Plan 6 mois — Thèse Doctorale Light (Ichimoku + ATR)
Date: 2025-08-21

Objectif: Focus BTC (H2) d’abord, robustesse via altcoins majeurs ensuite. Cadre reproductible, WFA, reporting et prêt à publier (arXiv/SSRN) en 6 mois.

Mois 1 — Revue & cadrage
- Construire une revue rapide (papers, posts, livres) et synthèse tendances
- Livrables:
  - Tableau littérature: `docs/PLAN_6_MOIS/templates/lit_review_table.md`
  - Périmètre de la stratégie (hypothèses, métriques, risques)

Mois 2 — Données & robustesse multi-actifs
- Étendre et nettoyer BTC (H2, ≥ 5–8 ans), ajouter altcoins majeurs (ETH, DOGE) pour cross‑validation
- Livrables:
  - Datasets CSV gelés + hash (BTC/ETH/DOGE)
  - Note de qualité (gaps, volumes, timestamps)

Mois 3 — Standardisation des backtests
- Normaliser la journalisation des runs (config, période, résultats) en YAML/JSON
- Livrables:
  - Modèles: `backtest_config_template.yaml` et `.json`
  - Procédure d’archivage (labels avec seed/phase)

Mois 4 — Validation (WFA / CV)
- Mettre en place walk‑forward annuel et/ou validation croisée par phases
- Livrables:
  - Rapport WFA (IS/OOS), variance inter‑seeds, MC

Mois 5 — Graphes & reporting
- Script Python pour générer automatiquement les visuels (equity, drawdown, comparatifs)
- Livrables:
  - `scripts/generate_backtest_graphs.py` + graphs `outputs/graphs/`

Mois 6 — Publication
- Préparer le papier (format, licence) arXiv/SSRN; pack de réplication
- Livrables:
  - PDF/TeX/MD final + dépôt (GitHub release)

Annexes & templates
- Littérature: `docs/PLAN_6_MOIS/templates/lit_review_table.md`
- Backtests: `docs/PLAN_6_MOIS/templates/backtest_config_template.yaml|json`
- Graphes: `scripts/generate_backtest_graphs.py` (CLI)


Suivi d'avancement — 2025-08-26
- Données & QC
  - Binance BTC/USDT: H2 et D1 à jour; Bitstamp BTC/USD: D1 étendu (2011→2024) et 2h par resampling 1h.
  - Rapports de qualité générés (`docs/QUALITY_REPORTS.md`), aucune anomalie bloquante.
- Fourier
  - Calculs rolling annual/mensuel: P1–P3, LFP, et variantes volume (PVOL/LFPv). Rapports consolidés (H2, D1, Bitstamp) + CSV agrégé `outputs/FOURIER_ALL_REPORTS.csv`.
- Documentation
  - Thèse mise à jour (sources, QC, H2 vs D1, conclusions préliminaires), dossier `docs/FOURIER_CONCLUSIONS/` créé.

Prochaines étapes (priorisées)
1) Stratifier par phases de halving (Accu/Expan/Euph/Distrib/Bear) et quantifier Δ(P1–P3, LFP) par phase; tableaux synthèse.
2) Dossier « émergence des cycles »: indexer top‑3 mensuels puis déclinaison daily; exports compatibles scheduler.
3) Intégration pipeline: mapping P→(tenkan,kijun,senkou_b,shift) + gating par LFP/volume; validation IS/OOS + MC.
4) Mise en page thèse continue (ajout figures et tableaux synthèse au fil de l’eau).



Mise à jour — 2025-09-24 (actions concrètes K3/K5, phase-aware)

- Objectif 1 (Semaine 1–3): stabiliser K3/K5 par phase
  - Lancer/achever 30 seeds par K (phase puis fixed), 300 trials, MDD≤50%, DATA locale, résultats sur E:.
  - Agrégation par phase: médiane/IQR des paramètres (tenkan/kijun/senkouB/shift/atr_mult) et des métriques (Calmar, Sharpe, Lyap, rendement mensuel OOS).
  - Critères de stabilité: IQR/median ≤ 10–15% par phase; si non atteint, augmenter les seeds.

- Objectif 2 (Semaine 3–4): métriques mensuelles OOS par phase
  - Extraire rendements mensuels OOS par phase; calculer médiane, IQR, CV; filtrer sous MDD≤50%.
  - Sélectionner K (3 vs 5) et mode (phase vs fixed) par Calmar/Sharpe/Lyap, pondérés, avec contrainte MDD.

- Objectif 3 (Semaine 5): timeframe gagnant
  - Comparer H2 vs D1 (avec anti‑alias D1) sur le meilleur K/mode (réglages figés par phase).
  - Retenir le timeframe par Calmar (MDD≤50%) + robustesse mensuelle.

- Objectif 4 (Semaine 6): documentation et release
  - Mettre à jour `docs/FOURIER_CONCLUSIONS/` (conclusion datée) et `docs/TOP_BEST_RESULTS.md`.
  - Générer graphs/tableaux; préparer un draft de publication et pack de réplication.
