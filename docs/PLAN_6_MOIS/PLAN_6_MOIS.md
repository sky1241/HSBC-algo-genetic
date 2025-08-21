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


