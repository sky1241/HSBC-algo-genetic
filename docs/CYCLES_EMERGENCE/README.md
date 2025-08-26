### Méthodologie — Émergence des cycles

1) Source des signaux: `outputs/FOURIER_ALL_REPORTS.csv` (trié ancien→récent), colonnes P1–P3 et LFP (prix) + P1_vol–P3_vol et LFP_vol (volume).
2) Agrégation mensuelle: index des top‑3 par mois (2h/1d) pour Binance et Bitstamp; export CSV/JSON pour le scheduler.
3) Vue quotidienne: récapitulatif daily (P1–P3/LFP) pour les réajustements de cadence/recherche.
4) Qualité: se référer à `docs/QUALITY_REPORTS.md`.

Livrables à venir:
- Tables mensuelles (top‑3) et synthèse daily (prix/volume).
- Hooks d’export vers le scheduler HSBC (ranges Ichimoku + intensité pools).

