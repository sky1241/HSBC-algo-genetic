### Thèse — Fourier & Phases (plan de lecture et segmentation)

0) Prérequis & données
- Données/QC: `docs/QUALITY_REPORTS.md` (rapports), CSV: `data/` (BTC_USDT 2h/1d, BTC_USD 2h/1d)

1) Cœur Fourier (méthodo)
- Guide d’usage: `docs/FOURIER_GUIDE.md`
- Stratégie & mapping Ichimoku: `docs/FOURIER_STRATEGIE_FR.md`
- Rappel & Welch/halving: `docs/FOURIER_SERIES_RAPPEL_FR.md`, `docs/FOURIER_SERIES_WELCH_HALVING_FR.md`

2) Rapports consolidés (lecture des graphes)
- Binance 2h / 1d: `docs/FOURIER_RAPPORTS_BTC_2h.md`, `docs/FOURIER_RAPPORTS_BTC_1d.md`
- Bitstamp 2h / 1d: `docs/FOURIER_RAPPORTS_BTCUSD_2h.md`, `docs/FOURIER_RAPPORTS_BTCUSD_1d.md`

3) Synthèses & phases
- Daily (prix+volume): `outputs/fourier/DAILY_SUMMARY_ALL.csv`
- Analyse par phase (stats, plots, JSON scheduler): `outputs/fourier/phase/<SYM>_<TF>/`
- Labellisations (3 régimes, 5/6 phases + confusions): `docs/PHASE_LABELS/INDEX.md`
- Comparatif durées & Fourier (3/5/6): `docs/PHASE_LABELS/COMPARATIF_DUREES_FOURIER_HALVING.md`
- CSV comparatif (détails agrégés): `docs/PHASE_LABELS/csv/comparatif_durees_fourier_halving.csv`
- Graphiques prix colorés par phase: `docs/PHASE_LABELS/PHASE_PRICE_PLOTS.md`

4) Conclusions & émergence
- Conclusions datées: `docs/FOURIER_CONCLUSIONS/INDEX.md`
- Émergence cycles (top‑3 mensuel, daily): `docs/CYCLES_EMERGENCE/INDEX.md`

5) Elliott (rappel & usage quant)
- Index: `docs/ELLIOTT/INDEX.md` (Overview, Math, Usage)
- Comparaison Fourier ↔ Elliott: `docs/FOURIER_ELLlOTT_COMPARAISON.md`

6) Exports opérationnels
- JSON scheduler (ranges par phase + gating LFP): `outputs/fourier/phase/<SYM>_<TF>/SCHEDULER_FOURIER_*.json`
- Packages par exchange (CSV/XLSX): `outputs/export/<exchange>/`

7) Plan & suivi
- Plan 6 mois (suivi d’avancement): `docs/PLAN_6_MOIS/PLAN_6_MOIS.md`

8) Algorithme spectral HMM
- Pipeline de trading spectral: `docs/HMM_SPECTRAL_ALGO_FR.md`

À faire (prochaines priorités)
- Tables mensuelles top‑3 cycles dans `docs/CYCLES_EMERGENCE/` (lecture unique).
- Intégrer les JSON scheduler au runtime (lecture ranges + gating LFP).
- WFA rapide pour valider P→Ichimoku par phase; MC court de robustesse.

