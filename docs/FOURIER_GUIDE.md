### Guide d’usage — Analyse de Fourier (style thèse)

Objectif: produire et exploiter P1–P3 (périodes dominantes) et LFP (ratio basse fréquence) sur BTC, en H2 et D1, avec pipeline reproductible et interprétation scientifique.

1) Commandes principales (Windows PowerShell)
- Générer signaux + graphes (optimisé, H2):
  - `py -3 scripts/fourier_signals_batch.py --csv data/BTC_USDT_2h.csv --symbol BTC/USDT --timeframe 2h --step-bars 12`
- Rapports mensuels (H2 ou D1):
  - `py -3 scripts/fourier_monthly_report.py --csv data/BTC_USDT_2h.csv --symbol BTC/USDT --timeframe 2h`
- Index mensuel consolidé (page unique, 2h ou 1d):
  - `py -3 scripts/build_fourier_index.py --symbol BTC/USDT --timeframe 2h --out docs/FOURIER_RAPPORTS_BTC_2h.md`
- Synthèse quotidienne (prix+volume):
  - `py -3 scripts/build_daily_summary.py`
- Analyse par phase (halving) + JSON scheduler:
  - `py -3 scripts/fourier_phase_analysis.py --symbol BTC_USDT --timeframe 2h`
- Labellisations (3 régimes, 5/6 phases) + matrices de confusion:
  - `py -3 scripts/build_phase_labels.py --symbol BTC_USDT --timeframe 2h`

2) Interprétation (scientifique)
- P1–P3: périodes dominantes (en barres) issues de la PSD/Welch; P1 pilote l’ordre de grandeur d’Ichimoku: `kijun ≈ P1/2`, `tenkan ≈ P1/8–P1/6`, `senkou_b ≈ P1`.
- LFP: part de puissance sous `f0` (cycles >~5 jours en H2); `LFP > 0.6` suggère régime tendanciel (gating haute intensité), `LFP < 0.75` rapproche d’un range.
- Volume: P1_vol–P3_vol et LFP_vol confirment/infirment la cohérence prix↔flux; divergences notables en euphorie/distribution.

3) Bonnes pratiques
- D1 pour robustesse structurelle (écarts faibles vs H2), H2 pour réactivité du scheduler.
- Fenêtres roulantes (annual/monthly) pour limiter la non‑stationnarité; considérer STFT/ondelettes si régimes très locaux.
- QC systématique: `scripts/quality_check_csv.py` + `docs/QUALITY_REPORTS.md`.

4) Optimisation & performance
- Paramètre `--step-bars` (≈ 1 jour en H2) pour accélérer calculs; les timelines sont forward‑fill pour conserver la granularité d’affichage.
- Welch par défaut (fallback périodogramme si `scipy` absent) avec `nperseg` auto.

5) Liens utiles
- Rapports consolidés: `docs/FOURIER_RAPPORTS_BTC_2h.md`, `docs/FOURIER_RAPPORTS_BTC_1d.md` (et versions Bitstamp).
- Phases & régimes: `docs/PHASE_LABELS/INDEX.md`.
- Conclusions datées: `docs/FOURIER_CONCLUSIONS/INDEX.md`.

6) Primitives factorisées (module central)

- `scripts/fourier_core.py` fournit des briques communes pour toute la pipeline:
  - `compute_welch_psd(close, fs) -> (freqs, psd)`
  - `dominant_period(freqs, psd) -> float`
  - `low_freq_power_ratio(freqs, psd, f0) -> float`
  - `spectral_flatness(psd) -> float` (GM/AM sur composantes positives; renvoie 0 si au moins un zéro est présent ou si toutes sont nulles)
  - `fir_lowpass_subsample(df, q, fs, cutoff) -> DataFrame`
  - `anti_aliased_daily(df_2h) -> DataFrame` (H2→D1 avec filtre anti‑alias, cutoff≈0.4 cycles/jour)

Les scripts alignés sur ce module: `scripts/fourier_utils.py`, `scripts/build_daily_summary.py`, `scripts/fourier_phase_analysis.py`, `scripts/build_phase_labels.py`.

