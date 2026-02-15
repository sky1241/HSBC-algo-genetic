## Conclusions Fourier — 2025-08-27

- LFP (Fourier) vs STFT/CWT: après calibration (scale2frequency, bande LF [64, 4096] barres), les mois à forte LFP sont cohérents avec des ratios LF élevés en STFT et CWT sur H2. Sur D1, CWT est moins stable par manque de barres mensuelles.
- Périodes dominantes: les médianes P1 (H2) restent centrées sur des multiples (120/180/360 barres), robustes entre Binance/Bitstamp. STFT/CWT confirment l’énergie concentrée sur des horizons comparables.
- Phases et durées: les parts de phases (regime3/phase5/phase6) montrent des alternances nettes autour des halvings. Les heatmaps mensuels rendent visibles des blocs de tendance où LFP et parts "up" augmentent conjointement.
- Implication Ichimoku: mapping P→(Tenkan,Kijun) reste pertinent (ex.: P≈120→(9,26); P≈360→(26,52)), avec gating par LFP pour l’intensité et l’allocation.
- Limites: D1 mensuel insuffisant pour une CWT fine; nécessité d’agrégations glissantes plus longues (ex.: 60–90 jours) et d’une validation IS/OOS.
- Prochaines étapes: intégrer ces métriques au scheduler, lancer walk-forward et Monte Carlo, puis publier les comparaisons consolidées.
