### Conclusions Fourier — 2025-08-26

Synthèse (2020→aujourd’hui):
- P1 (monthly) ≈ 26.4 j (H2) vs 26.16 j (D1); P2 ≈ 15.0 vs 14.95; P3 ≈ 10.36 vs 10.05.
- LFP (monthly) ≈ 0.87 ± 0.05; rolling annual LFP ≈ 0.993.
- Écarts H2–D1 faibles (ΔLFP_monthly ≈ −0.005; ΔLFP_annual ≈ −3.6e−4).

Implications:
- Baseline robuste sur D1; H2 pour le scheduler (réactivité) sans divergence.
- Cartographie vers Ichimoku: kijun ≈ P1/2; tenkan ≈ P1/8–P1/6; senkou_b ≈ P1; shift ≈ kijun/2.
- LFP élevé ⇒ allonger kijun et atr_mult; LFP modéré ⇒ resserrer + filtres cloud stricts.

Prochaines actions:
- Stratifie par phases de halving (Accu/Expan/Euph/Distrib/Bear) et mesure Δ(P1–P3, LFP) par phase.
- Ajouter STFT/ondelettes si rupture de régime localisée.

Sources:
- `outputs/FOURIER_ALL_REPORTS.csv`, `outputs/fourier/plots/*`, pages HTML/PDF dans `outputs/reports/`.

---

### Analyse détaillée (version verbatim)

- Données et qualité
  - Binance BTC/USDT (2h, 1d) et Bitstamp BTC/USD (1d, 2h resamplé) validées: pas d’anomalie bloquante (gaps/NaN/OHLC/volumes/outliers).
  - Les rapports et CSV agrégés confirment une base exploitable.
- Cycles dominants (rolling monthly, 2020→aujourd’hui)
  - P1 ≈ 26.4 j (H2) vs 26.16 j (D1), P2 ≈ 15.0 j vs 14.95 j, P3 ≈ 10.36 j vs 10.05 j.
  - LFP moyen ≈ 0.87 (écart-type ≈ 0.05) → forte énergie basse fréquence (régime plutôt tendanciel, mais variable selon mois).
- Cycles dominants (rolling annual)
  - P1 annual ≈ 325–330 j (prix), LFP annual ≈ 0.993 (très dominé par basses fréquences).
  - Confirme une structure 1/f^α (α>0) sur longues fenêtres.
- Volume vs prix
  - P1_vol–P3_vol et LFP_vol suivent de près les métriques prix (écarts faibles), suggérant une cohérence cycles/flux.
  - Écarts prix/volume susceptibles d’augmenter en phases « euphories/distributions » (à exploiter pour gating).
- H2 vs D1: écart faible
  - Différences P1–P3 en jours ≪ 1 jour; ΔLFP monthly ≈ −0.005; annual ≈ −3.6e−4.
  - Conclusion: D1 suffit pour ancrer des réglages robustes; H2 utile pour le scheduler (réactivité) sans divergence structurelle.
- Implications Ichimoku (heuristique P → paramètres)
  - Mapper P1 mensuel vers: kijun ≈ P1/2, tenkan ≈ P1/8–P1/6, senkou_b ≈ P1, shift ≈ kijun/2.
  - Ordres de grandeur (H2): P1≈26 j ⇒ kijun≈13 j (≈156 barres 2h), tenkan≈3–4.5 j (≈36–54 barres 2h), senkou_b≈26 j (≈312 barres 2h).
  - Remarque: les valeurs « classiques » (tenkan 9, kijun 26, senkouB 52) s’alignent partiellement; notre cartographie justifie leur « logique » et permet un ajustement phase-aware.
- Lecture de régime (via LFP)
  - LFP_monthly > 0.6 → régime tendanciel; nos moyennes ~0.87 indiquent une propension au trend, mais l’écart-type ~0.05 motive un gating dynamique (ex.: >0.9 haute intensité pools; 0.8–0.9 médium; <0.8 réduit).
- Conclusions opérationnelles (préliminaires)
  - Baseline stable sur D1; affinement H2 pour la cadence seeds/pools.
  - Intégrer P1–P3/LFP (prix+volume) au scheduler HSBC: allonger kijun/atr_mult en mois à LFP élevé; resserrer en LFP modéré.
  - Prochaine étape recommandée: stratifie par phases de halving (Accu/Expan/Euph/Distrib/Bear) pour quantifier les deltas P1–P3/LFP par phase et sceller les ranges.
- Limites
  - Dominance de basses fréquences sur longues fenêtres ⇒ vigilance sur non‑stationnarité; conserver fenêtres roulantes + contrôle IS/OOS.
  - Périodes proches des bornes de fenêtre (biais de Welch) à surveiller; compléter avec STFT/ondelettes si nécessaire.

### Mise à jour — 2025-09-24

- Croisement Fourier/HMM (BTC Fused 2h, WFA annuel, MDD≤50%)
  - Classement par Calmar/Lyapunov (préliminaire): K3 > K5 >> K8 ~ 0 > K2 < 0.
  - Phase-aware: en régimes tendanciels, réglages Ichimoku « longs » (kijun/senkouB élevés, shift élargi) dominent; en régimes neutres, réglages plus « courts ».
  - Référence: K3 seed=123 (run antérieur) Eqx×2.63, MDD≈8.8%, Calmar≈0.82.

- Robustesse et agrégation
  - Médianes/IQR par phase (K3/K5) différencient bien les paramètres; IQR encore élevé à 1–2 seeds ⇒ viser ≥10–15 seeds avant gel.
  - Exclure les segments « nan » du mapping.

- Implications opérationnelles
  - Prioriser K3 puis K5 (phase) pour finaliser 30 seeds; sélection par Calmar (CAGR/MDD) sous MDD≤50%, Sharpe et Lyap en appui.
  - Calculer rendements mensuels OOS par phase (médiane/IQR) et comparer H2 vs D1 (anti‑alias) pour retenir le timeframe gagnant.

