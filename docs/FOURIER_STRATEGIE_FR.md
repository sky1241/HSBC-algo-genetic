Fourier pour Ichimoku + ATR — Guide d’application
Date: 2025-08-21

Voir aussi :
- `FOURIER_SERIES_RAPPEL_FR.md` pour un rappel théorique.
- `FOURIER_SERIES_WELCH_HALVING_FR.md` pour l'analyse alignée sur le halving.


### 1) Pourquoi la transformée de Fourier ici ?
- Passer du temps aux fréquences pour comprendre « quelles » périodicités dominent et avec quelle puissance.
- Applications concrètes trading:
  - Détecter des cycles dominants (hebdo/mensuel/saisonniers)
  - Filtrer le bruit (passe‑bas/passe‑bande) avant les signaux Ichimoku
  - Accélérer des convolutions (moyennes, lissages ATR) via FFT en \( O(N \log N) \)
  - Créer des features ML (entropie spectrale, spectral flatness, Fourier features sin/cos)

### 2) Outils et définitions
- PSD (densité spectrale de puissance): mesure de l’énergie par fréquence; méthode de Welch recommandée.
- Période dominante: \( P = 1/f_* \) (en barres; convertir en jours pour intuition).
- Low‑Freq Power Ratio (LFP): \( LFP = \frac{\sum_{f < f_0} PSD(f)}{\sum_f PSD(f)} \), typiquement avec \( f_0 \) fixant >5 jours en H2.
- Entropie/flatness spectrale: niveau de « bruit » vs « tonalité » du spectre.

### 3) Recette plug‑and‑play
1) Fenêtre roulante: dernières 180–360 jours (~2160–4320 barres H2)
2) PSD (Welch), extraire \( f_* \) ⇒ \( P = 1/f_* \)
3) Calculer LFP pour \( f_0 \) (~ cycles > 5 jours H2)
4) Mapper vers Ichimoku:
   - `kijun ≈ P/2`, `tenkan ≈ P/8–P/6`, `senkou_b ≈ P`, `shift ≈ kijun/2`
   - Si `LFP > 0.6`: privilégier `kijun` long, `atr_mult` 3–5; sinon: `kijun` 26–55, `atr_mult` 2–3, filtre cloud strict

### 4) Détection de régime et scheduling
- Régime « lent/tendanciel »: LFP haut, flatness basse → Pool Trend (kijun/atr_mult plus élevés)
- Régime « bruyant »: flatness élevée → Pool Bruit (kijun/atr_mult plus serrés, règles strictes)
- Phase halving: aligne t=0 et calcule spectres moyens par phase; conditionne les plages et la cadence d’exploration.

### 5) Intégration pipeline
- Pré‑module « suggesteur » qui lit un CSV OHLCV, calcule (P, LFP, flatness) et produit un JSON baseline par symbole:
  - `{ symbol: { tenkan, kijun, senkou_b, shift, atr_mult } }`
- Le scheduler charge ce JSON comme baseline (option `--baseline-json`) et resserre/élargit les ranges en conséquence.

### 6) Limites & alternatives
- Non‑stationarité → préférer STFT/ondelettes si besoin de localisation temporelle.
- Si trous de données → Lomb–Scargle.
- Éviter le sur‑réglage: valider par walk‑forward et Monte Carlo; surveiller variance inter‑seeds.

### 7) Commande d’export PDF

### 8) Exemples chiffrés (BTC H2/D1 depuis 2020)
- Rolling monthly (médianes):
  - P1 ≈ 26.4 jours (H2) vs 26.16 jours (D1); P2 ≈ 15.0 vs 14.95; P3 ≈ 10.36 vs 10.05.
  - LFP: H2 − D1 ≈ −0.005 (faible écart).
- Rolling annual: ΔLFP ≈ −3.6×10⁻⁴.
- Interprétation: privilégier D1 pour la robustesse (peu de drift), H2 pour la finesse des réglages du scheduler.
```bash
python .\scripts\export_docs_to_pdf.py --docs .\docs\FOURIER_STRATEGIE_FR.md --out-dir .\outputs\reports
```


### 9) Règles de bascule de phase et indexation Ichimoku (opérationnel)

Cycles H2 typiques (P1): ~120 barres (~10j), ~180 barres (~15j), ~360 barres (~30j). On calibre Ichimoku sur P1 dominant et on « gate » l’intensité par LFP.

- 3 phases (up / down / range, H2)
  - up: LFP ≥ 0.83 ET momentum M ≥ +0.05 pendant ≥ 24–48 barres H2 ET P1 ≥ 120; confirmation si STFT/CWT_LFP_like ≥ 0.70. Sortie si LFP < 0.79 OU M < 0 pendant ≥ 24 barres (hystérésis).
  - down: LFP ≥ 0.83 ET M ≤ −0.05 pendant ≥ 24–48 barres H2 ET P1 ≥ 120. Sortie si LFP < 0.79 OU M > 0 pendant ≥ 24 barres.
  - range: LFP < 0.80 OU P1 ≤ 90 OU flatness élevée.

- 5 phases (accumulation, expansion, euphoria, distribution, bear)
  - accumulation: LFP remonte de <0.75 → >0.80, P1 s’allonge, M −0.02..+0.05, V modérée, DD se résorbe.
  - expansion: LFP ≥ 0.83, M +0.05..+0.12, V en hausse, P1 120–240.
  - euphoria: LFP ≥ 0.88, M ≥ +0.12, V élevée; P1 souvent 90–150.
  - distribution: LFP encore élevé mais M ↘ < 0 (ou < +0.02), V haute; P1 moyen/long.
  - bear: LFP ≥ 0.83 et M ≤ −0.05; si LFP < 0.80 → range baissier.

Déclencheurs de bascule
- Entrée: franchissement de seuils LFP/M maintenu N=24–48 barres + confirmation STFT/CWT.
- Sortie: franchissement inverse OU absence de confirmation N barres.
- Filtre D1 de contexte (optionnel): D1_LFP_mean > 0.80 favorise up/expansion; < 0.80 favorise range.

Indexation Ichimoku depuis P1–P3
- Tenkan ≈ round(P1/12), Kijun ≈ round(P1/6), Senkou shift ≈ round(P2/6) (bornes: 9/26/26, 12/34/26, 26/52/26).
- Range: set court (9/26), stops serrés; Trend fort (LFP haut, P1 long): set long (12/34 ou 26/52), ATR élargi.
- Gating intensité: LFP ≥ 0.88 pleine charge; 0.83–0.88 demi‑charge; <0.80 laisser passer.

Implémentation
- Lecture temps réel H2 → calcul LFP, M, P1… → « state machine » (3 ou 5 phases) avec hystérésis → mapping phase → (Ichimoku, ATR, taille, seed pool).


