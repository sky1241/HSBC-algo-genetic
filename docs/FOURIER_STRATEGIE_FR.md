Fourier pour Ichimoku + ATR — Guide d’application
Date: 2025-08-21

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
```bash
python .\scripts\export_docs_to_pdf.py --docs .\docs\FOURIER_STRATEGIE_FR.md --out-dir .\outputs\reports
```


