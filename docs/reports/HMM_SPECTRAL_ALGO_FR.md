# Algorithme de trading spectral (HMM + Fourier)

## Synthèse MA vs Ichimoku (2h)
- **Corrélation** `corr(M_MA, M_ICH)` ≈ 0.43–0.45 (relation moyenne, non équivalente).
- **Médianes** : `M_MA` ≈ 0.002–0.003 ; `M_ICH` ≈ 0.005–0.010 (biais plus haussier côté Ichimoku).
- **Écart moyen absolu** : `|ΔM|` ≈ 0.047.
- **Labels divergents** : ≈ 7–8 % des barres, surtout à la frontière accumulation ↔ distribution.

## Schéma des comportements attendus
```
               ┌────────────────────┐
               │   ACCUMULATION     │
               └────────────────────┘
          - Prix : range serré, volumes modestes
          - M (MA)  ≈ 0.002–0.003      ⇒ biais neutre/léger bullish
          - M (Ichimoku) ≈ 0.005–0.010 ⇒ biais plus bullish
          - Divergence MA vs Ichimoku : Ichimoku peut déjà signaler
            une légère « distribution »
          - Objectif : construction de positions avant tendance haussière

                     │
                     ▼

               ┌────────────────────┐
               │      MARKUP        │
               └────────────────────┘
          - Prix : cassure haussière, volumes en hausse
          - M (MA/ICH) ↑ fortement   ⇒ tendance confirmée
          - Ichimoku réagit plus tôt : passages Kumo, Tenkan>Kijun
          - Objectif : montée progressive, gains cumulés

                     │
                     ▼

               ┌────────────────────┐
               │   DISTRIBUTION     │
               └────────────────────┘
          - Prix : plateau, volatilité plus forte
          - M (MA) ≈ 0 ou en baisse, M (ICH) légèrement positif
          - MA signale « accumulation » ≈ 7–8 % du temps ;
            Ichimoku bascule plus vite vers « distribution »
          - Objectif : prise de profits, positions de sortie

                     │
                     ▼

               ┌────────────────────┐
               │     MARKDOWN       │
               └────────────────────┘
          - Prix : rupture baissière, volumes importants
          - M (MA/ICH) négatif ; Ichimoku confirme en dessous du nuage
          - Objectif : déclin, souvent jusqu’à nouvelle zone d’accumulation

                     │
                     └── Retour éventuel à l’accumulation
```

## Pipeline spectral → HMM
1. **Ne pas fixer K a priori**
   - Estimer le modèle pour K ∈ {3,4,5,6}.
   - Choisir K via critères d’information (AIC/BIC) et performance prédictive (log‑loss, Sharpe out‑of‑sample).
2. **Analyse temps‑fréquence**
   - Pré‑traitement : retours log, standardisation, anti‑alias.
   - Estimateurs locaux : STFT, multitaper (DPSS) ou ondelettes (Morlet/MODWT).
   - Réduction de variance : Welch.
3. **Features extraites (fenêtre roulante ≈ 256 barres H2)**
   - Band‑power par bandes log‑spacées, spectral centroid, entropy, slope (≈1/f^α), peak frequency, ratios low/high.
   - Différences temporelles (Δ band‑power, Δ centroid) ; cohérence retour‑volume/volatilité si disponible.
   - Variables prix/momentum : ATR, volatilité réalisée, skew/kurtosis, ADX, composantes Ichimoku (distance Kumo, Tenkan/Kijun, chikou).
4. **Modélisation HMM / Markov‑Switching**
   - Covariances diagonales en première approche, entraînement via EM.
   - États interprétés via diagnostics (vol, dérive, spectre) → phases (accumulation, haussier, distribution, baissier, éventuelle euphorie).
5. **Sélection et validation de K**
   - Walk‑forward (ré‑estimation périodique) pour évaluer la robustesse.
   - Garder K plus élevé (ex. 5) seulement si BIC/log‑loss s’améliorent nettement et que les états supplémentaires ont un sens économique (ex. scission haussière « expansion » vs « euphorie »).
6. **Stratégie par état**
   - Règles de trading spécifiques (taille, levier, stops, filtrage news) selon l’état détecté.
   - Ajustement des paramètres Ichimoku/ATR via mapping Fourier (ex. `kijun ≈ P/2`).

## Recommandation pragmatique
- Démarrer avec **K = 4** (Accumulation, Haussier, Distribution, Baissier).
- Tester **K = 5** : conserver seulement si le modèle sépare clairement l’état haussier en deux régimes et améliore BIC/log‑loss OOS.
- Intégrer la sélection automatique de K dans le pipeline pour chaque ré‑échantillonnage (walk‑forward).

