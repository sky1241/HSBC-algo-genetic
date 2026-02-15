# MEGA PROMPT - Deep Research Amélioration Algo Trading BTC

---

## PARTIE 1: CONTEXTE TECHNIQUE PRÉCIS

### Mon Système Actuel (résumé)

```
Pipeline: Données BTC H2 (14 ans) → Fourier Welch → HMM K-régimes → Labels stabilisés 1D → Ichimoku adaptatif par phase → WFA Optuna
```

### Données Réelles

| Métrique | Valeur |
|----------|--------|
| Période | 2011-2025 (14 ans) |
| Timeframe | 2 heures (H2) |
| Bougies | 60,531 |
| Asset | BTC/USDT futures |

### Résultats Réels (30 seeds, 300 trials chacun)

| Config | Monthly | Annual | MDD | Sharpe | Trades/an | Survie |
|--------|---------|--------|-----|--------|-----------|--------|
| K3 1D stable | +0.30% | +3.6% | 12% | ~0.3 | 32 | 100% |
| K5 1D stable | ~+0.25% | ~+3% | 11% | ~0.25 | 28 | 100% |
| Buy & Hold BTC | +5.2% | +85% | 83% | 0.9 | 0 | - |
| Fixed Ichimoku | -100% | RUINE | 100% | - | - | 0% |

**Gap critique:** Mon algo fait +3.6%/an vs Buy&Hold +85%/an. Je capture la SURVIE mais pas l'ALPHA.

### Paramètres Optimisés par Phase (K3)

```
Phase 0 (Stable/Accumulation):
  tenkan=27, kijun=102, senkou_b=180, shift=93, ATR=11.8

Phase 1 (Momentum/Volatilité haute):
  tenkan=29, kijun=58, senkou_b=232, shift=96, ATR=19.5

Phase 2 (Transition/Réactif):
  tenkan=24, kijun=40, senkou_b=99, shift=45, ATR=11.8
```

### Ce Qui Fonctionne
- HMM capture 3 régimes distincts (CV inter-phases > 60%)
- Adaptation paramètres → 100% survie (vs 0% sans)
- Stabilisation 1D → réduit whipsaw de 90%

### Ce Qui NE Fonctionne PAS
- Signal Fourier = fréquentiel, pas directionnel
- HMM = classification, pas prédiction
- Ichimoku = trend-following retardé
- Peu de trades (32/an) = peu d'opportunités

---

## PARTIE 2: DIAGNOSTIC DU PROBLÈME

### Hypothèse Principale

Mon système détecte "dans quel RÉGIME on est" mais pas "où le prix VA ALLER".

```
Régime actuel = Phase 1 (momentum)
→ J'adapte mes paramètres pour Phase 1
→ Mais je ne sais pas si c'est BULL ou BEAR momentum!
```

### Décomposition du Problème

1. **Signal Fourier:** Donne périodes dominantes (256 bars ~= 21 jours), mais pas de direction
2. **HMM:** Classifie en régimes basé sur features Fourier, mais transitions sont RÉACTIVES pas PRÉDICTIVES
3. **Ichimoku:** Trend-following classique, génère signaux APRÈS que le trend commence
4. **Combinaison:** Régime correct + signal retardé = entrée trop tard, sortie trop tard

### Questions Critiques à Résoudre

1. Comment transformer la PROBABILITÉ DE TRANSITION HMM en signal directionnel?
2. Quelles features ajoutent de l'information PRÉDICTIVE (pas juste descriptive)?
3. Comment anticiper les CHANGEMENTS de régime avant qu'ils arrivent?
4. Existe-t-il des indicateurs LEAD (avancés) vs LAG (retardés) pour crypto?

---

## PARTIE 3: PISTES À EXPLORER (RECHERCHE PROFONDE)

### A. Amélioration HMM/Régimes

Cherche spécifiquement:

1. **Regime Switching Models avec prédiction:**
   - Hamilton (1989) regime switching - comment utiliser les probas de transition?
   - MS-VAR (Markov Switching VAR) pour prédire returns
   - Papers: "regime switching trading strategy" + "prediction"

2. **Features pour HMM:**
   - Actuellement j'utilise: P1-P6 periods, LFP, volatility Fourier
   - Quelles features PRÉDICTIVES ajouter?
   - Order flow? Funding rates? On-chain?

3. **Transition comme signal:**
   - Quand P(transition Phase0→Phase1) > 70%, faut-il agir?
   - Papers sur "regime transition trading signal"

### B. Remplacement/Amélioration Ichimoku

Cherche:

1. **Indicateurs LEAD (avancés):**
   - RSI divergences
   - Volume divergences
   - Order flow imbalance
   - Funding rate extremes

2. **Indicateurs adaptatifs modernes:**
   - KAMA (Kaufman Adaptive MA)
   - FRAMA (Fractal Adaptive MA)
   - Papers "adaptive indicators cryptocurrency"

3. **Entrées/Sorties ML:**
   - Classification entrée/sortie avec XGBoost/LightGBM
   - Features = Ichimoku + momentum + volume + régime
   - Target = rendement forward 12-24 bars

### C. Features Prédictives Crypto-Spécifiques

Cherche impact réel (pas théorique) de:

1. **On-chain:**
   - MVRV Z-score (top/bottom detector)
   - SOPR (Spent Output Profit Ratio)
   - Exchange inflow/outflow
   - Whale transactions
   - Active addresses

2. **Derivatives:**
   - Funding rate (sentiment futures)
   - Open interest changes
   - Liquidations (cascade trigger)
   - Options put/call ratio
   - Basis (spot vs futures)

3. **Sentiment:**
   - Fear & Greed Index
   - Social volume (LunarCrush?)
   - Google Trends
   - Twitter/Reddit sentiment

**Question clé:** Lesquelles ont du POUVOIR PRÉDICTIF prouvé, pas juste corrélation contemporaine?

### D. Machine Learning Avancé

Cherche implémentations concrètes:

1. **Temporal Fusion Transformer (TFT):**
   - Google paper 2020
   - Multi-horizon forecasting
   - Attention sur features importantes
   - GitHub implémentations PyTorch

2. **Reinforcement Learning:**
   - PPO/A2C pour trading
   - Reward shaping (Sharpe vs PnL?)
   - Papers "deep reinforcement learning cryptocurrency"
   - Problème: sample efficiency, overfitting

3. **Meta-Learning:**
   - MAML pour adaptation rapide
   - Few-shot learning sur nouveaux régimes
   - Papers "meta-learning trading"

4. **Ensemble Methods:**
   - Combiner plusieurs modèles
   - Réduire variance des prédictions
   - Stacking: Fourier + ML + Sentiment

### E. Optimisation du Signal

Cherche:

1. **Multi-Timeframe:**
   - Signal H2 confirmé par D1?
   - Hierarchical momentum
   - Papers "multi-timeframe trading strategy"

2. **Confirmation croisée:**
   - Régime HMM + Momentum + Volume = signal fort?
   - Comment pondérer?

3. **Sizing dynamique:**
   - Position sizing selon confiance régime
   - Kelly criterion adaptatif
   - Volatility targeting (papier AQR)

---

## PARTIE 4: QUESTIONS DIRECTES

Réponds précisément à ces questions:

1. **Papiers académiques 2022-2025** les plus cités sur "cryptocurrency trading algorithm" - lesquels montrent des résultats OOS robustes?

2. **GitHub repos** avec backtests vérifiables qui battent buy&hold sur BTC (Sharpe > 1, MDD < 30%) - liens?

3. **Features on-chain** avec le meilleur pouvoir prédictif documenté - études quantitatives?

4. **Pourquoi** la détection de régime ne génère pas d'alpha? C'est un problème connu dans la littérature?

5. **Solutions pratiques** utilisées par les hedge funds crypto (si info publique) - interviews, podcasts, blogs?

---

## PARTIE 5: FORMAT DE RÉPONSE

Pour chaque recommandation:

```
### [Nom de la technique]

**Source:** [lien ou référence]
**Concept:** [2-3 phrases max]
**Evidence:** [résultats rapportés dans la source]
**Applicabilité:** [Haute/Moyenne/Faible] + pourquoi
**Implémentation:**
- Étape 1: ...
- Étape 2: ...
- Code si dispo: [lien GitHub]
**Risques:** [overfitting, data leakage, etc.]
**Priorité:** [1-5] selon ratio effort/gain
```

---

## PARTIE 6: CE QUE JE NE VEUX PAS

- Conseils génériques ("diversifiez", "gérez le risque")
- Théorie sans application pratique
- Techniques qui nécessitent HFT/co-location
- Solutions qui marchent "en théorie" mais pas en backtest
- Indicateurs techniques basiques (j'ai déjà tout essayé)

---

## CONTRAINTES TECHNIQUES

- Python (pandas, numpy, sklearn, optuna, pytorch)
- Données: OHLCV H2 + labels HMM déjà calculés
- Compute: PC standard (pas de GPU cluster)
- Capital: $1,000-10,000
- Pas de trading intraday < 2h
- Frais: 0.04% par trade

---

## MÉTA-INSTRUCTION

Agis comme un chercheur quantitatif senior qui doit résoudre CE problème spécifique. Pas de réponse superficielle. Cherche dans:
- arXiv (quant-ph, q-fin)
- SSRN
- Journal of Financial Economics
- Quantitative Finance journal
- Medium/Towards Data Science (articles techniques)
- GitHub (repos avec stars > 500)
- QuantConnect/Quantopian forums archives
- r/algotrading posts techniques
- Podcasts quant (Flirting with Models, Top Traders Unplugged)

Prends ton temps. Je préfère 5 recommandations solides que 20 superficielles.

---

FIN DU PROMPT V2
