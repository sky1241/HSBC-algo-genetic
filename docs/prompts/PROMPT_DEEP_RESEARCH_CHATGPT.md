# DEEP RESEARCH PROMPT - Amélioration Algorithme Trading

Copie tout ce qui suit dans ChatGPT (avec Deep Research activé):

---

## CONTEXTE - ALGORITHME DE TRADING CRYPTO (BTC)

Je développe un algorithme de trading systématique sur Bitcoin depuis plusieurs mois. J'ai besoin d'une recherche EXHAUSTIVE pour l'améliorer. Cherche dans TOUTES les sources: papiers académiques, blogs quantitatifs, forums de trading algo, GitHub, arXiv, SSRN, etc.

### Architecture Actuelle

```
DONNÉES (14 ans BTC H2)
    ↓
ANALYSE FOURIER (Welch PSD)
    → Extraction périodes dominantes P1-P6
    → Low-Frequency Power (LFP)
    ↓
HMM (Hidden Markov Model)
    → K=3/5/8 régimes de marché
    → Labels de phase par bougie
    ↓
STABILISATION 1D
    → Label majoritaire jour J → appliqué jour J+1
    → Réduit switches de 12/jour à 0.15/jour
    ↓
ICHIMOKU ADAPTATIF
    → Paramètres différents par phase (tenkan, kijun, senkou_b, shift)
    → ATR multiplier pour stop-loss
    → Take-profit multiplier
    ↓
WFA (Walk-Forward Analysis)
    → 14 folds temporels
    → Optuna (300 trials par fold)
    → Validation out-of-sample
    ↓
BACKTEST
    → Frais: 0.04% par trade
    → Slippage: modélisé
    → Futures rollover costs
```

### Résultats Actuels

| Configuration | Rendement Mensuel | Max Drawdown | Survie |
|---------------|-------------------|--------------|--------|
| K3 1D stable (30 seeds) | +0.30% | 12% | 100% |
| K5 1D stable (test) | variable | 9-17% | 100% |
| K5 H2 pur | -100% (ruine) | 100% | 0% |
| Fixed params (sans phases) | -100% (ruine) | 100% | 0% |

**Objectif:** 5%/mois (actuellement à 6% de l'objectif)

### Forces Identifiées
- Robustesse exceptionnelle (100% survie sur 30 seeds)
- Contrôle du risque (MDD ~12%)
- Différenciation forte entre phases (CV > 60%)
- Pas d'overfitting (validation WFA)

### Faiblesses Identifiées
- Rendement faible (0.30%/mois vs 5% objectif)
- Signal Fourier = détection fréquentielle ≠ prédiction directionnelle
- ~32 trades/an seulement
- Pas de prédiction de prix, juste adaptation de régime

---

## RECHERCHE DEMANDÉE

### 1. AMÉLIORATION DU SIGNAL FOURIER/HMM

Cherche des techniques pour:
- Améliorer la prédictivité des régimes HMM (pas juste détection)
- Combiner Fourier avec d'autres décompositions (Wavelets, EMD, VMD?)
- Utiliser les transitions de phase comme signal directionnel
- Papers sur "regime switching trading strategies"
- Papers sur "spectral analysis cryptocurrency"

### 2. ALTERNATIVES À ICHIMOKU

Cherche:
- Indicateurs plus adaptés aux cryptos haute fréquence
- Systèmes de trend-following modernes (2020-2025)
- Papers sur "adaptive technical indicators"
- Combinaisons d'indicateurs optimales pour crypto
- Machine learning pour sélection dynamique d'indicateurs

### 3. OPTIMISATION DE LA FONCTION OBJECTIF

Actuellement j'optimise equity finale. Cherche:
- Meilleures loss functions pour trading (Calmar, Sortino, etc.)
- Multi-objective optimization pour trading
- Papers sur "portfolio optimization loss function"
- Techniques anti-overfitting pour Optuna

### 4. FEATURES ADDITIONNELLES

Cherche l'impact de:
- Données on-chain (MVRV, SOPR, NVT, exchange flows)
- Sentiment analysis (Fear & Greed, social media)
- Corrélations cross-market (DXY, SPX, Gold)
- Volume profile / Order flow
- Funding rates / Open interest
- Whale tracking

### 5. AMÉLIORATION TEMPORELLE

Cherche:
- Multi-timeframe analysis optimale
- Techniques de synchronisation des signaux
- Papers sur "optimal trading frequency"
- Adaptive position sizing selon régime

### 6. MACHINE LEARNING AVANCÉ

Cherche:
- Transformers pour time series (temporal fusion transformer)
- Reinforcement learning pour trading
- Meta-learning pour adaptation rapide
- Ensemble methods pour réduire variance
- Papers récents (2023-2025) sur "deep learning cryptocurrency trading"

### 7. RISK MANAGEMENT AVANCÉ

Cherche:
- Dynamic position sizing (Kelly, fractional Kelly)
- Volatility targeting
- Regime-aware risk management
- Tail risk hedging
- Papers sur "cryptocurrency risk management"

### 8. EXÉCUTION ET MARKET MICROSTRUCTURE

Cherche:
- Optimal execution algorithms pour crypto
- Impact du slippage sur stratégies trend-following
- Papers sur "market microstructure cryptocurrency"

---

## FORMAT DE RÉPONSE ATTENDU

Pour CHAQUE piste d'amélioration trouvée, donne:

1. **Source** (paper, blog, GitHub, etc.)
2. **Concept clé** (résumé en 2-3 phrases)
3. **Applicabilité** à mon algo (haute/moyenne/faible)
4. **Implémentation** (pseudo-code ou étapes clés)
5. **Gain potentiel estimé** (si mentionné dans la source)

---

## CONTRAINTES À RESPECTER

- BTC uniquement (pas de diversification multi-asset pour l'instant)
- Timeframe H2 (2 heures)
- Données depuis 2011
- Infrastructure Python (pandas, numpy, optuna, sklearn)
- Pas de HFT (pas d'accès co-location)
- Capital modeste (~$1000-10000)

---

## QUESTIONS SPÉCIFIQUES

1. Pourquoi mon signal Fourier/HMM améliore la SURVIE mais pas le RENDEMENT?
2. Comment transformer une détection de régime en PRÉDICTION directionnelle?
3. Quelles features additionnelles ont le meilleur ratio signal/bruit pour BTC?
4. Y a-t-il des stratégies crypto open-source (GitHub) qui surperforment 0.30%/mois de façon robuste?
5. Quels papers académiques récents (2023-2025) sont les plus prometteurs pour le trading crypto systématique?

---

## IMPORTANT

- Cherche PROFONDÉMENT, pas juste des réponses génériques
- Cite tes sources avec liens si possible
- Priorise les techniques PROUVÉES sur données réelles (pas juste théoriques)
- Si une technique est prometteuse mais risquée, dis-le clairement
- Compare toujours au baseline (buy & hold BTC)

---

FIN DU PROMPT
