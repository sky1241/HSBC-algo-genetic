# Analyse ChatGPT Deep Research - Février 2025

## DIAGNOSTIC PRINCIPAL

Le système est **robuste mais ne génère pas d'alpha** car :

1. **Un seul archétype** : On reparamétrise du trend-following, on ne change pas le comportement économique
2. **Friction dominance** : Fees + slippage + funding mangent le petit edge
3. **Fourier = DESCRIPTIF, pas PRÉDICTIF** : Le PSD décrit la structure, il ne prédit pas les trades gagnants
4. **HMM Gaussien inadapté** : Crypto = heavy tails, le Gaussien traite les gros moves comme des outliers

---

## SOLUTIONS PAR IMPACT (du plus au moins impactant)

### 1. VOLATILITY TARGETING + DRAWDOWN THROTTLE ⭐⭐⭐ HIGH
```
Leverage = min(L_max, σ_target / σ_realized)
+ Mode panique si drawdown > seuil
```
- Garde le signal Ichimoku
- Scale l'exposition selon la volatilité
- **Si on a un petit edge mais sous-exposé → ça scale les returns**

### 2. MIXTURE-OF-EXPERTS ⭐⭐⭐ HIGH
Au lieu d'optimiser Ichimoku par phase, **changer de stratégie par phase** :
- **Phase trend** → Ichimoku (actuel)
- **Phase chop** → Mean-reversion (RSI, Bollinger, z-score)
- **Phase breakout** → Donchian, range expansion
- **Phase carry** → Basis/funding arbitrage

Utiliser soft-switching (probabilités HMM) pas hard-switching.

### 3. DERIVATIVES ALPHA (CARRY/BASIS) ⭐⭐ MEDIUM-HIGH
- Le **funding rate** et le **basis** sont des sources d'alpha orthogonales au trend
- BIS documente que le carry crypto peut être large et persistant
- Traiter comme un sleeve séparé, pas du leverage sur le trend

### 4. CROSS-CRYPTO PREDICTORS ⭐⭐ MEDIUM
- ETH, majors comme **leading indicators** pour BTC
- Meta-labeling : `size ∝ P(success)` basé sur cross-crypto features
- Horizons multiples (10m, 1h, 6h, 24h)

### 5. EXIT STRATEGY (TP/TRAILING) ⭐⭐ MEDIUM
- Fixed TP **tronque les winners** → tue le trend-following
- Test : trailing-only, time-based, partial TP + trailing
- Attribution : quel % de PnL vient du top décile de trades ?

### 6. HMM HEAVY-TAIL (t-distribution) ⭐ LOW-MEDIUM
- Remplacer Gaussian → Student-t emissions
- Améliore robustesse aux outliers et persistance des états
- Wavelets ou EMD au lieu de Fourier pour time-locality

---

## RED FLAGS IDENTIFIÉS

1. **Hard-switching = hidden turnover** : Changer les params à chaque transition de régime crée du slippage caché

2. **Sharpe-maximizing → inactivité** : Dans les phases sans edge, Optuna apprend à ne pas trader → survie mais pas de rendement

3. **5%/mois avec 12% MDD = irréaliste** sans levier significatif ou multiples alphas orthogonaux

4. **300 trials Optuna = overfitting probable** : Utiliser Deflated Sharpe Ratio et Reality Check

---

## BENCHMARKS

| Métrique | Notre système | Industrie |
|----------|---------------|-----------|
| Sharpe annualisé | ~0.3-0.5 ? | Médian ~1.1, top ~2.0 |
| Rendement mensuel | 0.30% | Variable, multi-strat |
| MDD | 12.2% | Dépend du levier |

**Conclusion** : Notre Sharpe est probablement sous 1.0, ce qui indique un problème d'**edge**, pas juste de risk budget.

---

## PLAN D'ACTION RECOMMANDÉ

### Étape 1 : PnL Decomposition (RAPIDE, HAUTE VALEUR)
Par année et par phase HMM :
- Nombre de trades
- Durée moyenne
- Gross return avant frais
- Coûts (fees, slippage, funding)
- Hit rate, payoff ratio
- % PnL du top décile

→ Savoir si le problème est edge ou exposure.

### Étape 2 : Volatility Targeting
- Implémenter vol-targeting + drawdown throttle
- Rerun sur subset de seeds
- Vérifier si returns scalent linéairement

### Étape 3 : Mixture-of-Experts
- Garder le régime detector
- Ajouter un expert mean-reversion pour phases non-trend
- Soft-switching via probabilités filtrées

### Étape 4 : Derivatives Sleeve
- Carry/funding comme stratégie séparée
- Market-neutral yield, pas leverage

### Étape 5 : Heavy-tail HMM
- Remplacer Gaussian → t-distribution
- Réduire regime churn

### Étape 6 : Validation Multiple Testing
- Deflated Sharpe Ratio
- Reality Check (White)
- Probabilistic Sharpe

---

## PAPERS ET RESSOURCES CLÉS

### Backtest Validity
- [Reality Check for Data Snooping](https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf)
- [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

### Sharpe Inference
- [Statistics of Sharpe Ratios](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios)
- [Probabilistic Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)

### Crypto Alpha
- [Crypto Momentum](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2872158_code2618281.pdf?abstractid=2872158)
- [BTC Technical Analysis](https://www.mdpi.com/2076-3417/9/7/1345)
- [Cross-Crypto Predictability](https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551)

### Derivatives
- [BIS Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [Crypto Carry CEPR](https://cepr.org/voxeu/columns/crypto-carry-market-segmentation-and-price-distortions-digital-asset-markets)

### Time-Frequency
- [Wavelets Finance](https://www.econstor.eu/bitstream/10419/257998/1/risks-08-00044.pdf)
- [SSA Review](https://www.researchgate.net/profile/Hossein-Hassani-12/publication/266685235_A_review_on_Singular_Spectrum_Analysis_for_economic_and_financial_time_series)

### HMM
- [Hamilton Regime Switching](https://www.ssc.wisc.edu/~bhansen/718/Hamilton1989.pdf)
- [t-HMM](https://jmlr.org/papers/volume26/23-0343/23-0343.pdf)

---

## CONCLUSION

**Le problème principal n'est pas la robustesse (on l'a), c'est l'alpha.**

Pour booster le rendement sans sacrifier la robustesse :
1. **Volatility targeting** pour scaler l'exposition
2. **Mixture-of-experts** pour monétiser tous les régimes
3. **Carry/funding** comme alpha orthogonal
4. **Heavy-tail HMM** pour stabiliser les régimes
