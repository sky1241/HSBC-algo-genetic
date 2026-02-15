# Deep Research Analysis - De la Survie vers l'Alpha

**Date:** 2026-02-07
**Statut:** Validé - Plan d'action en plusieurs phases
**Durée estimée:** 6-8 semaines

---

## 1. Diagnostic Principal

### Le Problème Identifié

```
Système actuel: Régime = "dans quel état est le marché"
Ce qu'il manque: Direction = "où va le prix"
```

| Composant | Ce qu'il fait | Ce qu'il ne fait PAS |
|-----------|---------------|----------------------|
| Fourier Welch | Détecte périodes dominantes | Pas de direction |
| HMM K-régimes | Classifie volatilité/microstructure | Pas de prédiction |
| Labels 1D stable | Réduit whipsaw 90% | Toujours réactif |
| Ichimoku adaptatif | Trend-following par phase | Signal retardé |

### Résultat Observé

- **Survie:** 100% (les régimes fonctionnent)
- **Alpha:** +3.6%/an vs +85%/an B&H (gap énorme)
- **Cause:** Régime correct + signal lag = entrée/sortie trop tard

### Conclusion Académique Validée

> "La détection de régimes améliore le contrôle du risque mais ne crée pas automatiquement de l'alpha, sauf si tu traduis ces régimes en espérance conditionnelle de rendement."

---

## 2. Les 5 Recommandations (Priorité Décroissante)

### Reco 1: NHHM - Non-Homogeneous HMM (Priorité 5/5)

**Source Vérifiée:**
- Koki, C., Leonardos, S., & Piliouras, G. (2022)
- "Exploring the predictability of cryptocurrencies via Bayesian hidden Markov models"
- Research in International Business and Finance, Volume 59
- [arXiv](https://arxiv.org/abs/2011.03741) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0275531921001756)

**Concept:**
```
HMM actuel:  Features → Label dur (0,1,2) → Params Ichimoku
NHHM:        Features + Covariables → P(bull|state) → Signal directionnel
```

**Résultats rapportés:**
- 4 états (bull/bear/calm) pour BTC/ETH/XRP
- Hit ratio directionnel ~55% OOS pour BTC
- Prédicteurs: momentum, VIX, rendement Treasuries US

**Implémentation:**
1. Redéfinir émission HMM pour inclure rendements (pas juste features Fourier)
2. Transitions time-varying: P(St=j|St-1=i, Xt) avec Xt = covariables exogènes
3. Signal = E[r_t+1|F_t] = Σ P(S_t=k|F_t) × μ_k
4. Outils: PyMC / NumPyro (Bayésien) ou statsmodels Markov-switching

**Complexité:** Élevée (1-2 mois)

---

### Reco 2: Carry/Funding Perpetuals (Priorité 5/5)

**Sources Vérifiées:**
- He, S. & Manela, A. (2022-2024) "Fundamentals of Perpetual Futures" [arXiv:2212.06888](https://arxiv.org/abs/2212.06888)
- BIS Working Paper No. 1087 (2023) "Crypto Carry" [PDF](https://www.bis.org/publ/work1087.pdf)

**Concept:**
Le funding rate encode QUI paie le levier et le sentiment directionnel:
- Funding > 0 et hausse → Longs paient → Sentiment bullish
- Funding < 0 et baisse → Shorts paient → Sentiment bearish

**Résultats rapportés:**
- Carry crypto moyen > 10%/an (vs ~2-3% traditionnel)
- Pics de carry prédisent liquidations futures (BIS)
- Sharpe in-sample très élevés pour carry hedgé (ATTENTION: probablement surestimé)

**Implémentation:**
```python
# Par bougie H2, calculer:
funding_rate_level      # Annualisé ou par 8h
funding_rate_change     # Δ et Δ² (accélération)
funding_zscore          # vs rolling 30 jours

# Polarité momentum
if phase == 1:  # Momentum détecté
    if funding_rate > 0.01 and funding_rising:
        polarite = "BULL_MOMENTUM"
    elif funding_rate < -0.01:
        polarite = "BEAR_MOMENTUM"
    else:
        polarite = "NEUTRAL_MOMENTUM"
```

**Complexité:** Faible (1-2 semaines) - données gratuites Binance API

---

### Reco 3: Net Inflows USDT (Priorité 4/5)

**Source Vérifiée:**
- arXiv:2411.06327 (2024-2025)
- "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets"
- [arXiv](https://arxiv.org/abs/2411.06327)

**Concept:**
USDT entrant sur exchanges = pouvoir d'achat déployable → signal lead

**Résultats rapportés:**
- Net inflows USDT vers exchanges prédisent positivement rendements BTC
- Horizons 1-6h = compatible H2
- Inflows BTC n'ont PAS de pouvoir prédictif pour rendements BTC

**Implémentation:**
```python
# Features H2
usdt_inflow_level       # Valeur brute
usdt_inflow_zscore      # vs baseline 30j
usdt_inflow_spike       # Flag si top quantile
```

**Problème:** Accès données
- Glassnode/CryptoQuant = payants (~$50-200/mois)
- APIs gratuites = délais ou qualité douteuse

**Complexité:** Moyenne (dépend du provider data)

---

### Reco 4: Meta-Labeling (Priorité 4/5)

**Sources:**
- Concept de Marcos Lopez de Prado (Advances in Financial ML)
- "A Confidence-Threshold Framework for Cryptocurrency" (2025, MDPI)

**Concept:**
Garder Ichimoku comme générateur de signaux, mais filtrer via ML:
```
Signal Ichimoku → Meta-model(features) → Trade OUI/NON + Sizing
```

**Implémentation:**
1. Event = signal entrée/sortie Ichimoku
2. Target = signe rendement net (après frais) sur horizon 12-24 bars H2
3. Features au moment de décision:
   - Probabilités de régimes (pas label dur)
   - Polarité funding (Reco 2)
   - Vol state, force trend, distance au nuage
   - Inflows USDT si dispo (Reco 3)
4. Classifier: LightGBM / Logistic Regression
5. Exécuter seulement si confiance > seuil optimisé

**Avantages:**
- Réutilise tout le système existant
- Filtre les "mauvais" trades Ichimoku
- Améliore Sharpe sans refonte

**Complexité:** Moyenne (2-4 semaines)

---

### Reco 5: Filtre Multi-Timeframe F&G + Stress (Priorité 3/5)

**Source Vérifiée:**
- "Predicting Cryptocurrency Returns for Real-World Investments" (2023)
- [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S154461232300778X)

**Concept:**
Filtre lent D1 pour éviter de trader "le mauvais sens" du momentum:
- Fear & Greed haut + monte → Risk-on, trend peut appuyer
- Fear & Greed bas + crash → Risk-off, défensif

**Résultats rapportés:**
- Pouvoir prédictif OOS significatif horizons 1 jour à 1 semaine
- Relation NON-linéaire (U-shaped)

**Implémentation:**
```python
# API gratuite: https://api.alternative.me/fng/
fear_greed_level    # 0-100
fear_greed_regime   # "risk_on" si > 50 et rising, sinon "risk_off"

# Combiné avec stress dérivés
if carry_zscore > 2:  # Stress levier extrême
    reduce_exposure = True
```

**Complexité:** Faible (quelques jours)

---

## 3. Sources Académiques Vérifiées

| Paper | Année | Journal/Source | Lien |
|-------|-------|----------------|------|
| Bayesian HMM Crypto | 2022 | RIBAF | [arXiv](https://arxiv.org/abs/2011.03741) |
| Perpetual Futures | 2022-24 | arXiv | [2212.06888](https://arxiv.org/abs/2212.06888) |
| BIS Crypto Carry | 2023 | BIS WP 1087 | [PDF](https://www.bis.org/publ/work1087.pdf) |
| On-Chain Flows | 2024-25 | arXiv | [2411.06327](https://arxiv.org/abs/2411.06327) |
| Fear & Greed Predict | 2023 | ScienceDirect | [Lien](https://www.sciencedirect.com/science/article/abs/pii/S154461232300778X) |

---

## 4. Plan d'Action par Phases

### PHASE 1 - Quick Wins (Semaines 1-2)

```
Objectif: Ajouter polarité directionnelle sans refonte
Effort: ~10-15h de dev

Tasks:
□ 1.1 Récupérer funding rate historique Binance (API)
□ 1.2 Calculer features: level, change, zscore
□ 1.3 Désambiguïser Phase 1: bull vs bear momentum
□ 1.4 Intégrer Fear & Greed API (filtre D1)
□ 1.5 Backtest rapide sur 3 seeds (997-999)
□ 1.6 Comparer métriques vs baseline K3/K5
```

**Livrables:**
- `scripts/fetch_funding_rate.py`
- `scripts/compute_polarity_features.py`
- Feature columns ajoutées au pipeline

---

### PHASE 2 - Meta-Labeling (Semaines 3-5)

```
Objectif: Filtrer signaux Ichimoku via confiance ML
Effort: ~20-30h de dev

Tasks:
□ 2.1 Générer dataset (signal Ichimoku, features, target)
□ 2.2 Split temporel strict (pas de leakage!)
□ 2.3 Entraîner LightGBM classifier
□ 2.4 Optimiser seuil confiance via WFA
□ 2.5 Intégrer au pipeline d'exécution
□ 2.6 Backtest 30 seeds, comparer Sharpe/MDD
```

**Livrables:**
- `models/meta_labeler.pkl`
- `scripts/train_meta_labeler.py`
- Métriques: Sharpe cible > 0.5 (vs ~0.3 actuel)

---

### PHASE 3 - NHHM Upgrade (Semaines 6-8)

```
Objectif: HMM prédictif avec transitions time-varying
Effort: ~40-50h de dev

Tasks:
□ 3.1 Étudier papier Koki et al. en détail
□ 3.2 Implémenter NHHM via PyMC ou statsmodels
□ 3.3 Définir covariables transitions (funding, momentum)
□ 3.4 Remplacer labels durs par probas directionnelles
□ 3.5 Intégrer E[return|state] comme signal
□ 3.6 Validation WFA complète (30 seeds, 300 trials)
```

**Livrables:**
- `models/nhhm_predictor.py`
- Nouveau pipeline: Fourier → NHHM → Signal directionnel → Exécution

---

## 5. Notes Importantes

### Sharpe Réalistes
Les Sharpe 7-10 cités dans le carry trade sont **irréalistes** pour application réelle:
- Probablement in-sample
- Sans frais/slippage
- Avec levier implicite

**Objectif réaliste:** Sharpe 0.8-1.2 avec MDD < 20%

### Risques à Surveiller
1. **Overfitting meta-labeler:** Utiliser splits temporels stricts
2. **Data leakage:** Fear & Greed est D1, attention au forward-fill
3. **Non-stationnarité:** Structure USDT/funding évolue
4. **Coûts cachés:** Toujours inclure 0.04%/trade + slippage

### Validation Multi-Seeds
Appliquer la même rigueur que pour K3/K5:
- 30 seeds minimum
- 300 trials par seed
- Métriques: CAGR, Sharpe, MDD, Survie

---

## 6. Références Rapides

### APIs Utiles
```python
# Binance Funding Rate
GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000

# Fear & Greed Index
GET https://api.alternative.me/fng/?limit=365

# Open Interest
GET /fapi/v1/openInterest?symbol=BTCUSDT
```

### Commandes de Suivi
```powershell
# Progression K5 en cours
Get-ChildItem 'outputs\wfa_phase_k5\seed_*\PROGRESS.json' | % {
    $j = Get-Content $_.FullName | ConvertFrom-Json
    "$($_.Directory.Name): $($j.percent)%"
}
```

---

## 7. Historique des Décisions

| Date | Décision | Raison |
|------|----------|--------|
| 2026-02-07 | Valider deep research | Sources vérifiées, cohérent avec système |
| 2026-02-07 | Prioriser funding (Reco 2) | Quick win, données gratuites |
| 2026-02-07 | Reporter NHHM à Phase 3 | Complexité élevée, nécessite Phase 1-2 d'abord |

---

*Document généré suite à l'analyse de la Deep Research V2*
*Prochaine revue: après completion K5 30 seeds*
