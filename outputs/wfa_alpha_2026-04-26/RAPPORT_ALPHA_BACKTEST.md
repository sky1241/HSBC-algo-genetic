# RAPPORT ALPHA-2 — Backtest comparatif Ichimoku K3 + filtres régime

Auteur : agent ALPHA-2 (2026-04-26).  
Asset : **BTC/USDT perpetuel**, granularité **H2**, période **2020-01-01 → 2025-08-01** (24470 bars).  
Coûts nets : Binance VIP 0 (taker 4 bps, maker 2 bps), funding +0.01 %/8 h (proxy historique 2020-2025), slippage spread 0.5 bp + composante ATR.

## 1. Hypothèses testées

- **V1 — Baseline Ichimoku K3** : Tenkan=21, Kijun=35, Senkou B=90, shift=+44. Long si `tenkan>kijun` ET `close>max(senkA,senkB)` ; symétrique au short.
- **V2 — V1 + filtre ER+ADX** : entrée seulement si `efficiency_ratio(close,30)>0.30` ET `adx(14)>22` (Kaufman/Wilder).
- **V3 — V2 + veto CHOP** : en plus, `choppiness_index(14)<=61.8` (Dreiss).
- **V4 — V1 + range_score<0.4** : score composite `src.range_detector.range_score` (ADX, CI, ER, BBW squeeze, Hurst rolling).

## 2. Méthodologie

- **Walk-forward** : 5 folds, train/test 70 / 30, expanding train, OOS rolling.
- Métriques calculées **par fold** puis médiane multi-fold.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) avec `n_trials=4` (les 4 stratégies comparées).
- **Hansen SPA** (consistent, lower, upper) : 1000 répliques bootstrap stationnaire, block size moyenne 20 bars. Test sur returns OOS chaînés des 5 folds.
- Coûts via `src.cost_model.apply_costs_to_returns` (frais + funding + slippage).
- `periods_per_year = 365 * 12 = 4380` pour annualiser le Sharpe sur barres H2.

## 3. Tableau métriques (médiane OOS multi-fold)

| Stratégie | Sharpe | DSR | Sortino | MDD | Ulcer | n_trades | Win rate | Total OOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | -1.906 | +0.040 | -1.393 | -26.94% | 0.1627 | 98 | 47.05% | -19.05% |
| V2 | -2.104 | +0.035 | -1.084 | -23.91% | 0.1492 | 86 | 46.41% | -18.69% |
| V3 | -2.754 | +0.015 | -1.292 | -26.92% | 0.1755 | 88 | 44.04% | -20.10% |
| V4 | -2.810 | +0.014 | -1.465 | -31.01% | 0.2019 | 96 | 44.33% | -18.20% |

Lecture : Sharpe et Sortino sont annualisés. DSR proche de 1.0 = edge significatif après data snooping ; proche de 0.5 = edge incertain.

## 4. Hansen SPA test

- Test statistic V = -0.0013  (observations OOS chaînées : 7340)
- **p-value SPA_consistent** = 0.530  (seuil rejet H0 < 0.05 ⇒ edge réel)
- p-value SPA_lower (conservateur) = 0.738  
- p-value SPA_upper (libéral) = 0.530  
- **Stratégie best (max f̄_k)** : V2 (index 0)

## 5. Recommandation finale

### NE PAS DÉPLOYER

Raison : Sharpe uplift -0.198 <= 0; DSR(V2)=0.035 ≤ DSR(V1)=0.040; Hansen SPA p-consistent=0.530 > 0.05.

Aucune stratégie filtrée n'a passé simultanément les trois gates : Sharpe > V1, DSR > V1, p-value Hansen SPA < 0.05. Cf hypothèse H_0 du rapport recherche : Ichimoku seul + filtre régime n'apporte pas d'edge net statistiquement détectable sur cet asset/timeframe.

## 6. Limitations (honnêteté impérative)

- **1 seul asset** (BTC/USDT). Aucune généralisation à ETH/altcoins.
- **1 seul timeframe** (H2). H1/H4 peuvent renverser les conclusions.
- **Frais & funding constants** : la dynamique de funding crowded est ignorée. Live, les funding extrêmes peuvent éroder 1-3 bps additionnels par bar long.
- **n_trials=4** est sous-estimé si on considère qu'on a aussi joué sur les seuils (0.30, 22, 61.8, 0.4). Le DSR rapporté est donc optimiste sur l'effet snooping.
- **Pas de stop-loss explicite** dans la simulation : la position est toujours long/short selon le signal, sans gestion fine de drawdown intra-trade. Live, l'interaction stop-loss + filtre régime peut différer.
- **Slippage proxy** : 0.5 bp + composante ATR ; un live BTC stressé (news, liquidations) peut taper 5-10 bps.
- **5 folds** seulement : intervalle de confiance large sur la médiane.

## 7. Reproductibilité

Commande exacte :

```bash
.venv/bin/python -m scripts.analysis.run_alpha_backtest \
    --data data/BTC_USDT_2h.csv \
    --funding data/funding_rate_BTCUSDT.csv \
    --start 2020-01-01 --end 2025-08-01 --folds 5
```

Tests : `pytest tests/test_alpha_backtest.py -v`
