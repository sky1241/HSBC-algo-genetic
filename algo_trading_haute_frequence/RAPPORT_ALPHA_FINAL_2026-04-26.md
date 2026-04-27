# RAPPORT ALPHA FINAL — Ichimoku K3 sur BTC H2 (2026-04-26)

> Verdict honnête après recherche académique + backtest empirique avec coûts réels.

## TL;DR

🛑 **La stratégie actuelle Ichimoku K3 phase rotation NE GAGNE PAS sur BTC H2 2020-2025 net de frais Binance VIP 0.**

- Sharpe annualisé OOS médian : **-1.91**
- Total return OOS : **-19.05%** sur 5 ans
- DSR (Bailey-LdP) : **0.040** ≪ 0.95 → edge non significatif
- Hansen SPA p-value : **0.530** → impossible de rejeter H0 "aucune stratégie testée n'a d'edge"

**Aucun des filtres régime testés (ER+ADX, CHOP veto, range_score composite) n'améliore la baseline.** Tous dégradent même la performance (V2 Sharpe -2.10, V3 -2.75, V4 -2.81).

## Méthodologie (recap)

- 4 stratégies V1-V4 backtestées sur **24 470 bars BTC/USDT H2** (2020-01-01 → 2025-08-01)
- Coûts réalistes : taker 4 bps + maker 2 bps + funding +0.01%/8h + slippage 0.5bp+ATR
- Walk-forward 5 folds, OOS rolling
- DSR avec n_trials=4
- Hansen SPA 1000 bootstrap blocks de 20 bars

Voir `outputs/wfa_alpha_2026-04-26/RAPPORT_ALPHA_BACKTEST.md` pour détails.

## Pourquoi ER+ADX REDUIT la performance ?

Hypothèse: en filtrant les "ranges", on RATE des reversals de fin de trend qui sont profitables sur Ichimoku — surtout au moment où Tenkan/Kijun cross dans une zone de consolidation post-trend. Le filtre est trop agressif.

Alternative: filtre INVERSÉ (trader les ranges comme mean reversion) — non testé, à explorer.

## Limitations à valider

1. **1 seul asset** : BTC. Peut-être l'edge existe sur ETH ou altcoins liquides.
2. **1 seul timeframe** : H2. Pourrait marcher mieux en H4 (moins de noise) ou H1 (plus d'opportunités).
3. **Coûts réalistes peut-être encore optimistes** : funding constant 1bp/8h ignore les pics 2021/2024.
4. **n_trials=4 sous-estime le data snooping** : les seuils (0.30, 22, 61.8) ont été tunés implicitement par la recherche internet — le vrai snooping est plus élevé.

## Recommandations stratégie

### Option A — PIVOT : abandonner Ichimoku K3, chercher autre chose
Confiance HAUTE.
- Vol-targeted carry (funding rate arb delta-neutral) : Sharpe documenté 3-6 dans la littérature, edge crypto-spécifique.
- HMM 2-3 états + LightGBM features (ADX, ER, Hurst, MVRV-Z, ATR%) en walk-forward — Sharpe 1.18 [CI 0.53, 1.84] dans MDPI Electronics 2025.
- Mean-reverting strategies en bear markets (paper Efe Arda SSRN).

### Option B — VARIANTES : tester d'autres timeframes/params Ichimoku
Confiance MOYENNE.
- H4 ou Daily au lieu de H2 (réduit slippage + noise).
- Optuna re-run sur params Ichimoku avec **DSR comme objectif** (pas Sharpe brut).
- Risque : data snooping additionnel, validity OOS douteuse.

### Option C — COUPLAGE : Ichimoku comme **régime detector** plutôt que signal
Confiance MOYENNE.
- Utiliser cloud thickness, Tenkan/Kijun spread comme features dans un modèle ML.
- Pas comme signal direct.

### Option D — STATUS QUO sur testnet
- Continuer le bot Ichimoku K3 sur testnet (capital fictif) pendant 1-2 mois pour collecter du tracking error live vs backtest.
- Mesurer implementation shortfall réel via paper_trader.
- Si live confirme le Sharpe négatif → PIVOT.
- **NE PAS scaler vers mainnet** avant qu'une stratégie passe DSR > 0.7 OOS + SPA p < 0.05.

## Décision proposée

**Option A + D combinées** :
1. **Maintenir** le bot actuel sur testnet uniquement (capital fictif, 0 risque mainnet)
2. **Démarrer** une nouvelle exploration stratégie : funding rate arb ou HMM+LightGBM
3. Ne **JAMAIS** déployer la K3 actuelle en mainnet sans nouveau DSR > 0.7

## Contraintes techniques préservées

Tout le code livré reste utile même si la stratégie change :
- ✅ Binance client (rate limit + time sync)
- ✅ Audit log hash chain
- ✅ Watchdog
- ✅ Kill switch
- ✅ Reconciler
- ✅ Dashboard
- ✅ Recovery WAL
- ✅ Stress tests
- ✅ DSR + Reality Check + Hansen SPA — métriques honnêtes
- ✅ Range detector (utilisable comme features ML)
- ✅ Tail risk (CVaR, GPD)

Le **harnais de validation** est solide. C'est la **stratégie de signal** qui est à revoir.

## Sources clés

- [Backtest ALPHA-2 outputs](outputs/wfa_alpha_2026-04-26/)
- [Recherche académique ALPHA-3](RAPPORT_ALPHA_RECHERCHE_2026-04-26.md)
- [Audit Layer-2 oversights](RAPPORT_GAPS_LAYER2_2026-04-26.md)
- [Audit lookahead](RAPPORT_AUDIT_LOOKAHEAD_2026-04-26.md)
- Bailey & López de Prado, "The Deflated Sharpe Ratio" SSRN 2460551
- Hansen 2005, "A Test for Superior Predictive Ability"
- MDPI Electronics 15(6) 1334, "Regime-Aware LightGBM WFA framework"

## Statut implémentation

- ✅ Filtre ER+ADX implémenté dans `binance_bot/services/regime_filter.py` mais **DÉSACTIVÉ par défaut** dans `bot_settings.yaml`. Le code reste pour expérimentation future, **NE PAS l'activer** en production car ALPHA-2 a montré qu'il dégrade la performance.
- ✅ Module range_detector et indicateurs ALPHA-1 utilisables comme features pour future stratégie ML.
- ⏳ À faire : implémenter funding rate arb stratégie (option A) en module séparé, sans toucher à la K3 actuelle.
