# BILAN QUANT FINAL — 2026-04-27 (nuit du 26 au 27)

> Synthèse honnête après 3 stratégies testées rigoureusement sur données réelles BTC H2 2020-2025.

## TL;DR — pour l'utilisateur

🛑 **Aucune stratégie testée ne bat HODL spot BTC sur 2020-2025 net de frais.**

| Stratégie | Sharpe OOS | Total return | Verdict |
|---|---:|---:|---|
| **HODL spot** ⭐ | **1.09** | **+1470%** | **Champion** |
| HODL perp 1x | 0.88 | +639% | Second (drag funding) |
| DCA hebdo | 0.74 | +348% | Sympa, faible MDD |
| HODL perp 2x | 0.88 | +442% | Risqué (MDD -98%) |
| HODL perp 3x | 0.88 | -63% | Pratiquement liquidé |
| Ichimoku K3 V1 (ton bot actuel) | -1.91 | -19% | ❌ Edge nul |
| Filtre ER+ADX | -2.10 | -18% | ❌ Pire que baseline |
| STRAT-A funding arb | -0.32 | ~-1% | ❌ Edge éliminé par frais |
| STRAT-B HMM+LightGBM | **-6.58** | **-95%** | ❌ Catastrophe |

## Ce qui a été testé (méthodo)

1. **Ichimoku K3** (ton setup actuel) — Sharpe -1.91. Cohérent avec ta perte de $66 mainnet : la stratégie n'avait jamais d'edge.
2. **Filtres régime** (ER+ADX, CHOP, range_score composite) — TOUS dégradent V1.
3. **Funding rate arb delta-neutral** — frais 16 bps round-trip > funding mean 1.2 bps/8h. Mathématiquement impossible.
4. **HMM 2-état + LightGBM directional** — AUC ROC = 0.53 (random). Le ML ne prédit rien d'exploitable sur ces features.

Tous les backtests :
- Walk-forward strict avec purge anti-leakage (López de Prado)
- Coûts réalistes Binance VIP 0 (4 bps taker, slippage ATR-aware, funding empirique)
- Deflated Sharpe Ratio (Bailey-LdP) pour data snooping
- Hansen SPA test (1000 bootstraps blocks)
- Aucun lookahead bias (audit grep clean — `RAPPORT_AUDIT_LOOKAHEAD_2026-04-26.md`)

## Pourquoi tout échoue

### 1. BTC n'est pas une stratégie, c'est l'asset
Le drift naturel de BTC (CAGR 64%/an sur 2020-2025) absorbe toute la prime de risque. Toute stratégie qui sort du marché manque le drift. Toute stratégie qui shorte ramasse le opposite drift.

### 2. Frais ronds-trips éclatent les marges
- Ichimoku K3 : 98 trades/fold × 8 bps round-trip = 7.8% de drag annuel net.
- Funding arb : 16 bps × N flips = 30%+ de drag/an si on flip souvent.
- HMM+LGBM : sur signaux noisy AUC 0.53, on flip souvent et on saigne.

### 3. Le levier amplifie les drawdowns BTC
- BTC -50% intraday s'est produit 4× depuis 2020 (LUNA, FTX, COVID, mai 2021).
- Levier 3x → liquidé sur drawdown -33%. Cf stress tests.

### 4. La littérature crypto est SURVIVORSHIP-BIASED
- Papers publiés = ceux qui ont marché. La VRAIE distribution est centrée sur Sharpe ~ 0.
- IC95% du paper MDPI 2025 (HMM+LGBM) : [0.53, 1.84] — borne basse à peine > 0.5. Réplication = chance.

## Recommandation honnête

### Court terme (semaines)
1. **Garde le bot K3 sur testnet uniquement.** Capital fictif, 0 risque. Mesure le tracking error live vs backtest pendant 1-2 mois.
2. **Si tu veux exposition BTC** : DCA hebdo ou HODL spot. Pas de bot algo.
3. **Apprends le code et l'infra** : tout ce qu'on a construit (audit log, watchdog, recovery WAL, etc.) reste utile peu importe la stratégie.

### Moyen terme (mois)
1. **Ne PAS scaler vers mainnet** avec aucune des stratégies testées. Aucune n'a Sharpe > 1.09 (HODL).
2. Si tu veux explorer plus : altcoins illiquides où le funding peut être 50+ bps/8h (mais slippage énorme et bid-ask spread peut bouffer l'edge).
3. Lis les papers cités dans `RAPPORT_ALPHA_RECHERCHE_2026-04-26.md` mais reste sceptique.

### Long terme (années)
1. **L'edge retail crypto vient de l'information** (early adoption, info on-chain, narrative trading). Pas du TA.
2. Ou de **niches HFT** inaccessibles retail.
3. Ou d'**arbitrage cross-exchange** quand spreads anormaux (DEX vs CEX, perp basis cross-venue) — demande infra plus avancée.

## Tu as quand même appris des trucs précieux

Ce n'est pas une perte — c'est une calibration honnête :

✅ **Code propre** : ~330 tests, audit log immutable, watchdog, recovery WAL, kill switch, dashboard. Réutilisable même sur autre asset/stratégie.

✅ **Méthodo quant** : DSR, Hansen SPA, WFA purgé, cost_model. Tu peux maintenant évaluer N'IMPORTE QUELLE stratégie crypto avec rigueur.

✅ **Limites perçues** : tu sais maintenant pourquoi ton $66 mainnet est parti. Pas un bug, pas du malchance — juste pas d'edge.

✅ **Infra production-grade** : systemd, linger, backup B2 prêt, Telegram alerting. Si tu trouves un edge un jour, tu sauras le déployer.

## Décision proposée — REPRENDS LA MAIN

Tu as 3 chemins :

**A) Accept the verdict, switch to HODL spot.** Vendre le projet "bot trading" et garder le code comme portfolio. Recommandation rationnelle.

**B) Continuer à explorer** sur testnet, données autres assets (ETH, alts liquides), MVRV/NVT features, Bollinger mean-rev en bear. Pas mainnet avant Sharpe OOS > 1.5.

**C) Pivot complet** : utilise le code base pour autre chose (paper trading apps, backtest service, blog quant, etc.).

C'est ta décision. Le bot continue à tourner sur testnet en autonomie en attendant. À toi.

---

*Rapports détaillés :*
- `outputs/baselines_2026-04-26/RAPPORT_BASELINES.md` — HODL benchmark complet
- `outputs/strat_funding_arb_2026-04-26/RAPPORT.md` — STRAT-A
- `outputs/strat_regime_lgbm_2026-04-26/RAPPORT.md` — STRAT-B
- `outputs/wfa_alpha_2026-04-26/RAPPORT_ALPHA_BACKTEST.md` — ALPHA-2 K3 + filtres
- `RAPPORT_ALPHA_FINAL_2026-04-26.md` — verdict K3 (à mettre à jour avec STRAT-B)
- `RAPPORT_GAPS_LAYER2_2026-04-26.md` — 12 oversights infra/risk/legal
- `RAPPORT_AUDIT_LOOKAHEAD_2026-04-26.md` — clean
