# Algo Trading Haute Fréquence — Bot Binance Futures BTC/ETH/SOL

Ce dossier centralise la documentation et les rapports du projet bot trading crypto futures live testnet, dérivé du repo `HSBC-algo-genetic` (stratégie Ichimoku K3 phase rotation).

> **Note terminologique** : "haute fréquence" est utilisé ici au sens large (algo trading actif). La stratégie tourne sur timeframe **H2 (2 heures)** — elle n'est pas du vrai HFT (sub-seconde). Le terme "high frequence" est préservé sur demande utilisateur.

## Structure

| Fichier | Contenu |
|---|---|
| `RAPPORT_TRAVAUX_2026-04-27.md` | Synthèse de tout ce qui a été fait sur la session du 27 avril (multi-symbole, bleed, fixes precision) |
| `RECOMMANDATIONS_TOP5_2026-04-27.md` | Top 5 actions priorisées issues de l'audit Layer 3 (focus multi-symbole) |
| `RAPPORT_GAPS_LAYER3_2026-04-27.md` | Audit complet 16 sections nouvelles (vs Layer 2), citations papiers réels |
| `RAPPORT_BILAN_QUANT_2026-04-27.md` | Verdict quant final : aucune stratégie testée ne bat HODL spot BTC |
| `RAPPORT_GAPS_LAYER2_2026-04-26.md` | Audit "ce qu'on a oublié" — 10 sections (risk metrics, on-chain, microstructure, etc.) |
| `RAPPORT_ALPHA_FINAL_2026-04-26.md` | Verdict Ichimoku K3 nu : Sharpe -1.91, DSR 0.040, perte $66 mainnet cohérente |
| `RAPPORT_ALPHA_RECHERCHE_2026-04-26.md` | État de l'art recherche alpha crypto |
| `RAPPORT_COSTS_2026-04-26.md` | Impact frais Binance VIP0 sur la stratégie |
| `RAPPORT_AUDIT_LOOKAHEAD_2026-04-26.md` | Audit lookahead bias — clean |

## Code source du bot (hors de ce dossier)

Le code reste dans la structure originale du repo pour ne pas casser les imports / systemd / paths :

- `binance_bot/` : bot Binance Futures multi-symbole (BTC/ETH/SOL)
- `src/` : modules quant (cost_model, vol_targeting, range_detector, tail_risk, stats_eval, etc.)
- `tests/` + `binance_bot/tests/` : 206+ tests pytest
- `forge.py` : runner de validation
- `routines/` : scripts daily/intraday/watchdog/backup
- `dashboard/` : Flask UI port 8080
- `outputs/` : résultats backtests (gitignored sauf .md/.py/.png)
- `configs/phase_params_K3.json` : paramètres Optuna par phase
- `data/K3_1d_stable.csv` : labels K3 BTC (Fourier)

## État actuel — 2026-04-27

- Bot multi-symbole (BTC 125x, ETH 100x, SOL 50x) opérationnel sur testnet
- Wallet testnet : ~$103 (bleed depuis $4998 pour mimer conditions $100 mainnet)
- Capital théorique state.json : $100, sizing 1% margin par trade
- Règle 3 LONG XOR 3 SHORT par symbole, 9 positions max worst case
- 206/206 tests verts
- **Verdict** : aucun edge confirmé sur K3 (Sharpe backtest -1.91). Ne PAS scaler mainnet sans Sharpe OOS > 1.5

## ⚠ Baseline de reporting — 2026-04-27T12:00 UTC

**TOUS les trades exécutés sur le compte Binance testnet AVANT ce timestamp sont des opérations de SETUP** (wallet bleed $4998→$103 + tests round-trip de validation BTC/ETH/SOL LONG+SHORT). Ils ne font PAS partie de la stratégie K3.

**Référence** : `binance_bot/data/report_baseline.json` + entry `seq=17` dans `binance_bot/data/trades_audit.jsonl` (immutable, hash chain SHA256).

**Helper code** : `binance_bot/bot/report_filter.py`
```python
from bot.report_filter import load_baseline, filter_after_baseline
baseline = load_baseline()
trades_clean = filter_after_baseline(ex.fetch_my_trades('BTC/USDT'), baseline)
```

Tout futur P&L report, audit perf, ou tracking error live vs backtest doit utiliser ce filtre pour exclure les trades de setup.

## Prochaines étapes recommandées

Voir `RECOMMANDATIONS_TOP5_2026-04-27.md` pour le détail. Top 5 :
1. Module `portfolio_risk.py` — corrélation rolling + portfolio CVaR + leverage agrégé cap 4x
2. Module `psr_live.py` — Probabilistic Sharpe Ratio rolling vs backtest
3. Funding-aware close-before-settlement
4. Module `pre_trade_gate.py` — sanity checks fat-finger
5. Implementation Shortfall complet (Perold) dans paper_trader
