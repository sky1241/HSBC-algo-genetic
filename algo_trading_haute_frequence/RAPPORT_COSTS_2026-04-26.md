# Rapport B5 — Coûts réalistes (fees + funding + slippage)

Date : 2026-04-26
Auteur : Claude (B5 quant audit)
Scope : repo `HSBC-algo-genetic`, simulator strategy backtest. Bot live `binance_bot/` non-touché.

## 1. Cartographie

Le simulator de stratégie backtest est isolé dans `src/risk_sizing.py`
(`simulate_strategy`, `run_phase_strategy`). Toutes les routes WFA passent
par lui :

- `src/wfa.py:137,142` — `run_phase_strategy` + `simulate_strategy` (baseline).
- `src/optimizer.py:92` — `simulate_strategy` dans la boucle Optuna/random search.
- `optimizers/cscv_pbo.py:227,232` — appels directs pour la matrice CSCV.

Le pipeline historique `ichimoku_pipeline_web_v4_8_fixed.py` expose
`backtest_long_short` (utilisé par `optimizers/fitness.py`, `aco_optimizer.py`,
et plusieurs scripts `scripts/production/run_*.py`). Il a déjà sa propre
gestion des fees (paramètre `fees_bps`) — hors scope ici.

Avant patch : aucun de ces appels ne déduisait fees, funding ou slippage.

## 2. Modifications

- **Nouveau** `src/cost_model.py` (293 lignes) — implémente :
  - `BinanceFutureFees` (taker 4 bp, maker 2 bp, VIP 0).
  - `compute_trade_cost(notional, is_taker, fees) -> USDT négatif`.
  - `apply_funding(returns, position, funding_rates, default_rate_8h=1e-4)` —
    lag d'une barre pour éviter le look-ahead, agrégation 8h ↔ barres.
  - `compute_slippage(atr, qty_btc, price, depth_usdt=50_000)` — modèle
    spread + impact + vol (formule donnée dans le brief).
  - `apply_costs_to_returns(...)` — wrapper qui applique les trois drags
    en *return space*.
- **Modifié** `src/risk_sizing.py` (53 → 137 lignes) — nouvelle signature
  `simulate_strategy(df, params, *, realistic_costs=True, fees=None,
  funding_rates=None, leverage=1.0, depth_usdt=50_000, is_taker=True)`.
  Helper privé `_compute_atr` (ATR Wilder approximé). Idem
  `run_phase_strategy`. **Rétrocompatibilité** : flag `realistic_costs=False`
  restitue exactement l'ancien comportement (vérifié par test).

`src/funding_rate.py` (déjà présent) reste inchangé : il fournit le loader
Binance API, exploitable plus tard via `funding_rates=...`.

## 3. Tests

19 nouveaux tests (`tests/test_cost_model.py` + `tests/test_simulate_with_costs.py`),
**tous verts** :

```
.venv/bin/python -m pytest tests/test_cost_model.py tests/test_simulate_with_costs.py -v
=> 19 passed in 6.76s
```

Couverture :

- `test_cost_model.py` (14) — fees taker/maker, signe négatif, custom tier,
  notional zéro/négatif, funding zéro-position invariant, long pays / short
  reçoit, no-look-ahead, slippage baseline, croissance avec taille &
  volatilité, valeur de référence (notional 50k + ATR 2k → 205 USDT).
- `test_simulate_with_costs.py` (5) — net < gross, zéro-fees ⇒ baseline,
  Δ Sharpe ∈ [-3, 0], `run_phase_strategy` appelle bien le drag,
  default = realistic.

Tests existants : `tests/test_wfa_pipeline.py` toujours vert (3 passed).
`tests/` complet : 62 passed / 1 timeout pré-existant
(`test_constraint_satisfaction` dans test_aco_basic, dépend de
`backtest_long_short`, hors scope B5).

## 4. WFA partiel — impact mesuré

Script `scripts/analysis/measure_costs_impact.py`. Run BTC/USDT 1D
2020-01 → 2024-12, 3 seeds, 2 states HMM, 10 trials Optuna, 9 folds OOS.
Résultats dans `outputs/wfa_with_costs_2026-04-26/summary.json`.

| Stratégie     | Métrique | Gross   | Net     | Δ        |
|---------------|----------|---------|---------|----------|
| phaseaware    | Sharpe   | +0.934  | +0.494  | **-0.44** |
| baseline      | Sharpe   | +0.303  | -0.197  | **-0.50** |
| phaseaware    | CAGR     | +23.2 % | +9.6 %  | -13.6 pp |
| baseline      | CAGR     | +4.6 %  | -8.1 %  | -12.7 pp |
| phaseaware    | MDD      | -23.9 % | -26.2 % | -2.3 pp  |

Le Δ Sharpe (-0.4 à -0.5) tombe pile dans la fenêtre prédite par l'audit
(-0.3 à -0.7). La baseline naïve perd son edge entièrement (Sharpe < 0
après coûts), confirmant que le signal Ichimoku K3 1D non-phasé est
dominé par les frais. Le phaseaware résiste mieux mais voit son
attractivité divisée par ~2.

## 5. Hypothèses majeures

- **Fee tier** = VIP 0 (4 bp taker / 2 bp maker). À ajuster si le compte
  HSBC bénéficie d'un rabais BNB ou volume.
- **Funding** constant à +1e-4 par 8h (= +0.03 %/jour) en l'absence de
  série historique. La donnée est dispo via `src/funding_rate.py`
  (`load_or_fetch_funding_rate`) — non-câblée par défaut pour rester
  testable offline.
- **Slippage depth** = 50 kUSDT à 1 bp, BTC top-1bp réel souvent plus
  large → modèle conservateur.
- **Order type** = taker (market). En mode maker post-only le drag est
  divisé par 2.
- **Leverage** = 1× ; passe à n× le drag scale linéairement
  (`apply_costs_to_returns(... leverage=n)`).

## 6. TODO restants

1. Câbler `src/funding_rate.py::load_or_fetch_funding_rate` dans
   `wfa.run_walk_forward` pour utiliser le funding réel (fallback constant).
2. Mesurer le depth réel via `binance_bot/...` orderbook snapshots et
   remplacer `depth_usdt=50_000` par une série `depth_t`.
3. Porter les coûts dans `ichimoku_pipeline_web_v4_8_fixed.py:backtest_long_short`
   (il a déjà `fees_bps` mais ni funding ni slippage dynamique).
4. Re-runner les WFA seeds 30 (`scripts/production/launch_30_seeds_k5.ps1`)
   après le câblage funding réel pour réviser les baselines `baselines.json`.
5. Ajouter un test sur `test_aco_basic.py::test_constraint_satisfaction`
   pour le timeout pré-existant (non-bloquant pour B5).

---

Files modifiés (chemins absolus) :
- `/home/ludov/HSBC-algo-genetic/src/cost_model.py` (créé, 293 L)
- `/home/ludov/HSBC-algo-genetic/src/risk_sizing.py` (réécrit, 137 L)
- `/home/ludov/HSBC-algo-genetic/tests/test_cost_model.py` (créé, 14 tests)
- `/home/ludov/HSBC-algo-genetic/tests/test_simulate_with_costs.py` (créé, 5 tests)
- `/home/ludov/HSBC-algo-genetic/scripts/analysis/measure_costs_impact.py` (créé)
- `/home/ludov/HSBC-algo-genetic/outputs/wfa_with_costs_2026-04-26/summary.json` (résultat)
