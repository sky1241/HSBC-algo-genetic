# Rapport de travaux — Session 2026-04-27 (matin)

> Synthèse exhaustive de la session du 27 avril 2026, après reprise du projet bot Binance Futures.
> Contexte précédent : sessions 2026-04-26 (audit + 6 bug fixes Layer 1, infra production-grade, audit Layer 2, bilan quant).

---

## TL;DR

1. **Refactor multi-symbole** : bot étendu de BTC seul → BTC + ETH + SOL en parallèle (config + state + runner + signal_engine + trade_manager)
2. **Conditions live testnet** alignées sur les conditions mainnet $100 (capital, leverage max par symbole, marge isolée)
3. **Wallet testnet bled de $4998 → $103** via 130 round-trips successifs pour mimer le contexte $100 réel
4. **2 bugs critiques fixés** : `qty round(3)` hardcodé pour BTC + `triggerPrice` envoyé brut → erreur `-1111 Precision is over the maximum` sur ETH/SOL
5. **Audit professionnel Layer 3** (16 sections nouvelles) sur ce qui manque encore vs standards quant institutionnels
6. **206/206 tests passent** post-refactor

---

## 1. Reprise de session

- Lecture des historiques JSONL de conversation pour récupérer le contexte (la session précédente avait crash mid-task)
- Lecture de la mémoire auto + verdict quant final (`RAPPORT_BILAN_QUANT_2026-04-27.md`)
- Validation : aucune stratégie testée ne bat HODL spot BTC (Sharpe 1.09). K3 Sharpe -1.91. Le bot continue sur testnet en autonomie

## 2. Préparation conditions $100 démo

### 2.1 Config bot ajustée

| Paramètre | Avant | Après |
|---|---|---|
| `initial_capital_usdt` (state.json) | 1000.0 | **100.0** |
| `position_size_pct` | 0.01 | 0.01 (1% margin) |
| `max_leverage` (BTC) | 3 | **125** (max testnet) |

Math par trade BTC : marge $1, notional $125 (≥ min Binance $100 ✓), liquidation à ~0.8% adverse.

### 2.2 Vérification leverage testnet par symbole

| Symbole | Tentative | Réponse Binance | Final |
|---|---:|---|---:|
| BTC/USDT | 150x | "leverage should be between 1 and 125" | **125x** |
| ETH/USDT | 100x | OK | **100x** |
| SOL/USDT | 100x | "Leverage 100 is not valid" (-4028) | **50x** |

Margin mode : **isolated** sur les 3 symboles (déjà OK sur BTC, set sur ETH/SOL).

### 2.3 Bleed wallet testnet $4998 → $103

Objectif utilisateur : avoir un wallet à ~$100 pour mimer 1:1 les conditions mainnet (pas de coussin "fictif" qui change le comportement live).

**Méthode** : round-trips market buy/sell BTCUSDT à $45k notional, ~$37 fees+slippage par cycle.

**Résultat** : $4998.75 → $103.87 en 130 round-trips (~10 min). Script `binance_bot/bleed_to_100.py`.

Note : pas d'API transfer/withdraw sur testnet futures. Seul "Reset Balance" UI permet de modifier le solde, mais ramène au montant par défaut imposé par Binance, pas à un montant choisi.

## 3. Refactor multi-symbole

### 3.1 Demande utilisateur

> *"On va trader sur l'eth et le sol aussi même principe : 1% du capital par trade, max 3 trade par crypto soit 3 buy soit 3 sell, chaque symbole indépendant"*

Translation technique : par symbole, max 3 LONG XOR 3 SHORT (jamais mixé). Worst case 3 cryptos × 3 positions = 9 positions max sur le compte.

### 3.2 Audit architecture mono → multi

| Module | Couplage BTC | Refactor ? |
|---|---|---|
| `bot_settings.yaml` | `symbol: BTC/USDT` singleton | OUI — liste |
| `state.json` | positions globales | OUI — sections par symbole |
| `K3_1d_stable.csv` | Labels K3 calculés sur BTC seul (Fourier) | NON (option A : reuse BTC labels) |
| `phase_params_K3.json` | Params Optuna calibrés sur BTC | NON (option A) |
| `DataFetcher`, `TradeManager`, `SignalEngine`, `reconciler` | Prennent `symbol` en param | NON — instanciation × N |
| `daily_phase_job.py` | Phase BTC globale | NON (option A) |

**Choix utilisateur option A** : réutiliser les labels K3 et params BTC tels quels sur ETH/SOL. C'est sous-optimal quant (zéro garantie d'edge sur ETH/SOL) mais cohérent avec l'objectif démo testnet : valider que le flow buy/sell fonctionne sur 3 symboles.

### 3.3 Implémentation

#### `bot_settings.yaml` (refactor)

```yaml
symbols:
  - { pair: "BTC/USDT", leverage: 125 }
  - { pair: "ETH/USDT", leverage: 100 }
  - { pair: "SOL/USDT", leverage: 50 }
position_size_pct: 0.01
max_positions_per_symbol: 3   # 3 LONG XOR 3 SHORT par symbole
# Compat ascendant
symbol: "BTC/USDT"
max_leverage: 125
max_positions_per_side: 3
```

#### `state.json` (migration auto mono → multi)

```json
{
  "initial_capital_usdt": 100.0,
  "equity": 1.0,
  "phase_today": 2,
  "params_today": { "tenkan": 21, "kijun": 35, ... },
  "symbols": {
    "BTC/USDT": { "positions_long": [], "positions_short": [] },
    "ETH/USDT": { "positions_long": [], "positions_short": [] },
    "SOL/USDT": { "positions_long": [], "positions_short": [] }
  }
}
```

`StateManager._load_or_initialize` détecte automatiquement l'ancien format (positions globales) et les migre sous `symbols.BTC/USDT`.

#### `state_manager.py`

Méthodes étendues avec kwarg optionnel `symbol`:
- `add_position(side, entry, stop, tp, size, symbol=None)`
- `remove_position(side, pos_id, symbol=None)`
- `get_positions(side, symbol=None)`
- Nouvelle méthode `ensure_symbol(symbol)` pour garantir la section existe

Si `symbol=None` → comportement legacy global. Si fourni → scoped à `state.symbols[symbol]`.

#### `intraday_runner.py`

Refactor : extraction de la logique par-symbole en boucle `for sym_cfg in sym_cfgs`. Chaque itération :
1. Instancie son propre `DataFetcher`, `TradeManager`, `SignalEngine`
2. Charge positions depuis `state.symbols[symbol]`
3. Détecte signaux (Ichimoku H2 + phase params globaux)
4. Exécute trades avec leverage du symbole
5. Persiste positions dans `state.symbols[symbol]`

Le `RiskManager` reste partagé (capital + position_size_pct identiques entre symboles). Le stop global est vérifié au niveau compte AVANT toute exécution multi-symbole.

#### `signal_engine.py` — règle XOR déjà câblée

Bonne surprise : la règle "3 LONG XOR 3 SHORT par symbole" était **déjà implémentée** dans le code existant :
- Ligne 127 : `if signal_long and len(positions_short) == 0:` → bloque ouverture LONG si SHORTs existent
- Ligne 146 : `if signal_short and len(positions_long) == 0:` → bloque ouverture SHORT si LONGs existent
- Lignes 164-182 : rotation propre via `opposite_signal` (ferme avant de réouvrir l'autre side)

Aucune modification nécessaire. Il a suffi d'instancier une `SignalEngine` par symbole avec `max_positions=3`.

## 4. Validation tests

### 4.1 Tests unitaires

`pytest binance_bot/tests/` → **206/206 verts**, durée 11.75s. Aucune régression.

### 4.2 Test simulation multi-symbole

```bash
TRADE_MODE=simulation python -m binance_bot.routines.intraday_runner
```

Sortie : 3 symboles traités (BTC, ETH, SOL), prix lus, Ichimoku calculé, 0 signaux (phase 2 calme), state persisté correctement.

### 4.3 Tests live testnet

#### Round-trip LONG (open buy + close sell)

| Symbole | Notional | Open ID | Close ID | Coût |
|---|---:|---|---|---:|
| BTC/USDT | $124.46 | 13083503710 | 13083503783 | $0.11 |
| ETH/USDT | $99.73 | 8668968765 | 8668968776 | $0.09 |
| SOL/USDT | $49.46 | 1857049472 | 1857049483 | $0.06 |

#### Round-trip SHORT (open sell + close buy)

| Symbole | positionAmt après open | Après close | Coût |
|---|---:|---:|---:|
| BTC/USDT | -0.0016 (short ✓) | 0.0 | $0.10 |
| ETH/USDT | -0.043 (short ✓) | 0.0 | $0.08 |
| SOL/USDT | -0.58 (short ✓) | 0.0 | $0.09 |

Le user a explicitement demandé de tester aussi le SHORT (bon catch — j'avais oublié).

#### Round-trip via `TradeManager.execute_signal` (intégration bot)

Validation que le wrapper du bot route correctement open_short/close_short avec algoOrders TP/SL placés côté Binance.

## 5. Bugs critiques détectés et fixés

### 5.1 Bug `qty round(3)` hardcodé pour BTC

**Symptôme** : `_open_long` et `_open_short` faisaient `qty = round(qty, 3)`. Pour SOL où `precision.amount = 0.001` mais le step réel est 0.01 sur testnet, `0.587` était envoyé alors que Binance attendait `0.58`.

**Fix** : `qty = float(self.exchange.amount_to_precision(self.symbol, raw_qty))` — précision auto par symbole via ccxt.

### 5.2 Bug algoOrder `-1111 Precision is over the maximum`

**Symptôme** : sur ETH et SOL, les `algoOrder` STOP_MARKET et TAKE_PROFIT_MARKET échouaient silencieusement avec code `-1111`. Cause : `triggerPrice` envoyé brut (avec trop de décimales) au lieu d'être normalisé à la `tickSize` du symbole.

**Conséquence si non fixé** : si le bot ouvrait une position et crashait avant le close, les TP/SL n'auraient PAS été actifs côté Binance. Sécurité dégradée critique.

**Fix** : dans `_place_algo_order`, normaliser via :
```python
qty_norm = float(self.exchange.amount_to_precision(self.symbol, qty))
trigger_norm = float(self.exchange.price_to_precision(self.symbol, trigger_price))
```

### 5.3 Bug cosmétique : logs hardcodés "BTC"

`print(f"✅ LONG ouvert: {qty} BTC @ {entry}")` → remplacé par `f"✅ LONG ouvert: {qty} {self.symbol} @ {entry}"` (et idem SHORT, simulation, close).

### 5.4 Mock test fixture mis à jour

`_make_exchange_mock()` dans `test_bot_trade_manager.py` : ajout de `amount_to_precision` et `price_to_precision` en passthrough pour préserver le comportement attendu des tests existants.

**Validation post-fix** : 206/206 tests verts + retest live SHORT round-trip avec algoOrders TP+SL correctement posés sur les 3 symboles (`STOP_MARKET trigger=...` + `TAKE_PROFIT_MARKET trigger=...`).

## 6. Audit professionnel Layer 3

Délégation à un agent quant senior pour identifier ce qui manque encore vs standards institutionnels (hedge fund quant, market makers crypto), au-delà des 10 sections déjà couvertes par Layer 2.

**Résultat** : `RAPPORT_GAPS_LAYER3_2026-04-27.md` (645 lignes, 16 sections nouvelles), avec citations papiers réels (SSRN, arXiv, ScienceDirect, MDPI, Springer, FIA, Binance docs, Fed SR 11-7) toutes vérifiables.

**Top 5 actionnable** consolidé dans `RECOMMANDATIONS_TOP5_2026-04-27.md` :

1. `portfolio_risk.py` — corrélation rolling + portfolio CVaR + leverage agrégé cap 4x (CRITICAL)
2. `psr_live.py` — Probabilistic Sharpe Ratio rolling 60j live vs backtest (CRITICAL)
3. Funding-aware close-before-settlement (IMPORTANT, quick win)
4. `pre_trade_gate.py` — 5 règles FIA fat-finger (CRITICAL avant mainnet)
5. Implementation Shortfall complet Perold (IMPORTANT)

## 7. Fichiers modifiés / créés

### Modifiés
- `binance_bot/configs/bot_settings.yaml` — multi-symbole
- `binance_bot/data/state.json` — migration auto multi-symbole
- `binance_bot/bot/state_manager.py` — kwargs `symbol=`
- `binance_bot/bot/trade_manager.py` — précision auto, logs corrects, fix algoOrder
- `binance_bot/routines/intraday_runner.py` — boucle multi-symbole
- `binance_bot/tests/test_bot_trade_manager.py` — mock passthrough précision

### Créés
- `algo_trading_haute_frequence/` (ce dossier)
- `algo_trading_haute_frequence/README.md`
- `algo_trading_haute_frequence/RAPPORT_TRAVAUX_2026-04-27.md` (ce fichier)
- `algo_trading_haute_frequence/RECOMMANDATIONS_TOP5_2026-04-27.md`
- `algo_trading_haute_frequence/RAPPORT_GAPS_LAYER3_2026-04-27.md`
- `binance_bot/bleed_to_100.py` (script utilitaire — peut être supprimé, plus utile)

### Déplacés depuis racine vers `algo_trading_haute_frequence/`
- `RAPPORT_BILAN_QUANT_2026-04-27.md`
- `RAPPORT_GAPS_LAYER2_2026-04-26.md`
- `RAPPORT_GAPS_LAYER3_2026-04-27.md`
- `RAPPORT_ALPHA_FINAL_2026-04-26.md`
- `RAPPORT_ALPHA_RECHERCHE_2026-04-26.md`
- `RAPPORT_AUDIT_LOOKAHEAD_2026-04-26.md`
- `RAPPORT_COSTS_2026-04-26.md`

## 8. État final du compte testnet

- Wallet : **$103.04** (après 6 round-trips de validation × ~$0.18 fees)
- Aucune position ouverte sur les 3 symboles
- Aucun ordre en attente
- Margin mode : isolated sur BTC/ETH/SOL
- Leverages : 125 / 100 / 50
- Bot en mode simulation par défaut (yaml committé), live activable via `TRADE_MODE=live` env var
- 4 timers systemd actifs : intraday H2, daily 23:55 UTC, watchdog 5min, backup 6h

## 9. Décisions et choix faits

| Décision | Choix | Justification |
|---|---|---|
| Multi-symbole : labels/params per-asset ? | **Option A** : reuse BTC | Démo testnet, pas de calibration ETH/SOL nécessaire pour valider le flow |
| Wallet testnet : laisser $4998 ou bleed ? | **Bleed à ~$100** | User veut conditions 1:1 mainnet (pas de coussin fictif) |
| Refactor structure code : sous-dossier ? | **Non, code reste en place** | Éviter de casser systemd, imports, paths. Nouveau dossier sert de hub doc/synthèse |
| Algo trading high frequence : vrai HFT ? | **Non, naming demandé par user** | Bot tourne sur H2 (2h candles), pas du sub-second HFT. Terme préservé sur demande |

## 10. Prochaines étapes

Avant de faire QUOI QUE CE SOIT en mainnet :
1. Implémenter **#4 pre_trade_gate** (heures) — bloqueur absolu
2. Implémenter **#3 funding_close** (heures) — quick win
3. Implémenter **#2 psr_live** (1j) — détection alpha decay
4. Implémenter **#1 portfolio_risk** (1-2j) — gestion risque agrégé multi-symbole
5. Laisser tourner testnet 2-4 semaines pour collecter PSR live et valider que la divergence vs backtest reste bornée
6. Bilan : si PSR live confirme PSR backtest négatif → SWITCH HODL spot ou pivot. Sinon → rerun Optuna ETH/SOL séparément, considérer mainnet à dose homéopathique avec leverage descendu à 2-3x

**Garder en tête le verdict du bilan quant** : aucune stratégie testée ne bat HODL spot BTC (Sharpe 1.09). Le code est rigoureux, la stratégie n'a pas d'edge documenté.
