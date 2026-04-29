# LIMITATIONS ACTÉES — HSBC-algo-genetic

> Document tracking les limitations connues et acceptées d'un chunk au moment
> de son commit. Permet de distinguer ce qui est **cosmétique aujourd'hui**
> (fonctionne en mode dégradé safe) de ce qui sera **bloquant demain** dès
> qu'un mode opérationnel précis est activé.
>
> Convention: chaque entrée = un point précis identifié par origine, impact,
> et chunk de résolution. Mettre à jour à chaque nouveau chunk si une
> limitation similaire apparaît.

---

## L-001 — VPIN data_fn placeholder retourne None — **RÉSOLUE + ACTIVÉE**

- **Status**: ✅ RÉSOLUE + ACTIVÉE 2026-04-29 sur cette machine.
  Code mergé `f5f22d7` 2026-04-28, daemon `hsbc-vpin-live.service`
  enabled+running depuis 2026-04-29 05:29 CEST. Bucket sizes calculées
  V/N=50 sur 7d klines : BTC=158M, ETH=123M, SOL=22.7M USDT. WS
  connectés (6 streams : 3 symbols × @trade + @bookTicker). Records
  flushés dans `data/vpin_live.jsonl` toutes les 30s.
- **Origine**: R5 / P7 — `binance_bot/routines/intraday_runner.py` fonction
  `_make_vpin_data_fn(symbol)`.
- **Description**: La fonction retournait toujours `None` jusqu'à P7-bis.
  Maintenant lit `data/vpin_live.jsonl` (filtré par symbol, stale 10min,
  validation [0,1] strict). Source : daemon `hsbc-vpin-live.service`
  (P7-bis) qui collecte via WS Binance `<sym>@trade` + `<sym>@bookTicker`.
  WS-001 mitigation : utilise `@trade` au lieu de `@aggTrade` (sémantiquement
  équivalent BVC).
- **Activation requise par Sky** :
  ```bash
  # 1. Flip le yaml
  sed -i 's/vpin_collector_enabled: false/vpin_collector_enabled: true/' \
    binance_bot/configs/bot_settings.yaml
  # 2. Lien systemd
  systemctl --user link \
    /home/ludov/HSBC-algo-genetic/binance_bot/systemd/hsbc-vpin-live.service
  systemctl --user daemon-reload
  systemctl --user enable --now hsbc-vpin-live.service
  # 3. Vérification 5 min après
  tail -5 binance_bot/data/vpin_live.jsonl  # devrait avoir des records
  scripts/bot_status.sh | grep vpin_live   # ligne count > 0
  ```
- **Tant que non activé** : data_fn retourne `None` → gate VPIN reste
  désactivé silencieusement (`("allow", "vpin_data_unavailable")`).
  Comportement identique au pré-P7-bis. Aucun risk side-effect.
- **Chunk de résolution**: **P7-bis CLOSED** (commits `97ef129`, `0ace487`,
  `64c8580`, `dfb195d`, mergé `f5f22d7`).
- **Impact si non résolue**:
  - **Cosmétique** en mode `vpin_mode: "log_only"` (default actuel) : le gate
    n'aurait rien fait de toute façon en log_only, donc l'absence de data
    ne change pas le comportement live.
  - **BLOQUANT** en mode `vpin_mode: "gate"` : si Sky bascule en gate sans
    avoir résolu cette limitation, le gate restera disabled silencieusement
    (toujours `allow`). Aucune cascade ne sera détectée. Risque réel.
- **Garde-fou**: avant de basculer `vpin_mode: "gate"` dans `bot_settings.yaml`,
  vérifier que P7-bis est CLOSED et que `data/vpin_live.jsonl` reçoit des
  records. Test : forcer VPIN=0.95 dans la dernière ligne du jsonl, lancer
  intraday_runner, observer qu'un signal de close apparaît.

---

## L-002 — Meta-label features pre_trade partiellement None — **RÉSOLUE 7/10**

- **Status**: ✅ RÉSOLUE 7/10 (R7-bis mergé `f5f22d7` 2026-04-28).
  Mécanisme `features_snapshot` capture 7 features à l'OPEN ; 3 champs
  restent None faute de source (à traiter par chunks dédiés).
- **Origine**: R7 / P9 — `binance_bot/routines/intraday_runner.py` fonction
  `_build_meta_context` (champs `pre_trade.{rv_predicted_har, vpin_at_entry,
  cloud_breakout_size_atr_units, volume_relative_30d, funding_rate_at_entry_bps,
  obi_at_entry}` + `context.btc_dominance`).
- **Description**: Avant R7-bis ces champs étaient à `None` parce
  qu'aucune source ne les capturait au moment de l'OPEN (cycle-time vs
  trade-time mismatch). R7-bis introduit
  `_build_features_snapshot_at_open` (intraday_runner.py:209) qui est
  appelé à chaque OPEN et persisté dans `pos["features_snapshot"]` via
  `state_manager.add_position(features_snapshot=...)`. Au CLOSE, le
  snapshot est relu et passé à `_build_meta_context(entry_features=...)`
  pour enrichir le meta_label P9.
- **Features désormais capturées à l'OPEN (7/10)** :
  - `atr_at_entry` ✓ — depuis df_ichimoku.iloc[-1]['ATR']
  - `regime_har` ✓ — via regime_gate_fn() (P4 HAR-RV)
  - `composite_signal` ✓ — via composite_log_fn() (P6.5)
  - `vpin_at_entry` ✓ — via vpin_data_fn() (subordonné à L-001 :
    valeur reste None tant que collecteur VPIN live pas activé)
  - `obi_at_entry` ✓ — idem (tuple[1] du vpin_data_fn)
  - `rv_predicted_har` ✓ — via src.har_rv.predict_rv (≥200 bars)
  - `cloud_breakout_size_atr_units` ✓ — derivé Ichimoku
    (close - cloud_top) / ATR
- **Features restant None (3/10)** — chunks dédiés futurs :
  - `funding_rate_at_entry_bps` : flow_oi a la donnée mais reader pas
    branché. Chunk : R10-bis (lecteur flow_oi.jsonl tail au open).
  - `volume_relative_30d` : avg volume 30j pas tracké. Chunk :
    R10-bis (collecteur klines daily 30j rolling).
  - `btc_dominance` : pas d'API CoinGecko/CMC configurée. Chunk : R11
    (client + cache 5min).
- **Chunk de résolution**: **R7-bis CLOSED** (commits `dfb195d`,
  mergé `f5f22d7`). Les 3 features restantes : R10-bis / R11 (futurs).
- **Impact si non résolue (résiduel sur 3/10)**:
  - **Cosmétique aujourd'hui** : `build_meta_label` accepte None (test
    `test_meta_label_handles_missing_optional_features`). Munin reçoit
    schema partiel mais valide. Hash chain OK.
  - **Bloquant partiel** pour clustering Munin sur funding/volume/btcdom :
    si Munin veut cluster sur "funding adverse + cap entrée" la dim
    sera dégradée tant que funding pas branché.
- **Garde-fou**: avant d'utiliser `trades_meta.jsonl` pour entraîner
  Munin/un classifier sur features pre_trade, vérifier la fraction de
  None par champ via :
  ```bash
  jq -r '.pre_trade | to_entries[] | "\(.key)\t\(.value)"' \
    binance_bot/data/trades_meta.jsonl | \
    awk -F'\t' '$2=="null"{n[$1]++} END{for(k in n) print k, n[k]}'
  ```
  Devrait être <10% pour les 7 features capturées par R7-bis.

## L-003 — VPIN data_fn placeholder (rappel L-001 sous nouveau angle)

`vpin_at_entry` dans meta_context = None à cause de L-001. Les deux limitations
seront résolues simultanément quand P7-bis (collecteur VPIN live) sera CLOSED.

## L-004 — Features composite + phase_K3 mockées à 0 dans le training P10

- **Origine**: R8 / P10 — `scripts/training/train_lgbm_combinator.py` fonction
  `build_full_features` (lignes ~225-235).
- **Description**: 5 features sont mises à 0 constant dans le dataset
  d'entraînement faute d'historique disponible :
    - `composite_signal_score` (P6.5) — collecteurs P6.1-P6.4 créés R3
      mais data live commence 2026-04-28, pas d'historique 4 ans
    - `comp_top_ls`, `comp_taker`, `comp_liq`, `comp_oi` (idem)
    - `phase_K3` (K3 daily phase rotation pas trivial à reconstituer
      historiquement sans le pipeline daily_phase_job complet)
- **Cause**: features dépendent de modules live (P6 polling) qui n'ont
  pas de données historiques rétroactives. La spec P10 disait "Dataset
  3-4 ans BTC H1 post-2021" mais la stack flow ne couvre pas cette période.
- **Chunk de résolution**: **R8-bis (futur)** — soit (a) attendre 30j+
  d'historique vivant via R3 collecteurs et re-train avec data réelle,
  soit (b) backfill historique via APIs alternatives (Binance n'expose
  topLongShortRatio que sur 30j).
- **Impact si non résolue**:
  - **Cosmétique au plan pipeline** : LightGBM ignore les features
    constantes (gain de splitting nul). L'AUC mesurée reflète alors
    uniquement les features non-mockées (VPIN, HAR, EGARCH, time-of-day,
    days_since_halving).
  - **BLOQUANT pour AUC > 0.55** : sans le composite signal qui est censé
    être le contributeur majeur (selon spec), l'AUC plafonne probablement
    sous le seuil deploy_min. `should_deploy` retournera donc False
    légitimement.
- **Mesure R8 (2026-04-28, quick mode)**:
    - val_auc = 0.6267 (au-dessus de AUC_DEPLOY_MIN=0.55, sous AUC_SUSPECT=0.65)
    - dsr_p   = 5e-126 (rejet écrasant après déflation 20 trials Optuna)
    - should_deploy = False (DSR < 0.95)
    - Conclusion : le signal est bien là sur 17000 bars BTC H1 mais reste
      sous le seuil de bruit attendu pour 20 trials de hyperparam search.
- **Garde-fou**: avant de bypass `should_deploy=False`, vérifier que les
  5 features ne sont plus mockées dans `train_lgbm_combinator.py:build_full_features`.

## L-007 — VPIN bucket calibration unit mismatch — **RÉSOLUE** (2026-04-29)

- **Status**: ✅ RÉSOLUE par fix `vpin_live_collector.py:add_trade`
  + test régression `test_vpin_live_builder_calibration_realistic_units_l007`.
  Découvert via audit honnête lors de l'activation P7-bis.
- **Origine**: P7-bis / R3 — `binance_bot/services/vpin_live_collector.py`
  fonction `VPINLiveBuilder.add_trade(price, qty, ts_ms)`.
- **Description**: `compute_bucket_sizes` retourne bucket_size_v en
  **quote currency (USDT)** — formule canonique Easley 2012 V/N sur
  `quote_volume_daily_7d` des klines (~158M USDT pour BTC). Mais
  `add_trade` stockait `q` (le payload Binance @trade `q` field) qui
  est en **base currency (BTC)**. Comparaison incohérente dans
  `build_volume_buckets` → ratio ~50000× trop petit → **0 buckets ne
  ferment jamais en production**.
- **Détection**: 2026-04-29 ~06:18 (45min après activation), heartbeat
  daemon : BTC=124k events, ETH=229k events, SOL=40k events,
  `n_buckets=0` partout, `connection_errors=0`. WS stables, OBI
  fonctionnel, mais VPIN figé à 0.0.
- **Cause racine**: tests P7-bis utilisaient `bucket_size_v=10.0` +
  `qty=1.0` (volumes abstraits sans dimensionnalité). Le mismatch ne
  se manifestait pas avec des unités arbitraires, n'apparaissait qu'au
  contact production.
- **Fix**: `self._volumes.append(p * q)` au lieu de `self._volumes.append(q)`
  → stocke notional USDT cohérent avec `bucket_size_v` USDT.
  VPIN value invariante (compute_vpin = mean(|buy-sell|/total),
  ratios indépendants de l'unité), seule la VITESSE de fermeture
  change : avant ~28 jours/bucket BTC, après ~28 min/bucket sur flow
  mainnet normal (158M / ~5.6M /min = ~28 min).
- **Test régression**: `test_vpin_live_builder_calibration_realistic_units_l007`
  utilise désormais price=$50k, qty=0.5 BTC, bucket=$158M. Asserts
  ≥1 bucket fermée sur 7000 trades. Si ce test repasse au vert avec
  `_volumes.append(q)`, c'est que le bug est revenu.
- **Impact si non résolue (rétro)**:
  - **Cosmétique** en `vpin_mode: "log_only"` (config actuelle) :
    gate désactivé silencieusement (`("allow", "vpin_data_unavailable")`),
    comportement identique au pré-P7-bis.
  - **BLOQUANT** en `vpin_mode: "gate"` : aucune cascade détectée,
    gate inutile (toujours allow). Recalibration nécessaire avant
    bascule gate.
- **Garde-fou**: avant bascule `vpin_mode: "gate"`, vérifier
  `data/vpin_live.jsonl` contient des records avec `n_buckets > 0`
  (au moins 5-10 sur 1h de soak, sinon investigation).

## L-005 — flow_liquidations daemon désactivé (WS-001)

- **Origine**: commit R3 (P6.3) — daemon hsbc-flow-liquidations créé
  mais Binance fstream.binance.com ferme le stream `!forceOrder@arr`
  silencieusement via CLOSE frame post-SUBSCRIBE (cf BUGS.md WS-001).
- **Description**: pas de data live de liquidations agrégées dans
  `data/flow_liq_buckets.jsonl` (fichier absent depuis le début).
  Daemon désactivé via `systemctl --user disable hsbc-flow-liquidations`
  pour ne pas garder de connexion WS zombie.
- **Cause**: Binance Futures USDM serveur-side. Endpoints
  `<symbol>@aggTrade` et `<symbol>@forceOrder` (et `!forceOrder@arr`)
  ferment la connexion proactivement. Pas un bug client.
- **Chunk de résolution**: à ouvrir séparément, pas urgent.
  Options envisagées :
    - REST polling `/fapi/v1/forceOrders` (limité au compte
      authentifié, ne convient pas pour cascade detection global)
    - Service externe (CoinGlass API, Coinalyze)
    - Ticket support Binance pour confirmer si CLOSE frame est
      intentionnel (deprecation) ou bug temporaire
- **Impact si non résolue**:
  - **Cosmétique aujourd'hui en mode log-only P6.5** :
    `compute_composite_score` accepte liq absent gracieusement.
    Test runtime confirme : `score = -0.207` (finite, pas NaN, pas
    crash) avec `components.liq = 0.0` et `n_obs.liq = 0`. Le
    composite continue à fonctionner sur 3/4 features.
  - **BLOQUANT pour activation mode "gate" P6.5** :
    score sous-estimé d'un facteur 0.75 (poids 0.25 de liq inclus
    mais toujours nul). Si on bascule en gate avec seuil 0.4, le
    seuil devient effectivement 0.4 / 0.75 = 0.53 sur les 3
    features dispo. À recalibrer post-fix.
  - **BLOQUANT pour analyse Munin clustering** : la dimension
    "capitulation" (liq imbalance) manque, l'analyse posteriori
    sera 4 → 3 dimensions (perte d'info pour clustering events
    type "long capitulation cascade").
- **Garde-fou**: avant de basculer P6.5 en `gate` mode, soit
  résoudre L-005 (collecteur alt), soit recalibrer les seuils
  composite explicitement pour 3 features (multiplier seuils par
  4/3 ≈ 1.33).

## Format pour futures entrées

```
## L-NNN — [titre court]

- **Origine**: [chunk] / [P-N] — [fichier:ligne]
- **Description**: [1-3 lignes du problème observé]
- **Cause**: [pourquoi ça marche pas dans le mode visé]
- **Chunk de résolution**: [PNNN-bis ou Rxxx], task #NN
- **Impact si non résolue**:
  - **Cosmétique** en mode [X] : [pourquoi ça ne pose pas de problème]
  - **BLOQUANT** en mode [Y] : [scenario concret où ça casserait]
- **Garde-fou**: [check à faire avant d'activer le mode bloqué]
```
