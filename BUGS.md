# BUGS — HSBC-algo-genetic

> Format: each bug has an ID, status, symptom, root cause, fix, and test.
> This file is READ BY CLAUDE AT BOOT. Keep it accurate.

<!-- TEMPLATE
## BUG-XXX: [short description]
- **Status**: OPEN / FIXED / WONTFIX
- **Symptom**: what happens
- **Root cause**: WHY it happens (not just where)
- **Fix**: what was done (commit hash if fixed)
- **Test**: which test covers this (file:test_name)
- **Regression**: did the fix break anything else?
-->

## BUG-PRE-001: tests legacy timeout > 30s bloquent Forge
- **Status**: FIXED (R-FIX-FORGE 2026-04-28) — Forge réutilisable avec `pytest -m "not slow"`
- **Symptom**: tests legacy (ACO smoke, alpha_backtest, regime_lgbm, MSM,
  WFA pipeline) excèdent 30s. Forge fail sur tous les chunks à cause de
  ces tests legacy même quand le chunk testé n'a aucun rapport.
- **Root cause**: tests d'intégration lents par nature (MLE Nelder-Mead
  sur 64 états MSM, WFA pipeline full sur 3 ans, ACO 100+ itérations).
  Pas de bug fonctionnel, juste durée.
- **Fix**: marker `@pytest.mark.slow` (au niveau module via `pytestmark`)
  sur 5 fichiers concernés + déclaration `markers = slow:` dans `pytest.ini`.
- **Commande Forge** : `pytest -m "not slow"`. **607/607 tests** passent
  en ~75s vs ~240s avec le suite complet (gain 3.2×).
- **Tests slow toujours run-able localement** : `pytest -m "slow"` ou
  `pytest tests/test_p12_msm_calvet.py` directement.
- **Fichiers markés slow**:
  - `tests/test_aco_basic.py`
  - `tests/test_alpha_backtest.py`
  - `tests/test_regime_lgbm.py`
  - `tests/test_p12_msm_calvet.py`
  - `tests/test_wfa_pipeline.py`

## QUANT-002: White's Reality Check + Hansen SPA test
- **Status**: FIXED (2026-04-26, 14/14 tests passent — `tests/test_reality_check.py`)
- **Symptom**: aucune significativité statistique multi-stratégie. Avec K configurations testées en parallèle, le best Sharpe peut être aléatoire.
- **Fix**: nouveau module `src/reality_check.py` avec `stationary_bootstrap_indices` (Politis-Romano 1994), `whites_reality_check` (White 2000), `hansen_spa_test` (Hansen 2005, 3 versions: lower/consistent/upper).
- **À intégrer**: dans le pipeline d'évaluation des params Optuna gagnants — calculer p-value SPA_consistent et rejeter si > 0.05.

## QUANT-001: Deflated Sharpe Ratio + PSR + Sortino + Ulcer + DD duration
- **Status**: FIXED (2026-04-26, 17/17 tests passent — `tests/test_dsr.py`)
- **Symptom**: avec N trials Optuna massifs, Sharpe rapporté biaisé (data snooping). Aucune correction → faux positifs sur l'edge.
- **Fix**: ajouté dans `src/stats_eval.py` les fonctions `probabilistic_sharpe_ratio`, `expected_max_sharpe_under_h0`, `deflated_sharpe_ratio` (Bailey & López de Prado 2012/2014). `compute_metrics` retourne aussi `sortino`, `ulcer` (Martin 1989), `dd_duration`. `METRIC_COLUMNS` étendu.
- **À intégrer**: appeler `deflated_sharpe_ratio` dans le pipeline de sélection des params Optuna (au lieu de Sharpe brut).

## BUG-001: close-position stubs vides
- **Status**: FIXED (2026-04-26, 8/8 tests passent)
- **Symptom**: en mode live, à la fermeture d'une position (TP/SL/opposite_signal), aucun ordre Binance n'est passé. La position reste ouverte sur l'exchange alors que `state.json` la considère fermée — divergence silencieuse, capital exposé.
- **Root cause**: `binance_bot/bot/trade_manager.py` `_close_long` (l.128-148) et `_close_short` (l.150-165) ne contiennent qu'un `print` puis `return None`. Pas d'appel à `cancel_order` pour SL/TP en attente, pas de `fetch_positions` pour récupérer la qty réelle, pas de market order `reduceOnly`.
- **Fix**: implémenter (1) `fetch_open_orders` puis `cancel_order` sur les SL/TP existants pour le symbol, (2) `fetch_positions` pour récupérer la qty exacte ouverte, (3) `create_market_sell_order` (long) ou `create_market_buy_order` (short) avec `params={'reduceOnly': True}`. Retourner l'order_id du market order de fermeture.
- **Test**: `binance_bot/tests/test_bot_trade_manager.py:test_close_long_cancels_pending_and_market_closes`, `:test_close_short_cancels_pending_and_market_closes` (mock ccxt.binance).
- **Regression**: vérifier que mode "simulation" continue de retourner `sim_close_*` sans toucher l'exchange.

## BUG-002: set_leverage absent au démarrage
- **Status**: FIXED (2026-04-26, 4/4 tests passent)
- **Symptom**: le bot trade en levier par défaut Binance (généralement 20x sur futures) au lieu du `max_leverage: 10` configuré. Sizing et risque réels divergent du backtest.
- **Root cause**: `binance_bot/routines/intraday_runner.py` instancie `RiskManager` avec `max_leverage` mais n'appelle jamais `exchange.set_leverage(N, symbol)` côté Binance. La valeur côté broker reste celle par défaut.
- **Fix**: après init `TradeManager` (l.67), lire `settings['max_leverage']` et appeler `data_fetcher.exchange.set_leverage(int(max_leverage), symbol)` en try/except. Logger la valeur appliquée.
- **Test**: `binance_bot/tests/test_bot_intraday_runner.py:test_set_leverage_called_on_boot` (mock exchange).
- **Regression**: en simulation, ne pas appeler set_leverage (mode local).

## BUG-003: trailing stop fixe au lieu d'ATR
- **Status**: FIXED (2026-04-26, 7/7 tests passent)
- **Symptom**: le `stop` enregistré à l'ouverture de position n'est jamais mis à jour. Comportement = stop loss fixe initial. La phrase "trailing_stop" en raison de close (`signal_engine.py:78`) est mensongère.
- **Root cause**: `binance_bot/services/signal_engine.py` `detect_signals` (l.69-72, 83-87) compare `current_price` à `pos.get("stop", ...)` mais ne réécrit jamais `pos["stop"]` à chaque itération. Le multiplicateur `atr_trailing_multiplier: 2.0` du `bot_settings.yaml` n'est lu nulle part.
- **Fix**: avant la détection des sorties, itérer sur `positions_long`/`positions_short` et faire `pos["stop"] = max(pos["stop"], current_price - atr * atr_trailing_mult)` pour LONG, `min(pos["stop"], current_price + atr * atr_trailing_mult)` pour SHORT. Ajouter `atr_trailing_mult` au constructeur (default 2.0).
- **Test**: `binance_bot/tests/test_bot_signal_engine.py:test_trailing_stop_ratchets_up_long`, `:test_trailing_stop_ratchets_down_short`, `:test_trailing_stop_never_loosens`.
- **Regression**: vérifier que le stop n'est JAMAIS desserré (toujours plus serré au fil du prix).

## BUG-004: bougie non-clôturée utilisée pour signal
- **Status**: FIXED (2026-04-26, 7/7 tests passent)
- **Symptom**: signaux Ichimoku basés sur la bougie en cours (incomplète) → faux croisements qui se résolvent à la clôture, déclenchent ordres prématurés.
- **Root cause**: `binance_bot/routines/intraday_runner.py` l.93 `data_fetcher.get_ohlcv(limit=300)` retourne potentiellement la bougie en formation. La dernière ligne du DataFrame est utilisée directement (l.105) sans filtrer.
- **Fix**: dans `data_fetcher.get_ohlcv` (ou dans `intraday_runner` après récupération), comparer le timestamp de la dernière bougie à `now()` — si elle n'est pas clôturée (timestamp + timeframe > now), la dropper. Pour H2 : `last_ts + 2h <= now()` sinon drop.
- **Test**: `binance_bot/tests/test_bot_data_fetcher.py:test_drop_unclosed_last_candle`, `binance_bot/tests/test_bot_intraday_runner.py:test_signal_uses_only_closed_bar`.
- **Regression**: vérifier que si toutes les bougies sont clôturées, aucune n'est droppée.

## BUG-005: sizing ignore le levier
- **Status**: FIXED (2026-04-26, 9/9 tests passent — leverage clampé via validate_leverage)
- **Symptom**: la qty calculée représente `position_size_pct * equity / price` sans tenir compte du levier. Avec levier 10x configuré mais sizing en notional brut, l'exposition réelle est 1/10e de l'attendue.
- **Root cause**: `binance_bot/bot/risk_manager.py` `calculate_position_size` (l.49-50) et `binance_bot/bot/trade_manager.py` `_open_long` (l.55-56) / `_open_short` (l.96-97) calculent `qty = capital * size_pct / price` sans multiplier par le levier. La méthode `validate_leverage` (l.53-55) existe mais n'est appelée nulle part.
- **Fix**: ajouter `leverage` (default 1.0) au constructeur de `TradeManager`, le passer depuis `intraday_runner` (lecture `settings['max_leverage']`). Dans `_open_long`/`_open_short`: `qty_btc = (capital_usdt * size_pct * leverage) / entry`. Idem dans `RiskManager.calculate_position_size` (signature `(equity, price, leverage=1.0)`).
- **Test**: `binance_bot/tests/test_bot_trade_manager.py:test_open_long_sizing_with_leverage`, `binance_bot/tests/test_bot_risk_manager.py:test_calculate_position_size_with_leverage`.
- **Regression**: avec leverage=1.0, comportement identique à l'ancien.

## BUG-006: pas de réconciliation Binance↔state.json
- **Status**: FIXED (2026-04-26, 10/10 tests passent — module bot/reconciler.py)
- **Symptom**: si le bot crash/reboot pendant qu'une position est ouverte, ou si une position est fermée manuellement sur Binance, `state.json` et l'état réel divergent. Le bot peut tenter d'ouvrir des positions concurrentes, ou ignorer une position existante.
- **Root cause**: `binance_bot/routines/intraday_runner.py` l.108-110 charge `positions_long`/`positions_short` depuis `state.json` directement, sans vérifier l'état réel via `exchange.fetch_positions(symbol)`.
- **Fix**: après `state_mgr` init, appeler `exchange.fetch_positions([symbol])`, comparer les positions ouvertes côté Binance avec `state.json`. Si divergence : (a) si Binance a une position non listée → ajouter au state, (b) si state liste une position absente côté Binance → la retirer du state. Logger toute correction. À faire UNIQUEMENT en mode live.
- **Test**: `binance_bot/tests/test_bot_intraday_runner.py:test_reconcile_adds_missing_position`, `:test_reconcile_removes_phantom_position`.
- **Regression**: en simulation, ne pas appeler fetch_positions.

## BUG-BONUS: ccxt 4.4+ casse futures sandbox
- **Status**: FIXED (2026-04-26, ccxt>=4.0.0,<4.4 dans binance_bot/requirements.txt)
- **Symptom**: `set_sandbox_mode(True)` lève `NotSupported` pour `binanceusdm` depuis ccxt 4.4. Impossible de tester sur testnet.binancefuture.com.
- **Root cause**: changement breaking dans ccxt 4.4. Versions <4.4 (testées 4.3.98) fonctionnent.
- **Fix**: pin `ccxt<4.4` dans `binance_bot/requirements.txt`.
- **Test**: import + sandbox mode validé via `binance_bot/test_connection.py`.
- **Regression**: pour mainnet (sans sandbox), les versions plus récentes sont OK — mais on garde le pin pour cohérence testnet.

## BUG-CRITICAL: trade_mode hardcodé "live"
- **Status**: FIXED (2026-04-26, dans le même Edit que BUG-002 — settings.trade_mode || (BINANCE_TESTNET=false → live, sinon simulation). Défaut = simulation.)
- **Symptom**: `binance_bot/routines/intraday_runner.py:62` `trade_mode = "live"` en dur, indépendamment de `.env` ou `bot_settings.yaml`. Lancer le runner = trader en réel, sans garde-fou. Cohérent avec les pertes constatées.
- **Root cause**: code écrit pour test rapide jamais reverti.
- **Fix**: lire `os.environ.get('BINANCE_TESTNET', 'true').lower() == 'false'` ou un champ dédié `trade_mode` dans `bot_settings.yaml` (default "simulation"). Logger explicitement le mode au démarrage.
- **Test**: `binance_bot/tests/test_bot_intraday_runner.py:test_trade_mode_default_is_simulation`, `:test_trade_mode_from_env`.
- **Regression**: aucune — c'est un fix de safety net.

## BUG-B13: pas de stream WebSocket user data (latence REST inacceptable)
- **Status**: FIXED (2026-04-26, 25/25 tests passent — `binance_bot/tests/test_bot_user_stream.py` (16) + `test_bot_event_queue.py` (9) + smoke testnet 30s OK, listenKey créé `r4h9Vo9i...`, 0 connection_error, WS open/close clean)
- **Symptom**: les fills/positions étaient récupérés via REST polling (`fetch_positions`, `fetch_open_orders`) toutes les N secondes. En marché volatil, latence inacceptable : un fill peut être visible jusqu'à plusieurs secondes après l'exécution → SL/TP calculés sur des positions partiellement remplies, exposition non contrôlée.
- **Root cause**: aucune intégration de l'user data stream Binance USDM Futures (`wss://stream.binancefuture.com/ws/{listenKey}`). Les events `ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`, `MARGIN_CALL` ne sont pas écoutés. Binance recommande explicitement le WS pour le live trading.
- **Fix (design)**: synchrone via `websocket-client` (compat avec le reste du code synchrone du bot — pas d'asyncio). 2 nouveaux modules :
  - `binance_bot/bot/user_stream.py` — classe `UserStreamListener(api_key, api_secret, base_rest, base_ws, on_event_callback)`. `start()` lance 2 threads daemon : (1) `_ws_loop` qui crée listenKey via POST /fapi/v1/listenKey, ouvre `WebSocketApp`, dispatch `_on_message` → callback ; auto-reconnect avec backoff exponentiel (2^n borné à 60s). (2) `_keepalive_loop` qui PUT /fapi/v1/listenKey toutes les `KEEPALIVE_INTERVAL_SEC = 30 * 60` (marge sur les 60min officiels). Sur event `listenKeyExpired` → `_reconnect_requested.set()` + ws.close() → recréation propre. `stop()` close WS + DELETE listenKey + join threads (idempotent).
  - `binance_bot/bot/event_queue.py` — classe `EventQueue` thread-safe (deque + Lock) avec persistance JSONL append-only sous `binance_bot/data/user_stream_events.jsonl` pour replay/debug. API : `push(dict)`, `pop_all() -> list[dict]`, `size()`, `close()`. Buffer borné (`max_buffer=100000`, drop FIFO en overflow ; le JSONL conserve TOUT).
- **Test**: `binance_bot/tests/test_bot_user_stream.py` — 16 tests : create_listen_key (POST signé, X-MBX-APIKEY), keepalive PUT, close DELETE, dispatch ORDER_TRADE_UPDATE/ACCOUNT_UPDATE, JSON invalide, callback exception isolée, listenKeyExpired → reconnect_flag + ws.close, ws_loop recreate listenKey, keepalive loop périodique, KEEPALIVE_INTERVAL_SEC=1800, start+stop end-to-end, idempotence start/stop, backoff borné. `binance_bot/tests/test_bot_event_queue.py` — 9 tests : push/pop_all, vide, skip non-dict, persist JSONL, parent_dir auto, in-memory only, concurrent 8×200 push, buffer borné FIFO, close idempotent. Smoke live : `binance_bot/scripts_smoke_userstream.py` lancé 30s contre testnet → listenKey créé OK, WS open/closed clean, 0 erreur.
- **Regression**: aucune — modules self-contained, pas d'intégration au runner dans cette session (à hooker plus tard dans `intraday_runner.py` pour remplacer le polling REST par drain de l'EventQueue).

## BUG-B12: WAL + clientOrderId déterministe pour idempotence après crash
- **Status**: FIXED (2026-04-26, 11/11 tests passent — `binance_bot/tests/test_bot_recovery_wal.py`)
- **Symptom**: entre `create_market_buy_order` (entrée) et `_place_algo_order` SL/TP, un crash laisse une position **naked** sur Binance (qty ouverte sans stop). Le `reconciler` la voit mais ne connaît pas le stop visé → position non protégée jusqu'à intervention manuelle.
- **Root cause**: aucune persistance d'intent avant l'appel API ; aucun mécanisme d'idempotence côté `clientOrderId` → un retry naïf après crash crée un doublon d'ordre côté Binance.
- **Fix**: nouveau module `binance_bot/bot/recovery_wal.py` — classe `WAL` (JSONL append-only + fsync) avec `record_intent` / `mark_completed` / `mark_failed` / `pending_intents` / `clear_completed`. Helper `deterministic_client_order_id(strategy, bar_ts_ms, leg) -> str` (≤36 chars, pure function → idempotence Binance native). Le runner enregistre l'intent AVANT l'API call et le marque completed après ; au boot, `pending_intents()` rejoue avec le **même** clientOrderId. Documentation : `binance_bot/bot/RECOVERY_NOTES.md`.
- **Test**: `binance_bot/tests/test_bot_recovery_wal.py` (11 tests : création fichier, append pending/completed/failed, pending_intents single+multi, clear_completed pending vs old, format/truncation/idempotence du clientOrderId).
- **Regression**: aucune — module self-contained, hook dans le runner reporté à une session future (autre agent).

## BUG-B9: pas de gestion des rate limits Binance ni backoff
- **Status**: FIXED (2026-04-26, 11/11 tests `test_bot_binance_client.py` + live testnet OK, used_weight=6 après /fapi/v2/balance)
- **Symptom**: les appels HTTP directs (`requests.post/get/delete` vers `/fapi/v1/algoOrder`) ne lisaient pas les headers `X-MBX-USED-WEIGHT-1m` / `X-MBX-ORDER-COUNT-1m`, n'avaient aucun backoff, et ignoraient HTTP 429/418. Risque de bannissement IP en burst.
- **Root cause**: `trade_manager.py` faisait des `requests.X(url, headers=..., timeout=10)` sans middleware. Aucun parsing rate-limit, aucun retry, aucune protection contre 429/418.
- **Fix**: nouveau module `binance_bot/bot/binance_client.py` avec classe `BinanceClient`. Méthodes : `public_get`, `signed_get`, `signed_post`, `signed_delete`, `is_banned`, `used_weight_1m`, `order_count_1m`, `time_offset_ms`. Comportements :
  - parse `X-MBX-USED-WEIGHT-1m` + `X-MBX-ORDER-COUNT-1m` à chaque réponse,
  - throttle préventif si `used_weight > 0.8 * 2400 = 1920` → sleep jusqu'à la fin de la minute courante,
  - HTTP 429 → respect strict de `Retry-After`,
  - HTTP 418 → log CRITICAL + `_banned_until = now + Retry-After`, refus de tout call jusqu'à expiration,
  - 5xx + connection errors → backoff exponentiel + jitter `min(2^attempt + uniform(0,1), 60)`, 5 tentatives max.
  `TradeManager` instancie automatiquement un `BinanceClient` quand `api_key`/`api_secret` sont présents (paramètre `binance_client` injectable pour tests). `_place_algo_order`, `fetch_open_algo_orders`, `cancel_algo_order` utilisent désormais `self.client.signed_*`.
- **Test**: `binance_bot/tests/test_bot_binance_client.py` (`test_rate_limit_headers_parsed`, `test_throttle_when_used_weight_above_threshold`, `test_backoff_exponential_on_500`, `test_429_respects_retry_after`, `test_418_triggers_ban`, `test_request_refused_when_banned`).
- **Regression**: tests `test_bot_trade_manager.py` mis à jour pour mocker `BinanceClient` au lieu de `requests.*` direct (28/28 verts).

## BUG-B11: pas de synchronisation d'horloge avec Binance
- **Status**: FIXED (2026-04-26, 4/4 tests `test_sync_time*` + `test_signed_request_uses_offset_in_timestamp` + `test_periodic_resync_after_interval` + live testnet : offset mesuré ~80 ms)
- **Symptom**: `params['timestamp'] = int(time.time() * 1000)` envoyé brut. Sur une machine avec horloge dérivée, Binance répondait `-1021 Timestamp for this request is outside of the recvWindow`.
- **Root cause**: aucun appel à `GET /fapi/v1/time` au boot et aucun offset persistant. Le `recvWindow=5000` ne suffit pas si l'horloge locale a > 5s d'écart.
- **Fix**: dans `BinanceClient`, méthode `_sync_time()` appelle GET `/fapi/v1/time`, calcule `offset = serverTime - localTimeMid` (compensation latence RTT/2), stocke `self._time_offset_ms`. Sync au boot puis re-sync auto toutes les 600 s via `_maybe_resync()` appelé avant chaque requête. Au moment de signer : `params['timestamp'] = int(time.time() * 1000) + self._time_offset_ms`. WARNING loggé si `|offset| > 1000` ms. Méthode `time_offset_ms()` exposée pour les tests.
- **Test**: `binance_bot/tests/test_bot_binance_client.py:test_sync_time_computes_offset`, `:test_sync_time_warning_when_offset_exceeds_1000ms`, `:test_signed_request_uses_offset_in_timestamp`, `:test_periodic_resync_after_interval`.
- **Regression**: `auto_sync=False` permet d'instancier sans appel réseau (utilisé par `TradeManager` en mode tests/simulation).

## BUG-ALGO: STOP_MARKET déprécié, endpoint algoOrder requis
- **Status**: FIXED (POST + GET + DELETE validés live testnet) (2026-04-26, 16/16 tests unit + probe live sur testnet. Endpoints REST officiels confirmés : POST `/fapi/v1/algoOrder` (création), GET `/fapi/v1/openAlgoOrders` (liste algo orders ouverts), GET `/fapi/v1/algoOrder` (query single), DELETE `/fapi/v1/algoOrder` (annulation), GET `/fapi/v1/allAlgoOrders` (historique). Le path `openAlgoOrders` (Algo entre 'open' et 'Orders') est la bonne orthographe — toutes les variantes `algoOpenOrders`, `conditional/openOrders`, `algo/futures/openOrders` renvoyaient `-5000 Path invalid`. Méthodes ajoutées dans `TradeManager` : `fetch_open_algo_orders()`, `cancel_algo_order(algo_id)`, `cancel_all_algo_orders()`. `_close_long` / `_close_short` appellent désormais `cancel_all_algo_orders` en plus de `_cancel_pending_orders`. Probe : `binance_bot/scripts_probe_algo_get.py`.)
- **Symptom**: `create_order(type='STOP_MARKET', ...)` retourne erreur Binance `-4120`. Depuis 2025-12-09 Binance a migré les conditional orders vers `/fapi/v1/algoOrder`.
- **Root cause**: ccxt 4.3.98 ne supporte pas le nouvel endpoint algoOrder. `trade_manager._open_long` (l.71-77) et `_open_short` (l.110-116) utilisent l'ancien endpoint.
- **Fix**: appel HMAC manuel direct vers `/fapi/v1/algoOrder` avec `algotype=CONDITIONAL`, `triggerprice=...` (lowercase), signature SHA256. Helpers `_place_algo_order` / `fetch_open_algo_orders` / `cancel_algo_order` / `cancel_all_algo_orders` dans `TradeManager`.
- **Test**: `binance_bot/tests/test_bot_trade_manager.py` (POST: `test_place_algo_order_*`, GET: `test_fetch_open_algo_orders_*`, DELETE: `test_cancel_algo_order_*` et `test_cancel_all_algo_orders_*`, intégration close: `test_close_long_calls_cancel_all_algo_orders`, `test_close_short_calls_cancel_all_algo_orders`).
- **Regression**: aucune — 73/73 tests pytest verts.
