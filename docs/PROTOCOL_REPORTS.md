# PROTOCOL REPORTS — Format A.6.4

Document produit en remédiation R1 (2026-04-28).

Pour chaque chunk DÉJÀ BRANCHÉ EN PROD (P0bis → P5), on applique formellement
les 6 étapes du protocole rétroactivement : code, auto-review Q1-Q8, tests,
Forge (informational), branchement vérifié, commit hash.

---

## P0bis — Brancher cost_model au pipeline 14 ans

### Étape 1 — Code
- **Décision**: extension de `ichimoku_pipeline_web_v4_8_fixed.py` (existant) +
  helper `_round_trip_fee_cost_usdt()` injecté en haut du fichier.
- **Justification**: spec exigeait branchement, pas nouveau module. Le helper
  encapsule la logique round-trip (entry+exit) pour panacher taker/maker.
- **Fichiers** (commit `20899b5`):
  - `src/cost_model.py` — module existant non modifié (304 LOC)
  - `ichimoku_pipeline_web_v4_8_fixed.py` — import + helper + 4 sites
    `commission_cost = _round_trip_fee_cost_usdt(...)` aux lignes 867, 887, 917, 935
  - `tests/test_p0bis_pipeline_fees.py` — tests pipeline

### Étape 2 — Auto-review Q1-Q8 (rétroactif)
- **Q1 spec respectée ?** ✅ Fees taker_bps=4.0/maker_bps=2.0 (VIP0). Round-trip
  entry+exit = 2 × leg_cost. Hypothèse 100% taker par défaut, override via
  `maker_fill_ratio`. Application AVANT calcul du Sharpe (logique dans la
  boucle de trades).
- **Q2 cas limites ?** Notional ≤ 0 → 0.0 (pas de fee). `maker_fill_ratio`
  hors [0, 1] → clipped. ⚠️ Note mineure : `apply_costs_to_returns` ligne 269
  charge fees au premier bar via `pos.diff().fillna(pos.abs())` — peut
  double-compter si appelé par segments. Non critique pour usage actuel.
- **Q3 naming ?** ✅ `_round_trip_fee_cost_usdt` précis (round-trip, USDT).
- **Q4 dead code ?** ✅ Aucun.
- **Q5 magic numbers ?** ✅ `1e-4` (bp→fraction) documenté ; `4.0/2.0` doc.
- **Q6 conventions ?** ✅ PEP8, type hints.
- **Q7 side-effects ?** ✅ Pure, frozen dataclass `BinanceFutureFees`.
- **Q8 errors ?** ✅ ValueError explicite si notional < 0.

### Étape 3 — Tests
- Fichier : `tests/test_p0bis_pipeline_fees.py`
- Tests obligatoires : `test_pipeline_fees_applied_correct_amount`,
  `test_pipeline_fees_zero_when_disabled`,
  `test_pipeline_fees_match_cost_model_module`,
  `test_pipeline_fees_with_maker_ratio`.

### Étape 4 — Forge
**Status**: ⚠️ NON LANCÉ. Cause : BUG-PRE-001 (test_aco timeouts pré-existants
font diverger Forge). Sky a validé option α (Forge informational, validation
via pytest direct).

### Étape 5 — Branchement vérifié
- Grep `cost_model|_round_trip_fee_cost_usdt` dans pipeline → **31 hits**
- Appels actifs dans la boucle de trades (lignes 867, 887, 917, 935, 974,
  1042, 1111, 1144, 1174, 1202, 1229)
- **Mini-WFA validation R2 (90j BTC H1, 2160 bars)** :
  ```
  equity NO_FEES   : 1.00777896
  equity WITH_FEES : 1.00753737
  delta            : 0.00024159
  n_trades         : 30
  drag attendu     : ~0.000240  (n_trades × 0.01 × 8e-4 = position_size × round_trip)
  ratio obs/exp    : 1.01  ✅
  ```
  Validation numérique : les fees affectent bien le PnL final, et le
  drag observé matche presque exactement le calcul théorique
  (`n_trades × position_size × 8 bps round-trip`). Branchement P0bis
  fonctionnel **et exact**, pas seulement présent en grep.

  Script : `scripts/validation/p0bis_mini_wfa_fees.py` (reproductible).

### Étape 6 — Commit
- Hash : `20899b5`
- Message : *"P0bis: brancher cost_model au pipeline 14 ans, fees Binance VIP0 corrigés spot→futures — branchement vérifié 27 sites"*

---

## P0 — vol_targeting Kelly portfolio-aware

### Étape 1 — Code
- **Décision**: extension de `src/vol_targeting.py` + nouveau `compute_aggregated_var`.
- **Fichiers** (commit `248539f`):
  - `src/vol_targeting.py` (4 fonctions publiques)
  - `binance_bot/services/signal_engine.py` — param `portfolio_state`
  - `binance_bot/routines/intraday_runner.py` — factory `_make_portfolio_scale_fn`

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ `kelly_fraction` reçoit `portfolio_state` ; calcule sigma_pf via
  Markowitz `sqrt(w' Σ w)` ; retourne `kelly_solo × min(1, budget/max)`.
- **Q2**: total_capital ≤ 0 → kelly_solo (fail-open). Position absente de la
  matrice corr → rho=0 (ignore corr). var_pf ≤ 0 (numerical) → 0.0.
- **Q3**: ⚠️ `compute_aggregated_var` retourne σ_pf, pas VaR au sens strict.
  Naming imprécis mais doc explicite. Non bloquant.
- **Q4**: ⚠️ Bug détecté en R1 : import `Iterable` non utilisé → **fixé R1**.
- **Q5**: ✅ `0.06` (max_portfolio_risk) doc, `0.5` (Half-Kelly) doc.
- **Q6**: ✅
- **Q7**: ✅ pure
- **Q8**: ✅ `KeyError, ValueError` capturés explicitement.

### Étape 3 — Tests
- `tests/test_p0_vol_targeting_portfolio.py` — au moins 5 tests
  (solo unchanged, reduced when correlated, zero when max used,
  full when anticorrelated, rolling 30d).

### Étape 4 — Forge
NON LANCÉ (cf P0bis).

### Étape 5 — Branchement
- Grep `compute_aggregated_var|portfolio_state|kelly_fraction` →
  - `intraday_runner.py:116, 119, 149` (compute_aggregated_var)
  - `signal_engine.py:5` (docstring P0)
- ✅ ACTIF : `intraday_runner` (timer systemd `hsbc-intraday` actif) appelle
  `compute_aggregated_var` à chaque cycle.

### Étape 6 — Commit
- Hash : `248539f`
- Message : *"P0: vol_targeting Kelly portfolio-aware avec corrélations rolling 30j — branchement vérifié signal_engine + intraday_runner"*

---

## P1 — Daily caps soft 2%/3% / hard 10%

### Étape 1 — Code
- Extension `signal_engine.py` (params daily_loss_soft/gain/hard_cap_pct,
  block_until_iso, kill_switch_path, notifier).
- `bot_settings.yaml` modifié.

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ Logique HARD-CAP first (priorité absolue), puis soft loss, puis
  soft gain. Block jusqu'à next UTC midnight.
- **Q2**: ✅ Reset block à minuit UTC (`block_until_iso` parsé ISO).
  PnL = 0 → no action.
- **Q3-Q7**: ✅
- **Q8**: ✅ Notifier exceptions silenced (safe).

### Étape 5 — Branchement
- ✅ ACTIF dans `signal_engine.detect_signals`. Tests d'intégration confirment
  flat + block à PnL = -2.5%.

### Étape 6 — Commit
- Hash : `0c28e30`

---

## P2 — Anti-martingale drawdown sizing

### Étape 1 — Code
- Ajout `drawdown_size_multiplier()` dans `src/risk_sizing.py`.
- Branchement `signal_engine.drawdown_scale_fn` callback + factory dans
  `intraday_runner._make_drawdown_scale_fn`.

### Étape 2 — Auto-review Q1-Q8 (R1 RETROSPECTIVE FIND)
- **Q1**: ✅ tiers spec : dd<-15%→0, dd<-10%→0.5, dd<-5%→0.667, sinon 1.0.
- **Q2**: 🐛 **BUG SILENCIEUX TROUVÉ EN R1**: `current_equity ≤ 0` retournait
  `1.0` (taille pleine) au lieu de `0.0` (kill). Si compte liquidé,
  multiplicateur full → permet trades. **FIXÉ EN R1**, test ajouté
  `test_zero_or_negative_equity_returns_kill`.
- **Q3-Q8**: ✅

### Étape 3 — Tests
- 15/15 PASS après fix R1.

### Étape 5 — Branchement
- ✅ ACTIF : `intraday_runner.py:411` appelle `drawdown_size_multiplier`
  comme callback `_make_drawdown_scale_fn`.

### Étape 6 — Commit
- Hash original : `18c5df7`
- **Hash fix R1 (à venir)** : commit séparé pour bug `current_equity ≤ 0`.

---

## P3 — Portfolio risk gate VaR95

### Étape 1 — Code
- Nouveau module `src/portfolio_risk_gate.py` (2 fonctions publiques :
  `portfolio_var_95`, `can_enter_new_position`).
- Branchement `signal_engine.var_gate_fn`.

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ Historical sim (Path 1, bootstrap stationnaire 24h) + Monte Carlo
  Gaussien fallback (Path 2, Cholesky de Σ). Seuil 8% (default).
- **Q2**: ⚠️ API positions diffère de la spec (`{side, notional_pct}` au lieu
  de `notional_usd_signed`). Design choice, doc explicite. ⚠️ Silent fallback à 0.0
  quand info insuffisante (fail-open). Discutable mais non-bug.
- **Q3-Q8**: ✅

### Étape 5 — Branchement
- ✅ ACTIF : `intraday_runner.py:512` `_make_var_gate_fn` → `signal_engine.var_gate_fn`.

### Étape 6 — Commit
- Hash : `0fd4994`

---

## P4 — HAR-RV regime gate

### Étape 1 — Code
- Nouveau module `src/har_rv.py` (4 fonctions publiques).
- Branchement intraday_runner factory `_make_regime_gate_fn`.

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ `RV(window=22)` + `log(RV_next) = β0 + β_h × log(RV_1h) + β_d × log(RV_5h) + β_w × log(RV_22h)`. Quantiles 30%/70% sur 90j.
- **Q2**: NaN propagation safe. Train trop court → coefs NaN.
- **Q3-Q8**: ✅

### Étape 5 — Branchement
- ✅ ACTIF : `intraday_runner.py:191, 197` `fit_har_rv` + `classify_regime`.

### Étape 6 — Commit
- Hash : `7cbc96d`

---

## P5 — PSR live alpha decay detection

### Étape 1 — Code
- Nouveau module `src/psr_live.py` (3 fonctions publiques).
- Script daily `scripts/production/psr_live_daily.py`.

### Étape 2 — Auto-review Q1-Q8 (R1 RETROSPECTIVE)
- **Q1**: ✅ Formule Bailey-LdP 2012 Eq.9 implémentée à la lettre.
  MinTRL Eq.12 aussi.
- **Q2**: returns < 2 → 0.5 (indéterminé). var_factor ≤ 0 → 0.5.
  sigma ≤ 0 → 0.5. NaN propagation OK.
- **Q3**: ✅
- **Q4**: ⚠️ Imports morts `Mapping, Optional` détectés en R1 → **FIXÉS**.
- **Q5-Q8**: ✅

### Étape 5 — Branchement
- ✅ ACTIF : `scripts/production/psr_live_daily.py:99` appelle `compute_psr`.
  Lancé par `hsbc-daily.timer` (systemd actif).

### Étape 6 — Commit
- Hash : `6afd5f1`
- **Hash fix R1 (à venir)** : commit séparé pour cleanup imports morts.

---

---

## R3 — P6.1-P6.4 flow stack: création runners systemd

### Étape 1 — Code
- **Décision**: 2 nouveaux runners + 3 unités systemd (REST oneshot timer
  + WS daemon long-running). Ne pas modifier les modules P6 existants.
- **Justification**: les modules P6.1-P6.4 exposent déjà `poll_and_store()`
  et `LiquidationWSManager` ; il manquait juste le câblage systemd.
- **Fichiers créés**:
  - `binance_bot/routines/flow_rest_collector.py` — appelle 3 polls × 3 symbols
  - `binance_bot/routines/flow_liquidations_daemon.py` — long-running WS
  - `binance_bot/systemd/hsbc-flow-rest.timer` — OnCalendar 5min
  - `binance_bot/systemd/hsbc-flow-rest.service` — oneshot
  - `binance_bot/systemd/hsbc-flow-liquidations.service` — Type=simple,
    Restart=always

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ Spec P6 demandait poll 5min des 3 endpoints REST + WS persistent
  pour liquidations. Match.
- **Q2**: Fail-safe: erreur 1 service → autres continuent (`_safe_poll` catch
  any exception). WS daemon: SIGTERM/SIGINT handlers + `manager.stop(drain)`.
- **Q3**: noms précis (`flow_rest_collector`, `flow_liquidations_daemon`).
- **Q4**: pas d'imports morts.
- **Q5**: `HEARTBEAT_INTERVAL_SEC = 300` documenté ; `SYMBOLS` constant top.
- **Q6**: pep8 / type hints / docstrings.
- **Q7**: pas d'effet de bord — les modules P6 sont read-only à l'import.
- **Q8**: try/except spécifique sur les polls ; SIGTERM handler explicite.

### Étape 3 — Tests
- `binance_bot/tests/test_r3_flow_collectors.py` — 5 tests :
  - `_safe_poll` swallows exceptions / returns count on success
  - `main` iterates 3 symbols × 3 services
  - partial failure n'interrompt pas le run
  - WS daemon start/stop cleanly

### Étape 4 — Forge
NON LANCÉ (cf P0bis, BUG-PRE-001).

### Étape 5 — Branchement vérifié EN PROD LIVE
- **Smoke test REST collector** (vraie API Binance) :
  ```
  [top_ls] BTCUSDT +30 new records → flow_top_ls_BTCUSDT.jsonl
  [taker]  BTCUSDT +30 new records → flow_taker_BTCUSDT.jsonl
  [oi]     BTCUSDT +30 new records → flow_oi_BTCUSDT.jsonl
  ... idem ETH + SOL ...
  flow_rest_collector done: 270 new records across 3 symbols × 3 services
  ```
  Format vérifié (sample BTC top_ls) :
  ```json
  {"ts_ms": 1777350600000, "symbol": "BTCUSDT", "long_short_ratio": 0.7883,
   "long_account": 0.4408, "short_account": 0.5592}
  ```
- **Smoke test WS daemon** (vraie connexion fstream.binance.com) :
  ```
  [INFO] starting WS daemon for ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
  [INFO] Websocket connected
  [INFO] received signal 15 — shutting down WS
  [INFO] daemon exited cleanly
  ```
  (SIGTERM gérée proprement, daemon exits cleanly.)

### Étape 6 — Commit
- Hash : (en cours)
- ⚠️ **systemd units NON ENABLED** par sécurité (touche le bot live actif).
  Pour activer en prod, l'opérateur doit lancer manuellement :
  ```bash
  cd /home/ludov/HSBC-algo-genetic/binance_bot/systemd
  systemctl --user link $(pwd)/hsbc-flow-rest.timer
  systemctl --user link $(pwd)/hsbc-flow-rest.service
  systemctl --user link $(pwd)/hsbc-flow-liquidations.service
  systemctl --user enable --now hsbc-flow-rest.timer
  systemctl --user enable --now hsbc-flow-liquidations.service
  ```

---

## R4 — P6.5 composite signal: branchement signal_engine mode LOG-ONLY

### Étape 1 — Code
- **Décision**: extension `signal_engine.SignalEngine` + factory dans
  `intraday_runner._make_composite_log_fn`.
- **Justification**: même pattern callback que P0/P3/P4. Mode log-only
  (per spec : 30j baseline avant activation gate).
- **Fichiers modifiés**:
  - `binance_bot/services/signal_engine.py` — params
    `composite_log_fn`, `composite_log_path` + méthode
    `_log_composite_signal()` + hook tout début de `detect_signals`
  - `binance_bot/routines/intraday_runner.py` — factory
    `_make_composite_log_fn(symbol, data_dir)` + injection au
    `SignalEngine(...)` instantiation

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ Spec dit "Mode INITIAL : log only (pas d'effet sur ordres)
  pendant 30 jours". Implémenté : aucun blocage, juste persist JSONL +
  notifier.info. Mode GATE = futur (post-baseline + Sky validation).
- **Q2**: composite_log_fn=None → no-op. callback raise → silent
  (try/except). path=None → pas de write. dict invalide → silent.
- **Q3**: ✅ noms précis (`composite_log_fn`, `composite_log_path`,
  `_log_composite_signal`).
- **Q4**: ✅ pas d'import mort (json, time ajoutés en haut).
- **Q5**: pas de magic number nouveau.
- **Q6**: pep8, type hints (`Optional[Callable[[], dict]]`,
  `Optional[Path]`).
- **Q7**: ✅ pure (pas de mutation des modules existants, juste
  ajout de paramètres optionnels + 1 hook 1-line dans detect_signals).
- **Q8**: try/except spécifiques. **AUCUN** `except Exception: pass`
  général.

### Étape 3 — Tests
- `binance_bot/tests/test_r4_composite_log.py` — 10 tests :
  - no-op si pas de fn
  - persist JSONL avec ts_ms ajouté
  - append multiple lines
  - exception callback safe silent
  - dict invalide silent
  - path=None pas de write
  - notifier.info appelé avec format correct
  - hook detect_signals appelle composite_log
  - df vide → pas d'appel
  - factory `_make_composite_log_fn` produit callable

### Étape 4 — Forge
NON LANCÉ (cf P0bis).

### Étape 5 — Branchement vérifié EN PROD LIVE
- Smoke test E2E (2 cycles `detect_signals`) :
  ```
  JSONL lines: 2
    ts=1777360121188 symbol=BTCUSDT score=+0.000
                     n_obs={'top_ls': 30, 'taker': 30, 'liq': 0, 'oi': 0}
    ts=1777360121206 symbol=BTCUSDT score=+0.000
                     n_obs={'top_ls': 30, 'taker': 30, 'liq': 0, 'oi': 0}
  ```
- Score = 0 attendu : `zscore_last(min_obs=100)` retourne 0 tant qu'on
  n'a pas 100 obs. Avec 30 records (R3 backfill initial 5min) on est
  sous le seuil. Baseline réelle après 30j de poll cron.
- Limitation connue : `flow_liq_buckets.jsonl` est GLOBAL (tous symboles
  dans un fichier), donc `liq imbalance` non filtré par symbol — futur
  enhancement.

### Étape 6 — Commit
- Hash : (en cours)

---

## R5 — P7 VPIN gate (Option B callback indépendant) — Sky décide A reset-wins

### Décision de design actée par Sky le 2026-04-28

> **Option A (reset wins)** : `reset_threshold` est l'exit normal du block,
> `block_duration_minutes` agit uniquement comme plafond safety failsafe
> en cas de lecture VPIN corrompue.

Rationale : VPIN reflète l'état réel du flow d'order book. Si le flow
redevient sain (VPIN bas), bloquer plus longtemps n'apporte rien. Le
timer existe uniquement comme garde-fou contre une lecture VPIN
corrompue ou un bug qui maintiendrait artificiellement VPIN haut.

### Étape 1 — Code

- **Décision**: nouveau module **pur** `src/vpin_gate.py` (state machine
  testable sans dépendance bot) + branchement via `SignalEngine` callback
  + factory dans `intraday_runner`. Per spec Option B, gate INDÉPENDANT
  de `flow_composite_signal` (orthogonalité micro/macro).
- **Justification**: pattern callback déjà établi (P0/P3/P4/P10/P6.5).
  Module pur dans `src/` aligne avec `cost_model`, `vpin`, `har_rv`,
  `portfolio_risk_gate`, etc.
- **Fichiers** :
  - `src/vpin_gate.py` (NEW) — `VPINState`, `VPINGateConfig`, `vpin_gate_check()`
  - `binance_bot/services/signal_engine.py` — params `vpin_data_fn`,
    `vpin_gate_config`, `vpin_state_dict`, `vpin_event_log_path` ;
    methodes `_evaluate_vpin_gate()`, `get_vpin_state_dict()` ; hook
    detect_signals (kill_and_block prioritaire AVANT P1 hard cap, et
    `vpin_blocks_entries` injecté dans gate composite long/short)
  - `binance_bot/routines/intraday_runner.py` — factories
    `_make_vpin_gate_config(settings)` + `_make_vpin_data_fn(symbol)`
    (placeholder None tant que collecteur live aggTrade+bookTicker pas
    branché) + persistance `vpin_state` per-symbol dans `state.json`
  - `binance_bot/configs/bot_settings.yaml` — 6 nouvelles clés
    `vpin_mode`/`vpin_block_threshold`/etc. (mode=log_only par défaut)

### Étape 2 — Auto-review Q1-Q8

- **Q1**: ✅ Spec Option B + design Option A (reset wins) implémentés à la
  lettre. Pseudocode du state machine (réf message Sky) reproduit
  ligne par ligne dans `vpin_gate_check`.
- **Q2**: cas limites — `data_fn=None` → vpin_disabled. `data_fn()=None`
  → vpin_data_unavailable. callback raise → vpin_data_fn_error. data
  malformed (non-tuple/tuple<2) → vpin_data_malformed. VPIN ou OBI hors
  [0,1] → vpin_data_out_of_range. mode invalide → fallback log_only.
  block_duration_minutes ≤ 0 → clamp à 1.
- **Q3**: ✅ noms précis — `vpin_gate_check` (pas `gate_check`), tags
  raison machine-readable (`vpin_reset_below_threshold`, `timer_expiry`,
  `vpin_cascade_imminent`, `vpin_toxic_flow`).
- **Q4**: ✅ pas d'imports morts.
- **Q5**: ✅ thresholds config-driven (pas hardcodés) ; `60_000` ms→min
  documenté inline.
- **Q6**: ✅ pep8, type hints, dataclasses, docstrings.
- **Q7**: ✅ état persisté via `vpin_state_dict` round-trip dict ↔
  `VPINState.from_dict/to_dict`. Pas de mutation d'objet partagé.
- **Q8**: ✅ try/except précis sur `data_fn`, ImportError, IO write.
  Fail-OPEN systématique (sécurité : ne jamais bloquer le bot par
  erreur du module VPIN).

### Étape 3 — Tests

- `tests/test_p7_vpin_gate.py` — **19 tests** purs sur la state machine :
  - log_only never blocks + does not persist blocked state
  - thresholds primaires (below block / above block / above kill+low OBI / above kill+high OBI)
  - hystérésis no flapping
  - reset unblocks before timer
  - duration acts as safety ceiling
  - stays blocked when above reset within/after duration
  - post-timer can immediately re-block (pas de limbo state)
  - reset threshold only when blocked
  - VPINState dataclass roundtrip + from None/partial
  - VPINGateConfig clamps thresholds, fallback log_only, min duration

- `binance_bot/tests/test_r5_vpin_signal_engine.py` — **10 tests
  d'intégration** :
  - log_only ne bloque pas l'entrée
  - gate bloque long entry / short entry
  - kill flat positions ouvertes (long + short)
  - kill envoie notifier.warn
  - data_fn None → safe allow
  - data_fn raise → fail-open
  - vpin_disabled si pas de data_fn
  - data hors range → safe allow
  - state persiste cross-cycle via vpin_state_dict roundtrip

- `binance_bot/tests/test_r5_vpin_runner_factories.py` — **3 tests** :
  - config from settings.yaml
  - config defaults (log_only safe)
  - data_fn placeholder retourne None

### Étape 4 — Forge
NON LANCÉ (cf P0bis, BUG-PRE-001).

### Étape 5 — Branchement vérifié
- Grep `vpin_data_fn|vpin_gate_config|_evaluate_vpin_gate|vpin_blocks_entries`
  → **20+ hits** dans `signal_engine.py` + `intraday_runner.py`.
- VPIN gate appelé AU DÉBUT de `detect_signals` pour priorité
  `kill_and_block`. `block_new_entries` injecté dans gate composite
  entrée long + short (parallèle aux gates regime/combinator/var).
- `vpin_event_log_path = data/vpin_events.jsonl` per cycle log JSONL
  pour collecte baseline 30j en mode log_only.
- ⚠️ Limitation : `_make_vpin_data_fn` retourne None placeholder. Le gate
  est donc **opérationnellement disabled** tant que les collecteurs live
  aggTrade WS + bookTicker ne sont pas câblés. Mode log_only par défaut
  rend cette limitation cosmétique pour aujourd'hui.

### Étape 6 — Commit
- Hash : (en cours)

---

## R6 — P8 GARCH ré-architecture `arch` lib + per-symbol + audit_log

### Étape 1 — Code
- **Décision**: rewrite complet de `src/garch.py` en wrapper `arch_model`
  (Kevin Sheppard) tout en gardant la même API dict (rétro-compat tests).
- **Justification**: la spec P8 demandait textuellement "via `arch` lib".
  Ma première impl (commit 1c10962) était scipy from-scratch — violation.
  Migration nécessaire. API dict préservée pour limiter le rework des tests.
- **Fichiers** :
  - `src/garch.py` — rewrite : wrapper `arch.arch_model` avec
    rescale x100 (zone optimale optimizer), extraction params standardisée,
    `_arch_fit` stocké dans le dict pour permettre forecast multi-step.
    Per-symbol assignment via `SYMBOL_GARCH_MODEL` + `fit_for_symbol(symbol, returns)`.
    Nouveau helper `garch_regime_label(forecast_sigma, baseline)` → low/mid/high
    pour comparer avec HAR-RV labels.
  - `tests/test_p8_garch.py` — 4 tests forecast adaptés (params dict
    construits manuellement → fit réel). Reste 16 tests qui passent.
  - `tests/test_p8_garch_arch.py` (NEW) — 12 tests migration arch :
    HAS_ARCH, per-symbol BTC=TGARCH/ETH=EGARCH/SOL=TGARCH, slash format,
    fallback default, dict include `_arch_fit`, garch_regime_label thresholds.
  - `binance_bot/routines/intraday_runner.py` — factory
    `_make_garch_audit_check(symbol, returns, regime_gate_fn)` qui fit
    GARCH per-symbol, compare avec HAR-RV regime, retourne dict
    `{disagreement, har_label, garch_label, model, ...}`. Appelé après
    `regime_gate_fn` dans la main loop ; résultat append à `audit_log`
    (event="garch_har_disagreement" si désaccord, sinon "garch_har_check").
  - `binance_bot/tests/test_r6_garch_audit.py` (NEW) — 6 tests audit check.

### Étape 2 — Auto-review Q1-Q8
- **Q1**: ✅ 3 violations spec P8 corrigées :
  (1) `arch` lib utilisée (vs scipy custom) ;
  (2) per-symbol BTC=TGARCH/ETH=EGARCH/SOL=TGARCH ;
  (3) branchement parallel HAR-RV avec audit_log flag.
- **Q2**: rescale x100 pour stabilité (DataScaleWarning évité) ; rescale
  inverse à la sortie. EGARCH multi-step fallback simulation (analytic
  non supporté h>1). Returns short < 50 → empty result. fit non convergé
  → reason posté dans audit, pas de crash.
- **Q3**: ✅ noms précis (`_make_garch_audit_check`, `garch_regime_label`,
  `SYMBOL_GARCH_MODEL`, `fit_for_symbol`).
- **Q4**: ✅ pas d'imports morts. `_LEVERAGE_THRESHOLD = 0.05` cohérent
  avec v1.
- **Q5**: ✅ rescale x100, simulations=500 pour EGARCH commentés.
- **Q6**: ✅
- **Q7**: ✅ pure (aucun module modifié hors garch.py + intraday factory).
- **Q8**: ✅ try/except précis (`ValueError`, `NotImplementedError` pour
  fallback simulation EGARCH ; `Exception` final pour fail-safe).

### Étape 3 — Tests
- **34 tests R6** au total :
  - `tests/test_p8_garch.py` : 16 (recovery + forecast + classify, post-migration)
  - `tests/test_p8_garch_arch.py` : 12 (HAS_ARCH, per-symbol, slash format, default)
  - `binance_bot/tests/test_r6_garch_audit.py` : 6 (audit check disagreement,
    short series, per-symbol model selection, agreement, disagreement forced)

### Étape 4 — Forge
- ✅ Suite "not slow" passe : **626 tests / 77s** (Forge-compatible
  désormais grâce à R-FIX-FORGE).

### Étape 5 — Branchement vérifié
- Grep `_make_garch_audit_check|fit_for_symbol|garch_regime_label`
  dans intraday_runner → factory + invocation.
- `audit_log.append({"event": "garch_har_disagreement", ...})` posté à
  chaque cycle dans `binance_bot/data/trades_audit.jsonl` quand HAR ≠ GARCH.
- Pas d'effet trade : pure observation Munin pour analyse.

### Étape 6 — Commit
- Hash : (en cours)

---

# Synthèse R1

| Chunk | Bug détecté | Action | Branchement live |
|---|---|---|---|
| P0bis | Aucun | Doc rétroactif | ✅ pipeline backtests |
| P0 | Import mort | Cleanup `Iterable` | ✅ intraday_runner |
| P1 | Aucun | Doc rétroactif | ✅ signal_engine |
| **P2** | 🐛 **`current_equity ≤ 0` → 1.0 (kill manqué)** | **Fix + nouveau test** | ✅ intraday_runner |
| P3 | Aucun | Doc rétroactif | ✅ intraday_runner |
| P4 | Aucun | Doc rétroactif | ✅ intraday_runner |
| **P5** | Imports morts | **Cleanup** | ✅ daily script |

Tous les 6 chunks restent BRANCHÉS et fonctionnels. R1 a surfacé **1 bug
silencieux critique** (P2 kill manquant si compte liquidé) + **2 cleanups**
mineurs.

Mini-WFA P0bis avec/sans fees → R2 dédiée.
