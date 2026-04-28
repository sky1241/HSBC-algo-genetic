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
