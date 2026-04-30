# REPORT_CHECK — 2026-04-30

> Audit de contrôle suite au résumé de session transmis par Claude Opus web.
> Source : audit READ-ONLY sur `/home/ludov/HSBC-algo-genetic` au commit
> `28d0e24` (main, propre côté code).
> Auteur : Claude Code local. Date : 2026-04-30T13:35:02Z (J+2 du soak).

---

## A. ÉTAT GLOBAL

**`PARTIELLEMENT PROPRE`** — la stack défensive et exécutive (P0–P11) est
solide et branchée. Une exception : **PSR live (CRIT-7) n'est PAS branché**
au runtime, malgré un module complet et un timer `hsbc-daily.timer` actif.
La thèse multi-TF n'est pas implémentée mais c'est attendu (cf section D).

| Domaine | État |
|---|---|
| Tests unitaires | ✅ 704/704 PASS (not slow) en 1m51 — tests slow non vérifiés (30 deselected) |
| Branche / commit | ✅ main, `28d0e24`, propre côté code |
| Tracking files | ✅ Tous présents (BUGS, LIMITATIONS, MISSION_REWFA, NEXT_STEPS_30D, AUDIT v2) |
| Configuration yaml | ✅ Tous les paramètres P1/P3/P7/P11 présents et conformes |
| Daemons systemd | ✅ hsbc-vpin-live + hsbc-dashboard + 6 timers actifs |
| Branchements P0–P6, P11 | ✅ 7/8 modules branchés et utilisés |
| Branchement P5 (PSR live) | ❌ Module orphelin, jamais appelé au runtime |
| WS-001 | ✅ Daemon liquidations supprimé du système (équivalent disable) |
| Thèse trading multi-TF | ❌ Non implémentée (mono H2, prévu post-J+30) |

---

## B. POINTS PROPRES

### B.1 — Section 6.1 (état du repo)

- ✅ Branche `main` propre côté code (modifs uniquement sur `data/*.jsonl`
  appendés par le bot — comportement attendu)
- ✅ HEAD = `28d0e24` "fix L-008: VPIN import silent failure"
- ✅ Pytest : `704 passed, 30 deselected in 111.83s`
- ⚠️ Pytest `slow` non lancé dans cet audit (estimé : 30 tests, à vérifier
  séparément si jamais douteux)

### B.2 — Section 6.2 (fichiers de tracking)

| Fichier | Présent | Note |
|---|---|---|
| `BUGS.md` | ✅ | 16 entrées dont WS-001 explicitement |
| `LIMITATIONS_ACTEES.md` | ✅ | L-001 à L-007 (sauf L-006), L-005 marque WS-001 |
| `MISSION_REWFA.md` | ✅ | Couvre BUG-C/D/E hors scope P0bis-P12 |
| `NEXT_STEPS_30D.md` | ✅ | 9.7 KB, scoping post-J+30 |
| `AUDIT_FINAL_2026-04-28_v2.md` | ✅ | Verdict READY FOR TESTNET |
| `data/soak_start.txt` | ✅ | Timestamp `2026-04-28T12:13:35Z` (J+2 aujourd'hui) |

### B.3 — Section 6.3 (configuration `bot_settings.yaml`)

Toutes les clés requises présentes et conformes :

```yaml
timeframe: "2h"
daily_loss_soft_cap_pct: 0.02
daily_loss_hard_cap_pct: 0.10
daily_gain_soft_cap_pct: 0.03
max_portfolio_risk: 0.06
portfolio_var_threshold: 0.08
vpin_block_threshold: 0.70
vpin_kill_threshold: 0.85
vpin_reset_threshold: 0.50
vpin_block_duration_minutes: 15
vpin_kill_obi_threshold: 0.30
vpin_mode: "log_only"
vpin_collector_enabled: true       # ← spec disait false ; activé pour P7-bis
funding_close_threshold_long_bps: 5.0
funding_close_threshold_short_bps: 5.0
funding_close_advance_minutes: 5
trade_mode: "simulation"
```

Note : `vpin_collector_enabled: true` au lieu de `false` (spec initiale).
Activation cohérente avec mission P7-bis mergée — pas une régression.

### B.4 — Section 6.4 (branchements CRIT-1 à CRIT-8)

| CRIT | Module | État | Preuve |
|---|---|---|---|
| **CRIT-1** | `cost_model` | ✅ OK | 27 appels `_round_trip_fee_cost_usdt` dans `ichimoku_pipeline_web_v4_8_fixed.py` lignes 49–1932 |
| **CRIT-2** | Kelly portfolio-aware | ✅ OK | `vol_targeting.py:108-173` (param `portfolio_state`) → `intraday_runner.py:104-163` factory → `signal_engine.py:613-619` usage |
| **CRIT-3** | Daily caps avant signal | ✅ OK | `signal_engine.py:550` hard cap → `:595-606` soft caps avant entrées (early return) |
| **CRIT-4** | Anti-martingale | ✅ OK | `risk_sizing.py:143-182` spec exacte, validé runtime aux 5 bornes (cf B.5) |
| **CRIT-5** | Portfolio VaR95 gate | ✅ OK | `portfolio_risk_gate.py:191-240` `can_enter_new_position` → `signal_engine.py:472-485` `_var_gate_blocks` |
| **CRIT-6** | HAR-RV regime gate | ✅ OK | `har_rv.py:161-193` `classify_regime` → `signal_engine.py:301-310` `_regime_gate_blocks` |
| **CRIT-7** | PSR live cron daily | ❌ **MANQUANT** | Module `src/psr_live.py` complet, timer `hsbc-daily.timer` actif, mais ZÉRO appel runtime — voir section C |
| **CRIT-8** | VPIN log_only effectif | ✅ OK | `vpin_gate.py:100-135` retourne toujours `("allow", "log_only_mode", state)` quand mode=="log_only" |

### B.5 — Validation runtime CRIT-4 (anti-martingale)

```
dd=-15%   (85.0/100.0)  → 0.5    ✓ (attendu: 0.5)
dd=-15.01% (84.99)      → 0.0    ✓ (attendu: 0.0)
dd=-10%   (90.0)        → 0.667  ✓ (attendu: 0.667)
dd=-5%    (95.0)        → 1.0    ✓ (attendu: 1.0)
dd=0      (100.0)       → 1.0    ✓ (attendu: 1.0)
```

`<` strict respecté (dd=-15% pile retourne 0.5, pas 0.0) — conforme spec.

### B.6 — Section 6.5 (WS-001)

- ✅ Daemon `hsbc-flow-liquidations.service` **n'existe pas** dans le système
  (pas seulement `disabled` : carrément absent du systemd user). Statut
  équivalent voire mieux que demandé.
- ✅ `L-005` documente la décision dans `LIMITATIONS_ACTEES.md`
- ⚠️ Le test composite `compute_score(top_ls=0.5, taker=0.5, liq=None, oi=0.5)`
  proposé dans le résumé **ne match pas** la signature réelle. La fonction
  `compute_composite_score` prend des **chemins de fichiers JSONL**, pas des
  scalaires. Test à reformuler ; impossible de valider robustesse `liq=None`
  via cet appel direct. À voir si la robustesse est testée en pytest dédié.

### B.7 — Section 7.5 (cap de gain journalier)

- ✅ `daily_gain_soft_cap_pct: 0.03` présent
- ✅ Évalué dans `signal_engine.py:259-267` (`_soft_gain_cap_triggered`) puis
  `:595-600` avant nouvelles entrées
- ✅ Trigger flat positions + block jusqu'à minuit UTC

### B.8 — Section 7.6 (cycle halving)

- ✅ Calculé : `intraday_runner.py:358-361`
  `BTC_HALVING_2024 = _dt(2024, 4, 19); days_since_halving = ...`
- ✅ Stocké dans meta_context : `intraday_runner.py:398-399`
- ✅ Validé comme champ obligatoire dans `bot/trade_meta.py:60`
- ⚠️ Utilisé UNIQUEMENT pour audit/contexte post-trade — **pas comme input
  de décision de bias** (cf section C, point C.4)

### B.9 — État de la mission P7-bis + R7-bis

- ✅ Branche `feat/p7bis-r7bis-status` mergée sur main (`f5f22d7`)
- ✅ 8 commits F-p7bis-1 à F-r7bis + F-003 + F-004 + F-r7bis dans l'historique
- ✅ Post-fixes L-007 (`4d526ad`) + L-008 (`28d0e24`) appliqués
- ✅ Daemon `hsbc-vpin-live.service` actif (running)

---

## C. POINTS À FIXER

### C.1 — `[CRITIQUE]` PSR live n'est pas branché (CRIT-7)

**Symptôme** : `binance_bot/data/psr_history.jsonl` contient UNE seule entrée
du 28 avril 05:08 (`{"n_obs": 0, "psr": null, "alert": "insufficient_data"}`),
**0 mises à jour depuis** alors que `hsbc-daily.timer` tourne (dernière
exécution : 2026-04-30 01:55, suivante : 2026-05-01 01:55).

**Vérification grep** :
```
grep -rn "psr\|PSR" binance_bot/ --include="*.py"  →  0 hit
grep -rn "compute_psr\|psr_history\|psr_live" binance_bot/ --include="*.py"  →  0 hit
```

**Conséquence** :
- Détection alpha decay live vs backtest **impossible**
- Pas de `Telegram alert` PSR < 0.3
- Pas de `kill_system` PSR < 0.1 sur 30j rolling
- L'objectif "Sharpe OOS > 1.5 stable sur 90j testnet" ne pourra pas être
  monitoré via le mécanisme prévu

**Fix proposé** : créer `binance_bot/routines/psr_daily_job.py` qui :
1. Lit `data/balance_history.jsonl` ou `data/trades_meta.jsonl`
2. Calcule SR live des 30 derniers jours
3. Appelle `src.psr_live.compute_psr(sr_live, sr_benchmark, n_obs, skew, kurt)`
4. Append `{ts, n_obs, psr, alert, sr_benchmark}` dans `psr_history.jsonl`
5. Telegram alert si PSR < 0.3
6. Kill switch si PSR < 0.1 sur 30j

Brancher ensuite dans `daily_phase_job.py` ou créer un timer dédié
`hsbc-psr.timer` (00:30 UTC).

### C.2 — `[MAJEUR]` Trou de tracking BUG-A à BUG-F

**Symptôme** : Le résumé Opus web mentionne `BUG-A` à `BUG-F` (les 6 bugs
critiques découverts à l'audit du 27/04 matin). `BUGS.md` n'utilise pas
ces identifiants — il liste BUG-001 à BUG-006, BUG-PRE-001, BUG-CRITICAL,
BUG-B9 à B13, BUG-ALGO, BUG-BONUS, WS-001.

**Recoupement probable** (à confirmer par contenu) :
- BUG-A "cost_model non importé" → traité par P0bis (commit dans pipeline)
- BUG-B "Kelly per-instance non portfolio-aware" → traité par P0
- BUG-C/D/E → explicitement renvoyés à `MISSION_REWFA.md` ✅
- BUG-F "DSR cumulé non déflaté" → partiellement traité (P5/P10 utilisent 50 trials)

**Fix proposé** : ajouter une section "Mapping audit 27/04" en tête de
`BUGS.md` qui matche les noms BUG-A/B/C/D/E/F aux entrées actuelles, ou
renommer/mettre des aliases. Sinon un futur lecteur (humain ou IA) ne
pourra pas tracer BUG-A → fix correspondant.

### C.3 — `[MINEUR]` Test robustesse composite signal liq=None non validé

**Symptôme** : la signature `compute_composite_score(top_ls_path, taker_path,
liq_path, oi_path, days_back)` prend des chemins de fichiers, pas des
valeurs. Le test du résumé "compute_score(liq=None)" est inopérable tel quel.

**Fix proposé** : créer un test pytest qui passe un `liq_path` pointant
vers un JSONL vide ou inexistant et vérifie que :
1. La fonction ne crash pas (pas de NaN, pas d'exception)
2. Renormalise les 3 autres poids (top_ls, taker, oi) à somme=1.0
3. Loggue clairement la situation "liq missing"

### C.4 — `[MAJEUR — design]` Phase K3 chargée mais pas utilisée pour décision

**Symptôme** : `phase_today` est chargé chaque jour (`daily_phase_job.py`)
et persisté dans `state.json`, mais **AUCUN code ne l'utilise pour décider
du bias directionnel**. Ichimoku traite indépendamment du contexte phase.

**Conséquence** : la première moitié de la thèse (couche CONTEXTE) n'a pas
d'effet sur les trades. Phase K3 + days_since_halving sont juste des
labels d'audit post-trade.

**Note** : ce point est traité dans la section D (thèse trading non
implémentée). C'est attendu — Sky n'a pas demandé son implémentation
immédiate. Mais il faut que ce soit **explicite** quelque part dans le repo
(pas juste deviné par audit). Recommandation : ajouter une note en tête
de `daily_phase_job.py` "Phase chargée pour audit uniquement — pas de
directional bias actif. À implémenter post-J+30."

---

## D. THÈSE TRADING

**Implémentée à ~30%** (briques défensives oui, briques directionnelles non).

| Pilier thèse | État | Détail |
|---|---|---|
| **7.1 — Bias directionnel daily (long/short/flat depuis K3+halving)** | ❌ MANQUANT | Phase K3 chargée mais pas utilisée pour décider direction. Halving calculé pour audit context uniquement. |
| **7.2 — Exécution multi-TF (1m/5m/15m/H1)** | ❌ MANQUANT | Mono-TF H2 uniquement (`bot_settings.yaml:8` + `data_fetcher.py:16`). Pas de cycle 1m/5m/15m/H1. |
| **7.3 — Cassure nuage Ichimoku multi-TF (pyramidal)** | ❌ MANQUANT | Cloud breakout solo, pas de confirmation cascade H2→H1→15m→5m→1m. |
| **7.4 — Fenêtre temporelle post-close [00:01, 02:00] UTC** | ❌ MANQUANT | Pas de logique de timing privilégié post-bougie H1/H4/D. |
| **7.5 — Cap journalier de GAIN à 3%** | ✅ PRÉSENT | `daily_gain_soft_cap_pct: 0.03` actif, flat + block jusqu'à minuit UTC. |
| **7.6 — Cycle halving comme prior** | ⚠️ PARTIEL | Calculé + persisté en meta_context, mais pas utilisé comme modificateur de proba. |

### Modules requis pour construire la thèse complète : tous présents

| Module | Présent | Chemin |
|---|---|---|
| HAR-RV (P4) | ✅ | `src/har_rv.py` (fit, predict, classify_regime) |
| VPIN gate (P7) | ✅ | `src/vpin_gate.py` (mode log_only effectif) |
| Composite signal (P6.5) | ✅ | `binance_bot/services/flow_composite_signal.py` |
| Funding-aware close (P11) | ✅ | `binance_bot/routines/funding_close_runner.py` |
| Portfolio VaR gate (P3) | ✅ | `src/portfolio_risk_gate.py` |
| Drawdown gate (P2) | ✅ | `src/risk_sizing.py:143` (anti-martingale) |
| Daily caps (P1) | ✅ | `bot_settings.yaml` + `signal_engine.py` |
| Portfolio sizing (P0) | ✅ | `intraday_runner.py:104-163` |
| LightGBM combinator (P10) | ✅ | `src/lgbm_combinator.py` |
| EGARCH/TGARCH (P8) | ✅ | tests présents `test_p8_garch.py` + `test_p8_garch_arch.py` |

**Verdict thèse** : Sky a tous les BLOCS défensifs et observatoires. Il manque
les 4 briques DIRECTIONNELLES (bias daily, multi-TF, pyramidal, post-close
window). Cohérent avec ce que l'agent web a écrit Section 9 du résumé :
ces 4 éléments sont pour **post-J+30** après analyse des données soak.

---

## E. RECOMMANDATIONS

### E.1 — Avant la prochaine mission autonome

1. **Brancher PSR live** (CRIT-7) — bloquant pour le monitoring de stabilité
   sur 30 jours. Sinon on n'aura aucune métrique objective de drift edge
   live vs backtest à J+30.
2. **Ajouter mapping BUG-A→F** dans `BUGS.md` — 5 minutes, évite confusion
   future.
3. **Test pytest composite signal `liq=None`** — robustesse à valider
   suite à WS-001.
4. **Note explicite dans `daily_phase_job.py`** : "phase chargée pour audit,
   pas pour décision — voir REPORT_CHECK_2026-04-30 section D".

### E.2 — Questions ouvertes pour Sky (post-mission)

1. **Bias directionnel** : on garde phase K3 + halving en pure observation,
   ou on commence à coder la décision long/short/flat post-J+30 ?
2. **Multi-TF** : option A (pyramidal H2→H1→15m→5m→1m) ou B (parallèle 3
   trades indep) ? La conversation initiale recommandait pyramidal mais
   Sky n'a jamais tranché.
3. **Window 60-120s post-close** : doit-on attendre la collecte de données
   intra-bar pendant le soak avant d'implémenter ?
4. **PSR < 0.3 alert** : Telegram (canal préféré) ou autre mécanisme ?

### E.3 — Ordre de priorité suggéré post-soak J+30

1. **Phase 1 (J+30)** : analyse soak, calibration composite, retrain LGBM,
   bilan PSR (cf NEXT_STEPS_30D.md)
2. **Phase 2 (~J+45)** : fix CRIT-7 (PSR live cron) — devrait être fait AVANT
   J+30 idéalement
3. **Phase 3 (~J+60)** : MISSION_REWFA pour réconcilier Sharpe (BUG-C/D/E)
4. **Phase 4 (post-REWFA)** : implémentation thèse multi-TF si Sharpe live
   > backtest pruné honnête + edge confirmé

---

## F. VERDICT

**`GO mission autonome — avec UNE correction préalable`**

La stack est solide et déployable telle quelle pour le soak 30j. **Une seule
chose bloque vraiment** : sans PSR live cron (CRIT-7), on n'aura aucune
métrique objective à J+30 pour décider de la suite. C'est un fix de ~2h
(créer `psr_daily_job.py` + brancher dans timer hsbc-daily ou nouveau timer
hsbc-psr).

**Proposition** :

- **Court terme (cette semaine)** : fix CRIT-7 + C.2 (mapping BUG-A→F) +
  C.3 (test pytest composite). 4-5h de travail.
- **Soak continue** : pas d'autre intervention sur le code pendant les 28
  jours restants. Trade_mode reste en simulation.
- **À J+30** : revue manuelle complète selon `NEXT_STEPS_30D.md`, décisions
  sur thèse multi-TF basées sur données collectées.

**Pas de STOP système** : aucun bug critique de runtime, aucun risque de
liquidation, anti-martingale validé, daily caps actifs, VPIN log_only
effectif, daemons live stables.

---

## ANNEXE — Trous de la conversation Opus web

(Réponse à la demande "si je n'ai pas mentionné quelque chose d'important")

1. **Tests slow non comptés** : le résumé annonce 671 = 641 normal + 30
   slow. L'audit local trouve **704 not slow** + 30 slow. Soit le projet a
   ajouté ~63 tests depuis la session web, soit le compte initial était
   approximatif. Pas un problème, juste à noter.
2. **L-006** : la séquence L-001 à L-007 saute L-006. Intentionnel ou trou
   de numérotation ? À vérifier dans la conversation source.
3. **`backup_status.json`** existe dans `data/` — le résumé n'en parle pas
   mais c'est lié au timer `hsbc-backup.timer` (toutes les 6h).
4. **Daemon `hsbc-vpin-live.service` est `active running`** mais avec PID
   711, uptime 4h+ au moment de l'audit. Cohérent avec post-mission.
5. **L-008** post-merge (commit 28d0e24) : VPIN sys.path missing repo root
   — bug subtil corrigé après le merge initial. Bien tracé via L-008.

---

*Rapport rédigé sans modification du code.*
*Pour questions : voir branchement détaillé dans les annexes ou re-lancer*
*l'audit avec `pytest -m "not slow" -v` pour la liste exhaustive des tests.*
