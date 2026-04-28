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

## L-001 — VPIN data_fn placeholder retourne None

- **Origine**: R5 / P7 — `binance_bot/routines/intraday_runner.py` fonction
  `_make_vpin_data_fn(symbol)` (lignes ~225-235 du fichier).
- **Description**: La fonction retourne toujours `None`. Le gate VPIN
  ne reçoit donc jamais de (vpin, obi) live et la branche
  `_evaluate_vpin_gate` retourne `("allow", "vpin_data_unavailable")`.
- **Cause**: les collecteurs live aggTrade WS (pour VPIN volume buckets)
  + bookTicker / depth5 (pour OBI = order book imbalance) ne sont pas
  câblés dans le bot.
- **Chunk de résolution**: **P7-bis** — collecteur VPIN live (task #69).
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

## L-002 — Meta-label features pre_trade partiellement None

- **Origine**: R7 / P9 — `binance_bot/routines/intraday_runner.py` fonction
  `_build_meta_context` (champs `pre_trade.{rv_predicted_har, vpin_at_entry,
  cloud_breakout_size_atr_units, volume_relative_30d, funding_rate_at_entry_bps}` +
  `context.btc_dominance`).
- **Description**: Ces 6 champs sont mis à `None` parce qu'aucune source live
  ne les capture au moment de l'OPEN du trade (feature computation cycle-time
  vs trade-time mismatch). `regime_har`, `atr_at_entry`, `composite_signal`
  sont disponibles au close-time (recompute) → ces 3 sont remplis.
- **Cause**:
  - `vpin_at_entry` : cf L-001 (collecteur VPIN live pas encore branché).
  - `btc_dominance` : pas d'API client (CoinGecko etc.) configuré.
  - `rv_predicted_har` : calculable via P4 mais pas persisté à open.
  - `cloud_breakout_size_atr_units` : Ichimoku breakout taille pas calculée.
  - `volume_relative_30d` : avg volume 30j pas tracké.
  - `funding_rate_at_entry_bps` : flow_open_interest a la donnée mais
    pas requêtée à open.
- **Chunk de résolution**: **R7-bis (futur)** — capture features à open
  dans state_manager.add_position et lecture au close. Ou : runner cron
  qui pré-calcule + persiste le snapshot features chaque 5min.
- **Impact si non résolue**:
  - **Cosmétique aujourd'hui** : `build_meta_label` accepte None (test
    `test_meta_label_handles_missing_optional_features`). Munin reçoit
    juste un schema partiel mais valide. Hash chain OK.
  - **Bloquant** pour clustering Munin si on veut featurer/labeliser sur
    ces variables — l'analyse posteriori serait truquée si la majorité
    des trades a `vpin_at_entry=None`.
- **Garde-fou**: avant d'utiliser `trades_meta.jsonl` pour entraîner
  Munin/un classifier sur features pre_trade, vérifier la fraction de
  None par champ (devrait être <10% pour features critiques type
  vpin/regime/composite).

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
