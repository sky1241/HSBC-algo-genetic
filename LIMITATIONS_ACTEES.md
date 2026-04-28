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
