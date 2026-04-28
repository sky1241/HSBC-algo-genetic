# NEXT STEPS J+30 — Revue post-soak testnet

> Document de travail pour la revue manuelle à effectuer **30 jours après
> le démarrage du soak testnet** sur la stack P0bis-P12 + R1-R10 + F-001/3/4.
> **Pas d'agent automatique** — Sky lance cette revue manuellement quand
> il estime que les 30j sont écoulés.

## Date cible

- **Démarrage soak** : à fixer par Sky (probablement à partir du 2026-04-29
  si la dernière commit `77dea99` est validée pour testnet).
- **Date revue J+30** : démarrage + 30 jours civils.

## Tâches à effectuer

### Tâche 1 — Calibration des seuils composite signal (P6.5)

- Inspecter `binance_bot/data/flow_composite_log.jsonl` accumulé sur 30j.
- Calculer la distribution empirique du `score` composite :
  ```
  $ python -c "
  import json
  scores = [json.loads(l)['score'] for l in open('binance_bot/data/flow_composite_log.jsonl')]
  import numpy as np
  print(f'mean={np.mean(scores):.3f}, std={np.std(scores):.3f}')
  for q in (0.05, 0.25, 0.5, 0.75, 0.95):
      print(f'q{int(q*100)}: {np.quantile(scores, q):.3f}')
  "
  ```
- Décider :
  - Seuil GATE actuel ≈ 0.4 → si quantile 95% absolu < 0.3, le seuil ne
    sera jamais atteint → **recalibrer plus bas**.
  - Si quantile 5% > 0.3, le bot serait bloqué la moitié du temps →
    seuil trop bas → **recalibrer plus haut**.

### Tâche 2 — Retrain LightGBM P10 avec features pre_trade enrichies

- Pré-requis : R7-bis doit avoir été fait (capture features à open dans
  `state_manager.add_position`). Sinon, `trades_meta.jsonl` reste avec
  les `pre_trade.{vpin, rv, ...}` à None.
- Lancer le training :
  ```
  $ python scripts/training/train_lgbm_combinator.py --no-quick \
      --n-trials 50 --years-back 1
  ```
  (Avec `--no-quick`, HAR-RV + EGARCH rolling sont calculés réellement.
  Avec `--years-back 1` au lieu de 4, on utilise les 30j live + ~11 mois
  historiques BTC H1, ce qui donne un mix réel/synthétique.)
- Si `should_deploy=True` cette fois → déployer le modèle dans
  `intraday_runner._make_low_vol_combinator_fn(model_path)` (factory
  à créer) et activer le hook `low_vol_combinator_fn` dans SignalEngine.
- Si `should_deploy=False` à nouveau → analyser pourquoi (features pas
  assez riches ? distribution `low_vol next_hour` mal posée ?).

### Tâche 3 — Bilan PSR live quotidien sur 30 jours

- Inspecter `binance_bot/data/psr_history.jsonl` accumulé.
- Calculer :
  - PSR médian sur 30j
  - PSR rolling 7j (lissage)
  - Nombre de jours où PSR < 0.3 (alpha decay) ou < 0.1 (kill)
- Comparer avec les seuils opérationnels P5 :
  - PSR ≥ 0.5 : OK cohérent backtest
  - PSR < 0.5 : edge live INFÉRIEUR backtest
  - PSR < 0.3 : alpha decay sérieux → investigation
  - PSR < 0.1 sur 30j rolling → **arrêt système**, audit complet

### Tâche 4 — Decision tree

```
Si PSR médian 30j < 0.3 :
  → ARRÊT SYSTÈME
  → Lancer mission MISSION_REWFA.md (BUG-C/D/E)
  → Investiguer composite signal calibration
  → Retrain LGBM avec features enrichies
  → NE PAS bascule testnet → mainnet

Si PSR médian 30j ∈ [0.3, 0.5] :
  → Edge dégradé mais soutenable
  → Garder en testnet 30j supplémentaires
  → Recalibrer composite si nécessaire
  → Considérer activer mode `vpin_mode: "gate"` SEULEMENT si P7-bis
    (collecteur VPIN live) a été closed entretemps

Si PSR médian 30j > 0.5 :
  → Edge cohérent backtest
  → Considérer bascule testnet → mainnet (capital très petit, $50)
  → Activer P12 MSM si désiré (V2)
  → Maintenir P5 daily PSR check
```

## Notifications à mettre en place avant J+30

- ☐ Telegram alert quotidien si PSR < 0.3 (déjà branché P5)
- ☐ Telegram alert si `funding_close` exécute un close (R9 + R7
  audit_log entry)
- ☐ Telegram alert si `vpin_gate` passe en `kill_and_block` (déjà
  branché P7 R5, mais data_fn = None aujourd'hui — actif post-P7-bis)

## Logs à conserver pour le bilan

- `binance_bot/data/psr_history.jsonl` (P5)
- `binance_bot/data/flow_composite_log.jsonl` (P6.5/R4)
- `binance_bot/data/trades_meta.jsonl` (P9/R7)
- `binance_bot/data/trades_audit.jsonl` (audit_log central avec
  `event=garch_har_*`, `event=funding_close_*`, etc.)
- `binance_bot/data/vpin_events.jsonl` (P7/R5 — restera vide tant que
  P7-bis n'est pas fait)
- `binance_bot/logs/intraday.log`, `funding_close.log`, etc.

## Statut

- ☐ J+30 atteint
- ☐ Tâche 1 effectuée (calibration composite)
- ☐ Tâche 2 effectuée (retrain LGBM)
- ☐ Tâche 3 effectuée (bilan PSR)
- ☐ Décision prise (continuer testnet / mainnet petit / arrêt + REWFA)
