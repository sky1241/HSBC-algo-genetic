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

## ⛔ Critères de kill prématuré (avant J+30)

Si **l'un** de ces critères est observé pendant le soak, **arrêt système
+ audit complet** :

1. **Drawdown > 15%** depuis equity_high — déclenché par P2
   `drawdown_size_multiplier=0.0` (kill par design). Vérifier
   `binance_bot/data/balance_history.jsonl` pour `equity_max_observed`
   et `equity_min_observed`.
2. **PSR live médian sur 7j rolling < 0.1** — alpha decay sévère.
   Calculer via `binance_bot/data/psr_history.jsonl` :
   ```
   $ python -c "
   import json, numpy as np
   psr_vals = [json.loads(l).get('psr') or 0.5
               for l in open('binance_bot/data/psr_history.jsonl')]
   rolling_7d = np.median(psr_vals[-7:])
   print(f'PSR median 7j rolling = {rolling_7d:.3f}')
   "
   ```
3. **Plus de 3 hard daily caps déclenchés en 30j** — discipline du bot
   trop souvent dépassée par marché. Compter dans
   `binance_bot/data/trades_audit.jsonl` les events
   `event=hard_cap_triggered` ou `kill_switch`.

**Action en cas de kill prématuré** :
- `touch binance_bot/data/.killed` (déclenche kill_switch immédiat)
- Lancer `MISSION_REWFA.md` (BUG-C/D/E re-evaluation pipeline historique)
- Pas de bascule mainnet avant remédiation complète

## ⏰ J+10 — Activation `HSBC_FUNDING_CLOSE_LIVE=1`

**Date cible** : timestamp `data/soak_start.txt` + 10 jours.

**Procédure d'activation** :
```bash
# 1. Vérifier que P11 a accumulé au moins 10 fires DRY-RUN dans
#    binance_bot/logs/funding_close.log (3 fires/jour × 10 jours = 30 fires).
grep -c "DRY-RUN" binance_bot/logs/funding_close.log

# 2. Calculer baseline funding paid sur les 10 jours :
#    On veut savoir combien de fees funding ont été payés pendant
#    cette période (sera la référence "sans P11").
python -c "
import json
total = 0.0
for line in open('binance_bot/data/balance_history.jsonl'):
    rec = json.loads(line)
    total += rec.get('funding_paid_usdt', 0.0)
print(f'Total funding paid (10j baseline) = \${total:.2f}')
"

# 3. Modifier l'unité systemd pour ajouter HSBC_FUNDING_CLOSE_LIVE=1 :
sudo systemctl --user edit hsbc-funding-close.service
# Dans le drop-in, ajouter :
#   [Service]
#   Environment="HSBC_FUNDING_CLOSE_LIVE=1"

# 4. Recharger + restart :
systemctl --user daemon-reload
systemctl --user restart hsbc-funding-close.timer

# 5. Vérifier au prochain fire (HH:55-59) que les logs contiennent
#    "mode=LIVE" au lieu de "mode=DRY-RUN" :
tail -50 binance_bot/logs/funding_close.log | grep "mode="
```

**Mesure attendue après J+20 (10j post-activation)** :
- Comparer funding_paid_usdt 10j avant LIVE vs 10j après LIVE
- Si delta > 0 (économie) → P11 fait son job
- Si delta ~ 0 → P11 ne sauve rien (funding rates faibles sur la période)
- Si delta < 0 → P11 close trop tôt et coûte plus → désactiver

## Notifications

**Décision Sky 2026-04-28** : Telegram **skip volontaire** (mode démo
testnet, pas de capital réel à risque). Toutes les alertes restent en
log-only dans `binance_bot/logs/*.log`. Sky vérifie manuellement par SSH
ou directement sur la machine quand il le souhaite.

Commandes utiles pour monitoring manuel (à lancer périodiquement) :
```bash
# État du bot intraday (dernier cycle)
tail -50 /home/ludov/HSBC-algo-genetic/binance_bot/logs/intraday.log

# PSR live history (alerte si dernier psr < 0.3)
tail -5 /home/ludov/HSBC-algo-genetic/binance_bot/data/psr_history.jsonl

# Dernières décisions funding-close
tail -20 /home/ludov/HSBC-algo-genetic/binance_bot/logs/funding_close.log

# Composite signal accumulation (target ~30j de baseline)
wc -l /home/ludov/HSBC-algo-genetic/binance_bot/data/flow_composite_log.jsonl

# Audit log central (events kill / disagreement / etc.)
tail -30 /home/ludov/HSBC-algo-genetic/binance_bot/data/trades_audit.jsonl

# Vérifier qu'aucun kill_switch ne s'est déclenché
ls /home/ludov/HSBC-algo-genetic/binance_bot/data/.killed 2>/dev/null \
  && echo "KILL SWITCH ACTIF — investiguer" \
  || echo "OK — pas de kill"
```

Si plus tard Sky veut activer Telegram, il suffit d'ajouter
`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` dans `binance_bot/.env`. Le
`TelegramNotifier` détecte automatiquement et bascule en push live.

**À ajouter avant le soak réel** :
```
$ vi binance_bot/.env
# Ajouter :
TELEGRAM_BOT_TOKEN=<ton_token_BotFather>
TELEGRAM_CHAT_ID=<ton_chat_id>
```

Une fois configuré, tester avec :
```
$ python -c "
from binance_bot.bot.notifier import TelegramNotifier
n = TelegramNotifier()
print('enabled:', n.is_enabled())
n.info('TEST routing P5/P11/P7 — soak start')
"
```

Une fois Telegram ON, les alertes suivantes seront déclenchées :
- ☐ Daily PSR < 0.3 (déjà branché P5 via `scripts/production/psr_live_daily.py`)
- ☐ funding_close kill executed (R9, audit_log + notifier.warn dans runner)
- ☐ VPIN gate kill_and_block (P7 R5, mais inactif tant que P7-bis pas fait)
- ☐ Hard cap daily loss déclenché (P1)
- ☐ Drawdown > 15% (P2 kill)

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
