# HSBC Ichimoku K3 — Dashboard local

Petite UI Flask qui tourne sur `http://localhost:8080` pour visualiser le bot
sur Binance Futures testnet.

## Ce qu'on voit

- Pill `Running / KILLED` selon `binance_bot/data/.killed`
- Mode (`simulation` / `live`) et réseau (`testnet` / `mainnet`)
- Solde USDT (free / used / total) lu via `ccxt.binanceusdm` côté serveur
- P&L vs starting balance 5000 USDT (vert si positif, rouge sinon)
- Dernier prix BTC/USDT
- Phase du jour + paramètres Ichimoku (tenkan, kijun, senkou_b, shift, atr_mult, tp_mult)
- Positions ouvertes (côté Binance, fetch_positions) avec entry / mark / unrealized P&L / leverage / liquidation
- Algo orders ouverts (SL/TP) via `TradeManager.fetch_open_algo_orders()` (`/fapi/v1/openAlgoOrders`)
- 10 derniers blocs du log `intraday.log` et 5 du `daily.log`
- Bouton `Force intraday run` qui fait `systemctl --user start hsbc-intraday.service`

Auto-refresh : 30 s côté client (vanilla `fetch` + `setInterval`).
Cache serveur : 8 s sur les calls Binance pour éviter de cogner sur le rate limit.

## Sécurité

- Les clés API ne sortent JAMAIS du serveur Flask. Le client n'a que l'output JSON
  filtré par `/api/status` (balance, positions, algoOrders, last_price, state, logs).
- Par défaut écoute sur `127.0.0.1` (loopback) — non exposé sur le LAN.
  Pour exposer (si besoin), changer `HSBC_DASHBOARD_HOST=0.0.0.0` dans le service.

## Lancement

### Option 1 — directement (foreground)

```bash
./binance_bot/dashboard/run.sh
```

Puis ouvrir [http://localhost:8080](http://localhost:8080).

### Option 2 — via systemd user (recommandé, restart auto)

```bash
./binance_bot/systemd/install.sh        # installe + enable timers + dashboard
./binance_bot/systemd/install.sh status  # vérifie l'état
```

L'install copie aussi `hsbc-dashboard.service` dans `~/.config/systemd/user/`
et l'active. Logs : `binance_bot/logs/dashboard.log`.

Pour qu'il tourne sans login actif :
```bash
sudo loginctl enable-linger $USER
```

## Endpoints HTTP

| Endpoint              | Méthode | Description                                                  |
|-----------------------|---------|--------------------------------------------------------------|
| `/`                   | GET     | Page HTML (auto-refresh JS)                                  |
| `/api/status`         | GET     | JSON snapshot (balance, positions, algoOrders, state, logs)  |
| `/api/status?force=1` | GET     | Idem mais bypass le cache 8s                                 |
| `/api/run-intraday`   | POST    | Trigger `systemctl --user start hsbc-intraday.service`       |

## Kill / reset

Le bouton "kill" n'est pas exposé dans l'UI (volontaire — éviter clic accidentel).

- Pour killer manuellement le bot :
  ```bash
  echo "manual kill at $(date -Is)" > /home/ludov/HSBC-algo-genetic/binance_bot/data/.killed
  ```
  Le pill UI bascule en `KILLED` (rouge) au prochain refresh, et `intraday_runner`
  refusera de tourner.

- Pour réactiver :
  ```bash
  rm /home/ludov/HSBC-algo-genetic/binance_bot/data/.killed
  ```

- Pour stopper le dashboard :
  ```bash
  systemctl --user stop hsbc-dashboard.service
  ```

## Variables d'env utiles

- `HSBC_DASHBOARD_HOST` (défaut `127.0.0.1`)
- `HSBC_DASHBOARD_PORT` (défaut `8080`)
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` / `BINANCE_TESTNET` (lus depuis `binance_bot/.env`)
- `TRADE_MODE` (lu pour affichage uniquement; le dashboard ne place jamais d'ordre)
