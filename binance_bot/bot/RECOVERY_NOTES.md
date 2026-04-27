# Recovery WAL & Idempotence — BUG-B12

## Pourquoi un WAL ?

Entre `create_market_buy_order` (entrée) et `_place_algo_order` (SL/TP) un crash
laisse une **position naked** sur Binance : qty ouverte sans stop. `reconciler`
voit la position mais ignore le stop voulu (entry/stop/tp = 0). Le WAL persiste
l'**intent** AVANT chaque appel API. Au boot, on rejoue les intents `pending`
avec le même clientOrderId — sans doublon.

## Pattern d'usage runner (pseudo-code)

```python
from bot.recovery_wal import WAL, deterministic_client_order_id

wal = WAL(Path("data/wal.jsonl"))

# Boot recovery
for intent in wal.pending_intents():
    p = intent["payload"]
    try:
        result = trade_mgr._place_algo_order(
            symbol=p["symbol"], side=p["side"], stop_price=p["stop"],
            qty=p["qty"], client_order_id=p["client_order_id"],
        )
        wal.mark_completed(intent["intent_id"], result)
    except Exception as e:
        wal.mark_failed(intent["intent_id"], str(e))

# Trading loop
bar_ts = int(df.index[-1].timestamp() * 1000)
coid_sl = deterministic_client_order_id("ich2h", bar_ts, "sl")
intent_id = f"open-long-sl-{bar_ts}"

wal.record_intent(intent_id, {
    "action": "open_long_sl", "symbol": "BTC/USDT", "side": "SELL",
    "stop": stop_price, "qty": qty, "client_order_id": coid_sl,
})
result = trade_mgr._place_algo_order(..., client_order_id=coid_sl)
wal.mark_completed(intent_id, result)
```

## clientOrderId déterministe

`deterministic_client_order_id(strategy, bar_ts_ms, leg)` = pure function des
inputs. Binance USDM rejette les `newClientOrderId` doublons → un **retry
identique** retourne la même réponse plutôt que de créer un second ordre.
Tronqué à 36 chars (limite Binance).

## Format JSONL

Une ligne = un événement, append-only :
```jsonl
{"intent_id":"open-long-sl-1700000000000","status":"pending","ts_iso":"2026-04-26T10:00:00.000Z","action":"open_long_sl","payload":{...}}
{"intent_id":"open-long-sl-1700000000000","status":"completed","ts_iso":"2026-04-26T10:00:00.500Z","result":{"orderId":12345}}
```

`pending_intents()` ne retourne que les intents dont la **dernière** ligne est
`pending`. `clear_completed(before_iso)` compacte : supprime les intents avec
statut final `completed`/`failed` antérieurs au cutoff. Tous les writes sont
suivis d'un `fsync` (durabilité kernel/disque).

## Limitation

Si Binance reçoit l'ordre mais on perd la réponse (network timeout après
acceptation), l'intent reste `pending`. Au boot, replay avec le **même**
clientOrderId → Binance retourne la réponse originelle (pas de second ordre).
Le WAL ne couvre pas la corruption silencieuse de la réponse — mais
l'idempotence Binance la gère.
