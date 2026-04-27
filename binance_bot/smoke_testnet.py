#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests live sur testnet.binancefuture.com.

À runner depuis la racine du repo après config de binance_bot/.env :
    .venv/bin/python -m binance_bot.smoke_testnet [step]

Steps : 1=balance, 2=ohlcv, 3=leverage, 4=reconcile, 5=algo_sl, 6=open_close, all=tous.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from services.data_fetcher import DataFetcher
from bot.trade_manager import TradeManager
from bot.reconciler import reconcile_positions
from bot.state_manager import StateManager


SYMBOL = "BTC/USDT"
TIMEFRAME = "2h"
LEVERAGE = 5  # plus prudent que 10 pour les tests


def banner(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def step1_balance(dfetcher):
    banner("STEP 1 — Connexion + balance testnet")
    bal = dfetcher.exchange.fetch_balance()
    usdt = bal.get('USDT', {})
    print(f"  USDT free  = {usdt.get('free')}")
    print(f"  USDT used  = {usdt.get('used')}")
    print(f"  USDT total = {usdt.get('total')}")
    return float(usdt.get('free') or 0)


def step2_ohlcv(dfetcher):
    banner("STEP 2 — fetch_ohlcv + filtre closed-bar (BUG-004)")
    df = dfetcher.get_ohlcv(limit=10)
    print(f"  Bougies retournées: {len(df)}")
    print(f"  Première: {df.iloc[0]['timestamp']}")
    print(f"  Dernière: {df.iloc[-1]['timestamp']}")
    import pandas as pd
    now = pd.Timestamp.now(tz='UTC')
    last_ts = df.iloc[-1]['timestamp']
    tf_delta = pd.Timedelta(TIMEFRAME)
    age_minutes = (now - last_ts).total_seconds() / 60
    print(f"  Now (UTC):       {now}")
    print(f"  Last_ts + 2h:    {last_ts + tf_delta}")
    print(f"  Last bougie age: {age_minutes:.1f} minutes (clôturée car last_ts+2h <= now)")
    assert last_ts + tf_delta <= now, "BUG-004 violé: dernière bougie pas clôturée"
    print("  ✅ Filtre closed-bar OK")
    return df


def step3_leverage(tm):
    banner("STEP 3 — apply_leverage_on_exchange (BUG-002)")
    ok = tm.apply_leverage_on_exchange()
    print(f"  Retour: {ok}")
    return ok


def step4_reconcile(tm, dfetcher):
    banner("STEP 4 — reconcile sans positions (BUG-006)")
    state_path = ROOT / "data" / "smoke_state.json"
    if state_path.exists():
        state_path.unlink()
    sm = StateManager(str(state_path))
    added, removed = reconcile_positions(sm, dfetcher.exchange, SYMBOL)
    print(f"  added={added}, removed={removed}")
    print(f"  positions_long: {sm.get('positions_long')}")
    print(f"  positions_short: {sm.get('positions_short')}")
    return (added, removed)


def step5_algo_sl(tm, dfetcher):
    banner("STEP 5 — _place_algo_order POST (BUG-ALGO)")
    ticker = dfetcher.exchange.fetch_ticker(SYMBOL)
    last_price = float(ticker['last'])
    print(f"  Prix actuel: {last_price}")
    # Stop très loin sous le marché → ne déclenchera pas
    trigger = round(last_price * 0.5, 1)
    print(f"  Trigger SELL STOP_MARKET @ {trigger} (50% sous le marché — sera annulé après)")
    # qty minimum BTCUSDT futures = 0.001
    algo_id = tm._place_algo_order("SELL", "STOP_MARKET", 0.001, trigger)
    print(f"  algo_id = {algo_id}")
    if algo_id:
        # Tenter de l'annuler (ce sera utile pour valider le cleanup)
        # Note: l'endpoint cancel est aussi un /fapi/v1/algoOrder/{algoId} probablement
        # Pour l'instant on note juste l'id
        print(f"  ⚠️  Note: cet algo_id {algo_id} reste ouvert. À cancel manuellement si nécessaire.")
    return algo_id


def step6_open_close(tm, dfetcher):
    banner("STEP 6 — Open LONG + Close LONG (BUG-001 + BUG-005 + BUG-ALGO)")
    ticker = dfetcher.exchange.fetch_ticker(SYMBOL)
    last_price = float(ticker['last'])
    print(f"  Prix actuel: {last_price}")

    # Signal LONG mini : qty=0.001 (min Binance)
    # On force la qty via size_pct ajusté
    # qty = capital * size * leverage / price → 0.001 = capital * size * 5 / price
    # Avec capital=20: size = 0.001 * price / (20 * 5) = price * 0.00001
    # À price=78000, size = 0.78
    capital_test = 20.0
    needed_size_pct = (0.001 * last_price) / (capital_test * tm.leverage)
    print(f"  size_pct ajusté pour qty 0.001: {needed_size_pct:.4f}")

    stop_init = round(last_price * 0.95, 1)
    tp = round(last_price * 1.10, 1)
    print(f"  Open LONG @ market | SL initial @ {stop_init} | TP @ {tp}")

    order_id = tm.execute_signal(
        {"action": "open_long", "entry": last_price, "stop": stop_init,
         "tp": tp, "size": needed_size_pct},
        capital_usdt=capital_test,
    )
    print(f"  open_long order_id = {order_id}")
    if not order_id:
        print("  ❌ Echec ouverture LONG — abort step 6")
        return None

    # Attendre 2s pour que la position s'enregistre côté Binance
    time.sleep(2)

    # Vérifier la position côté Binance
    positions = dfetcher.exchange.fetch_positions([SYMBOL])
    for p in positions:
        if p.get('symbol') == SYMBOL and float(p.get('contracts') or 0) > 0:
            print(f"  📊 Position détectée: side={p.get('side')}, qty={p.get('contracts')}, "
                  f"entry={p.get('entryPrice')}, leverage={p.get('leverage')}")

    # Fermer
    print(f"  Closing LONG (cancel SL/TP + market sell reduceOnly)...")
    close_id = tm.execute_signal(
        {"action": "close_long", "exit": last_price, "reason": "smoke_test"},
        capital_usdt=capital_test,
    )
    print(f"  close_long order_id = {close_id}")

    time.sleep(2)
    # Vérifier que la position est bien fermée
    positions = dfetcher.exchange.fetch_positions([SYMBOL])
    still_open = False
    for p in positions:
        if p.get('symbol') == SYMBOL and float(p.get('contracts') or 0) > 0:
            still_open = True
            print(f"  ❌ Position toujours ouverte: {p}")
    if not still_open:
        print(f"  ✅ Position bien fermée côté Binance")
    return close_id


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    dfetcher = DataFetcher(symbol=SYMBOL, timeframe=TIMEFRAME)
    # Détecter le mode du compte (hedge vs one-way)
    try:
        dual = dfetcher.exchange.fapiPrivateGetPositionSideDual()
        is_hedge = dual.get('dualSidePosition') in (True, 'true')
    except Exception as e:
        print(f"⚠️ Détection mode position: {e} — défaut hedge")
        is_hedge = True
    pos_mode = "hedge" if is_hedge else "oneway"
    print(f"⚙️  position_mode détecté: {pos_mode}")

    tm = TradeManager(
        exchange=dfetcher.exchange,
        symbol=SYMBOL,
        mode="live",
        leverage=LEVERAGE,
        algo_base_url="https://testnet.binancefuture.com",
        position_mode=pos_mode,
    )

    if step in ("1", "all"):
        step1_balance(dfetcher)
    if step in ("2", "all"):
        step2_ohlcv(dfetcher)
    if step in ("3", "all"):
        step3_leverage(tm)
    if step in ("4", "all"):
        step4_reconcile(tm, dfetcher)
    if step in ("5", "all"):
        step5_algo_sl(tm, dfetcher)
    if step in ("6", "all"):
        step6_open_close(tm, dfetcher)

    banner("SMOKE TESTS DONE")


if __name__ == "__main__":
    main()
