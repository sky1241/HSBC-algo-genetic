#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconciliation: synchronise state.json avec les positions réelles côté Binance.

BUG-006: si le bot crash/reboot pendant qu'une position est ouverte, ou si une
position est fermée manuellement, state.json et l'état réel divergent.
"""
from typing import Tuple


def reconcile_positions(state_mgr, exchange, symbol: str) -> Tuple[int, int]:
    """Synchronise state.positions_{long,short} avec exchange.fetch_positions.

    Stratégie: agrégat par côté.
        - Si Binance a une qty ouverte sur un côté et state n'a rien → ajoute une
          position "reconciled" avec entry/stop/tp inconnus (à 0).
        - Si state a des positions sur un côté et Binance n'a rien → vide la liste.

    Args:
        state_mgr: instance StateManager (méthodes get/set/save)
        exchange: ccxt exchange (méthode fetch_positions)
        symbol: paire (ex: "BTC/USDT")

    Returns:
        (added_count, removed_count): positions ajoutées et retirées du state.
    """
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception as e:
        print(f"⚠️ fetch_positions failed during reconcile: {e}")
        return (0, 0)

    binance_long_qty = 0.0
    binance_short_qty = 0.0
    canon = symbol.replace("/", "")  # "BTCUSDT"
    for p in positions or []:
        psym = p.get('symbol')
        info_sym = (p.get('info') or {}).get('symbol')
        # ccxt linéaires retournent "BTC/USDT:USDT"; on accepte plusieurs formats
        if psym not in (symbol, f"{symbol}:USDT") and info_sym != canon:
            continue
        contracts = float(p.get('contracts') or 0)
        if contracts == 0:
            continue
        side = (p.get('side') or '').lower()
        if side == 'long':
            binance_long_qty += abs(contracts)
        elif side == 'short':
            binance_short_qty += abs(contracts)

    state_long = list(state_mgr.get('positions_long', []) or [])
    state_short = list(state_mgr.get('positions_short', []) or [])
    state_long_qty = sum(float(p.get('size', 0) or 0) for p in state_long)
    state_short_qty = sum(float(p.get('size', 0) or 0) for p in state_short)

    added = 0
    removed = 0

    # BUG-007: utiliser ±inf (pas 0) pour ne pas trigger faux tp_hit/stop_hit
    # Une position reconciled a entry/stop/tp inconnus — on évite que signal_engine
    # croie que current_price >= tp=0 ou current_price <= stop=0.
    INF = float("inf")
    if binance_long_qty > 0 and state_long_qty == 0:
        state_long.append({
            'id': f'reconcile_long_{symbol}',
            'entry': 0.0,
            'stop': -INF,    # long stop = très bas → stop_hit jamais True (current >= stop)
            'tp': INF,       # long tp  = très haut → tp_hit jamais True
            'size': binance_long_qty,
            'reconciled': True,
        })
        added += 1

    if binance_short_qty > 0 and state_short_qty == 0:
        state_short.append({
            'id': f'reconcile_short_{symbol}',
            'entry': 0.0,
            'stop': INF,     # short stop = très haut → jamais déclenché
            'tp': -INF,      # short tp  = très bas → jamais déclenché
            'size': binance_short_qty,
            'reconciled': True,
        })
        added += 1

    if binance_long_qty == 0 and state_long_qty > 0:
        removed += len(state_long)
        state_long = []

    if binance_short_qty == 0 and state_short_qty > 0:
        removed += len(state_short)
        state_short = []

    state_mgr.set('positions_long', state_long)
    state_mgr.set('positions_short', state_short)

    if added or removed:
        state_mgr.save()
        print(f"🔄 Reconcile {symbol}: +{added} ajout(s), -{removed} retrait(s)")
    return (added, removed)
