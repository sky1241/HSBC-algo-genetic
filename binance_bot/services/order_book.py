#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Order book metrics (Layer-2 quick win): OBI + spread + depth at 50bps.

Lecture via REST `fetch_order_book` (gratuit Binance, weight 5). Pour usage WebSocket
avec depth20 stream voir `bot/user_stream.py`.

OBI (Order Book Imbalance) ∈ [-1, +1] :
  OBI = (bid_size_top - ask_size_top) / (bid_size_top + ask_size_top)
  >0 = pression acheteuse, <0 = pression vendeuse, ±0.5+ = signal court terme

spread_bps = (best_ask - best_bid) / mid * 10000
  Veto trade si spread > 5-10 bps anormal (default 10).

depth_at_bps(side, bps_distance) = somme des qty sur N niveaux dans la fourchette.
Capacity check avant un gros ordre.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BookSnapshot:
    bid_top_price: float
    ask_top_price: float
    bid_top_size: float
    ask_top_size: float
    mid: float
    spread_bps: float
    obi: float
    bid_depth_50bps: float    # somme qty bids dans 50 bps sous mid
    ask_depth_50bps: float    # somme qty asks dans 50 bps au-dessus mid


def parse_order_book(book: dict, depth_bps_threshold: float = 50.0) -> Optional[BookSnapshot]:
    """Construit un BookSnapshot à partir d'un return ccxt.fetch_order_book.

    book attendu: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
    bids triés DESC, asks triés ASC.

    Returns None si carnet vide.
    """
    bids = book.get('bids') or []
    asks = book.get('asks') or []
    if not bids or not asks:
        return None

    bid_p, bid_q = float(bids[0][0]), float(bids[0][1])
    ask_p, ask_q = float(asks[0][0]), float(asks[0][1])
    if bid_p <= 0 or ask_p <= 0 or bid_q <= 0 or ask_q <= 0:
        return None

    mid = (bid_p + ask_p) / 2.0
    spread_bps = (ask_p - bid_p) / mid * 10000.0

    obi = (bid_q - ask_q) / (bid_q + ask_q)

    # Depth dans X bps de la mid
    bid_thr = mid * (1 - depth_bps_threshold / 10000.0)
    ask_thr = mid * (1 + depth_bps_threshold / 10000.0)
    bid_depth = sum(float(q) for p, q in bids if float(p) >= bid_thr)
    ask_depth = sum(float(q) for p, q in asks if float(p) <= ask_thr)

    return BookSnapshot(
        bid_top_price=bid_p, ask_top_price=ask_p,
        bid_top_size=bid_q, ask_top_size=ask_q,
        mid=mid, spread_bps=spread_bps, obi=obi,
        bid_depth_50bps=bid_depth, ask_depth_50bps=ask_depth,
    )


def is_safe_to_trade(
    snap: Optional[BookSnapshot],
    max_spread_bps: float = 10.0,
    min_depth_usdt_per_side: float = 5000.0,
    min_price_for_usdt: Optional[float] = None,
) -> tuple[bool, str]:
    """Veto pré-trade: spread anormal ou liquidité insuffisante → bloquer.

    Args:
        snap: snapshot du carnet.
        max_spread_bps: spread au-delà duquel on refuse de trader (default 10 bps).
        min_depth_usdt_per_side: liquidité minimum à 50 bps de la mid en USDT par côté.
            Si min_price_for_usdt fourni, on convertit qty BTC → USDT via ce prix.
        min_price_for_usdt: prix BTC à utiliser pour la conversion qty→USDT.
            Default = snap.mid.

    Returns:
        (ok, reason). ok=True = trade autorisé.
    """
    if snap is None:
        return False, "order book empty"
    if snap.spread_bps > max_spread_bps:
        return False, f"spread {snap.spread_bps:.1f}bps > max {max_spread_bps:.1f}bps"
    price = min_price_for_usdt or snap.mid
    bid_depth_usdt = snap.bid_depth_50bps * price
    ask_depth_usdt = snap.ask_depth_50bps * price
    if bid_depth_usdt < min_depth_usdt_per_side:
        return False, f"bid depth 50bps {bid_depth_usdt:.0f} USDT < min {min_depth_usdt_per_side}"
    if ask_depth_usdt < min_depth_usdt_per_side:
        return False, f"ask depth 50bps {ask_depth_usdt:.0f} USDT < min {min_depth_usdt_per_side}"
    return True, "ok"


__all__ = ["BookSnapshot", "parse_order_book", "is_safe_to_trade"]
