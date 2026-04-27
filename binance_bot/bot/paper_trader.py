#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper trader: logge chaque trade live + simule le même trade avec coûts.

Permet de mesurer:
- Implementation shortfall = simulé_PnL - live_PnL (différence entre théorie et réalité)
- Tracking error = std des shortfalls

À chaque event live (open / close), on append une ligne CSV. Le rapport agrège.

Coûts par défaut (Binance USDM Futures, VIP 0):
- Taker fees: 0.04% (4 bps)
- Maker fees: 0.02% (2 bps)
- Funding rate moyen ≈ +0.01% par 8h (proxy si pas de data réelle)
- Slippage estimé: 2 bps + impact qty (cf B5 cost_model une fois dispo)

Note: la version v1 utilise des constantes hardcodées. Quand `src/cost_model.py`
sera dispo (B5 en cours), wrapper PaperTrader.compute_costs pour l'utiliser.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


CSV_HEADERS = [
    "timestamp_iso",
    "action",            # open_long / open_short / close_long / close_short
    "side",              # long / short
    "qty",
    "price",             # entry ou exit price (live, post-slippage)
    "live_order_id",     # ID retourné par Binance
    "fees_usdt",         # frais théoriques pour ce trade (signed: négatif = coût)
    "slippage_usdt",     # slippage estimé (signed: négatif = coût)
    "funding_usdt",      # funding accumulé sur la position fermée (0 si open)
    "live_pnl_usdt",     # P&L réalisé live (0 sur open, calculé sur close)
    "sim_pnl_usdt",      # P&L simulé (sans frais/slippage/funding)
    "shortfall_usdt",    # sim - live - fees - slippage - funding
    "signal_id",         # liens open ↔ close
    "notes",
]

# Constantes par défaut — à remplacer par src.cost_model quand B5 est intégré.
DEFAULT_TAKER_BPS = 4.0
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_FUNDING_PER_8H = 0.0001  # +0.01% par 8h (proxy bullish)


@dataclass(slots=True)
class PaperReport:
    """Agrégat de paper-trade events sur une fenêtre."""
    n_trades: int
    total_fees: float
    total_slippage: float
    total_funding: float
    total_live_pnl: float
    total_sim_pnl: float
    total_shortfall: float
    avg_shortfall_per_trade: float
    tracking_error: float
    open_positions: int  # nb d'opens sans close correspondant


class PaperTrader:
    """Persistance CSV append-only des trades live + leur miroir simulé."""

    def __init__(
        self,
        log_path: str | Path,
        taker_bps: float = DEFAULT_TAKER_BPS,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        funding_per_8h: float = DEFAULT_FUNDING_PER_8H,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.taker_bps = float(taker_bps)
        self.slippage_bps = float(slippage_bps)
        self.funding_per_8h = float(funding_per_8h)
        if not self.log_path.exists():
            self._write_headers()

    def _write_headers(self):
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)

    def compute_trade_cost(self, qty: float, price: float) -> tuple[float, float]:
        """Calcule (fees, slippage) en USDT (positive = coût)."""
        notional = abs(qty) * abs(price)
        fees = notional * (self.taker_bps / 10000.0)
        slippage = notional * (self.slippage_bps / 10000.0)
        return fees, slippage

    def compute_funding_cost(
        self,
        qty: float,
        price: float,
        position_side: str,  # 'long' ou 'short'
        held_seconds: float,
    ) -> float:
        """Approxime les funding payments sur la durée de la position.

        Long avec funding>0 paye au short.
        Funding rate proxy = self.funding_per_8h (positive = bullish bias).
        Retour positif = coût pour la position (short reçoit, long paye).
        """
        n_periods = held_seconds / (8 * 3600)
        notional = abs(qty) * abs(price)
        rate_total = self.funding_per_8h * n_periods
        if position_side == "long":
            return notional * rate_total  # long paye si rate > 0
        # short reçoit (donc coût négatif si rate > 0)
        return -notional * rate_total

    def log_open(
        self,
        action: str,                 # 'open_long' ou 'open_short'
        qty: float,
        price: float,
        live_order_id: Optional[str],
        signal_id: str,
        notes: str = "",
    ):
        """Enregistre l'ouverture d'une position."""
        side = "long" if "long" in action else "short"
        fees, slippage = self.compute_trade_cost(qty, price)
        # Pour l'open, P&L = 0 (pas encore réalisé), shortfall = -fees - slippage seulement.
        row = {
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "side": side,
            "qty": qty,
            "price": price,
            "live_order_id": live_order_id or "",
            "fees_usdt": -fees,
            "slippage_usdt": -slippage,
            "funding_usdt": 0.0,
            "live_pnl_usdt": 0.0,
            "sim_pnl_usdt": 0.0,
            "shortfall_usdt": -(fees + slippage),
            "signal_id": signal_id,
            "notes": notes,
        }
        self._append_row(row)

    def log_close(
        self,
        action: str,                 # 'close_long' ou 'close_short'
        qty: float,
        exit_price: float,
        entry_price: float,
        live_order_id: Optional[str],
        signal_id: str,
        held_seconds: float = 0.0,
        notes: str = "",
    ):
        """Enregistre la fermeture d'une position et calcule le P&L."""
        side = "long" if "long" in action else "short"
        fees, slippage = self.compute_trade_cost(qty, exit_price)
        funding = self.compute_funding_cost(qty, entry_price, side, held_seconds)
        # P&L brut (sans coûts) = sim
        if side == "long":
            sim_pnl = (exit_price - entry_price) * qty
        else:
            sim_pnl = (entry_price - exit_price) * qty
        # P&L réel = sim - fees - slippage - funding (estimés)
        live_pnl = sim_pnl - fees - slippage - funding
        shortfall = sim_pnl - live_pnl  # = fees + slippage + funding
        row = {
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "side": side,
            "qty": qty,
            "price": exit_price,
            "live_order_id": live_order_id or "",
            "fees_usdt": -fees,
            "slippage_usdt": -slippage,
            "funding_usdt": -funding,
            "live_pnl_usdt": live_pnl,
            "sim_pnl_usdt": sim_pnl,
            "shortfall_usdt": -shortfall,
            "signal_id": signal_id,
            "notes": notes,
        }
        self._append_row(row)

    def _append_row(self, row: dict):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

    def report(self) -> PaperReport:
        """Aggrège le log CSV en un PaperReport."""
        if not self.log_path.exists():
            return PaperReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        rows = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if not rows:
            return PaperReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        total_fees = sum(float(r["fees_usdt"]) for r in rows)
        total_slip = sum(float(r["slippage_usdt"]) for r in rows)
        total_funding = sum(float(r["funding_usdt"]) for r in rows)
        total_live = sum(float(r["live_pnl_usdt"]) for r in rows)
        total_sim = sum(float(r["sim_pnl_usdt"]) for r in rows)
        total_shortfall = sum(float(r["shortfall_usdt"]) for r in rows)

        # Pair open/close par signal_id pour mesurer tracking error
        opens = [r for r in rows if r["action"].startswith("open_")]
        closes = [r for r in rows if r["action"].startswith("close_")]
        open_ids = {r["signal_id"] for r in opens}
        closed_ids = {r["signal_id"] for r in closes}
        unmatched = open_ids - closed_ids

        # Tracking error: std des shortfalls par trade fermé
        close_shortfalls = [float(r["shortfall_usdt"]) for r in closes]
        if len(close_shortfalls) > 1:
            mean = sum(close_shortfalls) / len(close_shortfalls)
            var = sum((x - mean) ** 2 for x in close_shortfalls) / (len(close_shortfalls) - 1)
            tracking_error = var ** 0.5
        else:
            tracking_error = 0.0

        n_trades = len(closes)
        avg = total_shortfall / n_trades if n_trades else 0.0

        return PaperReport(
            n_trades=n_trades,
            total_fees=total_fees,
            total_slippage=total_slip,
            total_funding=total_funding,
            total_live_pnl=total_live,
            total_sim_pnl=total_sim,
            total_shortfall=total_shortfall,
            avg_shortfall_per_trade=avg,
            tracking_error=tracking_error,
            open_positions=len(unmatched),
        )


__all__ = ["PaperTrader", "PaperReport", "CSV_HEADERS"]
