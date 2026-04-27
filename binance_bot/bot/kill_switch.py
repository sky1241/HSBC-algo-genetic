#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kill switch: arrêt définitif du bot en cas de perte critique.

Quand un seuil global est franchi (equity <= seuil, ou erreurs API massives,
ou N pertes consécutives), le bot:
 1. Ferme toutes les positions ouvertes
 2. Annule tous les ordres en attente
 3. Crée un fichier .killed qui empêche les runs suivants jusqu'à reset manuel

Pour reactiver : `rm binance_bot/data/.killed`.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional


def is_killed(flag_path: Path) -> bool:
    """Retourne True si le kill flag existe → le bot doit refuser de tourner."""
    return Path(flag_path).exists()


def read_kill_reason(flag_path: Path) -> str:
    """Lit la raison du kill, ou '' si pas killable."""
    p = Path(flag_path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return "(reason unreadable)"


def trigger_kill(
    flag_path: Path,
    reason: str,
    state_mgr=None,
    trade_mgr=None,
    current_price: Optional[float] = None,
    capital_usdt: float = 0.0,
) -> List[str]:
    """Active le kill switch.

    Si trade_mgr est en mode 'live' et state_mgr fourni:
      - ferme toutes les positions long/short
      - annule tous les ordres ouverts (best effort)

    Écrit un fichier .killed avec le timestamp et la raison.

    Returns: liste des actions exécutées (pour log).
    """
    actions: List[str] = []
    flag_path = Path(flag_path)
    flag_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fermer positions et cancel orders (live uniquement)
    if trade_mgr is not None and getattr(trade_mgr, "mode", "simulation") == "live":
        if state_mgr is not None and current_price is not None:
            for pos in list(state_mgr.get("positions_long", []) or []):
                try:
                    oid = trade_mgr.execute_signal(
                        {"action": "close_long", "exit": current_price,
                         "reason": "kill_switch", "pos_id": pos.get("id", "")},
                        capital_usdt=capital_usdt,
                    )
                    actions.append(f"close_long order_id={oid}")
                except Exception as e:
                    actions.append(f"close_long FAILED: {e}")
            for pos in list(state_mgr.get("positions_short", []) or []):
                try:
                    oid = trade_mgr.execute_signal(
                        {"action": "close_short", "exit": current_price,
                         "reason": "kill_switch", "pos_id": pos.get("id", "")},
                        capital_usdt=capital_usdt,
                    )
                    actions.append(f"close_short order_id={oid}")
                except Exception as e:
                    actions.append(f"close_short FAILED: {e}")
        # Cancel any leftover open orders
        try:
            trade_mgr.exchange.cancel_all_orders(trade_mgr.symbol)
            actions.append(f"cancel_all_orders({trade_mgr.symbol}) ok")
        except Exception as e:
            actions.append(f"cancel_all_orders FAILED: {e}")

    # 2. Marquer le bot comme killed
    body = (
        f"KILLED at {datetime.utcnow().isoformat()}Z\n"
        f"reason: {reason}\n"
        f"actions:\n"
        + "\n".join(f"  - {a}" for a in actions)
        + f"\n\nTo re-arm: rm {flag_path}\n"
    )
    flag_path.write_text(body, encoding="utf-8")
    return actions
