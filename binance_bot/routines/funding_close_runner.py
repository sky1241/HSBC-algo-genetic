#!/usr/bin/env python3
"""P11 / R9 — Funding-aware close runner (systemd timer entrypoint).

Déclenché par `hsbc-funding-close.timer` toutes les minutes entre HH:55 et HH:59
pour HH ∈ {0, 8, 16} UTC. Workflow :

  1. Charge les positions ouvertes depuis `data/state.json` (BTC, ETH, SOL).
  2. Pour chaque symbole, fetch le funding rate courant via Binance.
  3. Évalue les signaux de close via `funding_close.evaluate_close_signals()`.
  4. **MODE DRY-RUN par défaut** : log seulement, aucun ordre envoyé.
  5. **MODE LIVE** (env var `HSBC_FUNDING_CLOSE_LIVE=1`) : émet les close
     via `TradeManager.execute_signal({"action": "close_long", ...})`.

Safety
------
- DRY-RUN par défaut (variable d'env requise pour activer le live).
- Trade mode forcé à `bot_settings.yaml::trade_mode` (héritage). Si
  `trade_mode=simulation`, le TradeManager ne place pas de vrais ordres
  même avec `HSBC_FUNDING_CLOSE_LIVE=1` set.
- Audit log de chaque décision (event=`funding_close_decision`).
- Si la fenêtre n'est pas active → skip immédiat (pas de fetch coûteux).

Activation systemd
------------------
    systemctl --user link binance_bot/systemd/hsbc-funding-close.timer
    systemctl --user link binance_bot/systemd/hsbc-funding-close.service
    systemctl --user enable --now hsbc-funding-close.timer
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ajout binance_bot/ au path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from services.funding_close import evaluate_close_signals, is_settlement_window
from bot.audit_log import AuditLog

logger = logging.getLogger("funding_close_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"failed to load state: {e}")
        return {}


def _flatten_positions(state: dict) -> list[dict]:
    """Aplatit state.json multi-symbole en liste {id, side, symbol, entry, size}."""
    out: list[dict] = []
    for symbol, blob in (state.get("symbols") or {}).items():
        for pos in blob.get("positions_long", []) or []:
            out.append({
                "id": pos.get("id"),
                "side": "long",
                "symbol": symbol,
                "entry": float(pos.get("entry", 0.0)),
                "size": float(pos.get("size", 0.0)),
            })
        for pos in blob.get("positions_short", []) or []:
            out.append({
                "id": pos.get("id"),
                "side": "short",
                "symbol": symbol,
                "entry": float(pos.get("entry", 0.0)),
                "size": float(pos.get("size", 0.0)),
            })
    return out


def _fetch_funding_rate_bps(symbol: str) -> Optional[float]:
    """Fetch live funding rate via Binance Futures premiumIndex.

    Retourne le funding rate en bps (1 bp = 0.01 %). None si API indispo.
    """
    try:
        import requests
    except ImportError:
        return None
    sym = symbol.replace("/", "")
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": sym},
            timeout=5,
        )
        if r.status_code != 200:
            logger.warning(f"premiumIndex {sym}: HTTP {r.status_code}")
            return None
        data = r.json()
        rate = float(data.get("lastFundingRate", 0.0))
        return rate * 10000.0  # → bps
    except Exception as e:
        logger.warning(f"premiumIndex {sym} failed: {e}")
        return None


def _build_trade_manager(symbol: str, settings: dict, trade_mode: str):
    """Construit TradeManager pour un symbole. Retourne None si non dispo."""
    try:
        from services.data_fetcher import DataFetcher  # type: ignore
        from bot.trade_manager import TradeManager  # type: ignore
    except ImportError as e:
        logger.warning(f"TradeManager import failed: {e}")
        return None
    # Leverage du symbole depuis settings
    leverage = 1.0
    for sym_cfg in settings.get("symbols", []):
        if sym_cfg.get("pair") == symbol:
            leverage = float(sym_cfg.get("leverage", 1.0))
            break
    try:
        data_fetcher = DataFetcher(symbol=symbol, timeframe="1h")
        # Auto-detect position mode (testnet généralement oneway)
        pos_mode = "oneway"
        try:
            info = data_fetcher.exchange.fapiPrivateGetPositionSideDual()
            if info.get("dualSidePosition"):
                pos_mode = "hedge"
        except Exception:
            pass
        return TradeManager(
            exchange=data_fetcher.exchange,
            symbol=symbol,
            mode=trade_mode,
            leverage=leverage,
            position_mode=pos_mode,
        )
    except Exception as e:
        logger.warning(f"build TradeManager {symbol}: {e}")
        return None


def _execute_close(signal: dict, position: dict, settings: dict,
                   trade_mode: str, audit) -> Optional[str]:
    """Émet un close réel via TradeManager si live mode + autorisé.

    Returns:
        order_id string si exécuté, None sinon (dry-run / failed).
    """
    symbol = position["symbol"]
    side = position["side"]
    action = "close_long" if side == "long" else "close_short"

    trade_mgr = _build_trade_manager(symbol, settings, trade_mode)
    if trade_mgr is None:
        return None
    close_signal = {
        "action": action,
        "pos_id": position.get("id"),
        "exit": float(position.get("entry", 0.0)),  # exit price approximatif
        "reason": signal.get("reason", "funding_close"),
        "size": float(position.get("size", 0.01)),
    }
    try:
        order_id = trade_mgr.execute_signal(close_signal, capital_usdt=100.0)
        audit.append({
            "event": "funding_close_executed",
            "symbol": symbol, "side": side,
            "pos_id": position.get("id"),
            "order_id": order_id,
            "trade_mode": trade_mode,
            "reason": signal.get("reason"),
        })
        return order_id
    except Exception as e:
        logger.error(f"execute close {symbol}: {e}")
        audit.append({
            "event": "funding_close_failed",
            "symbol": symbol, "side": side,
            "pos_id": position.get("id"),
            "error": str(e),
        })
        return None


def main() -> int:
    now = datetime.now(timezone.utc)
    if not is_settlement_window(now, advance_minutes=5):
        logger.info(f"[skip] {now.isoformat()} hors fenêtre settlement")
        return 0

    state_path = ROOT / "data" / "state.json"
    state = _load_state(state_path)
    positions = _flatten_positions(state)
    if not positions:
        logger.info(f"[skip] no open positions at {now.isoformat()}")
        return 0

    # Settings (trade_mode, leverages, thresholds)
    cfg_path = ROOT / "configs" / "bot_settings.yaml"
    try:
        settings = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"settings load failed: {e}")
        settings = {}
    trade_mode = settings.get("trade_mode", "simulation")
    threshold_long = float(settings.get("funding_close_threshold_long_bps", 5.0))
    threshold_short = float(settings.get("funding_close_threshold_short_bps", 5.0))
    advance_min = int(settings.get("funding_close_advance_minutes", 5))

    # Live mode = DRY-RUN par défaut (env var requise + trade_mode != simulation)
    live_env = os.environ.get("HSBC_FUNDING_CLOSE_LIVE", "0") == "1"
    live_mode = live_env and trade_mode == "live"

    audit = AuditLog(ROOT / "data" / "trades_audit.jsonl")
    audit.append({
        "event": "funding_close_runner_invoked",
        "ts": now.isoformat(),
        "live_env": live_env,
        "trade_mode": trade_mode,
        "live_mode_effective": live_mode,
        "n_positions": len(positions),
    })

    # Évaluer par symbole (funding rate symbol-spécifique)
    by_symbol: dict[str, list[dict]] = {}
    for p in positions:
        by_symbol.setdefault(p["symbol"], []).append(p)

    n_signals = 0
    n_executed = 0
    for symbol, syms_pos in by_symbol.items():
        funding_bps = _fetch_funding_rate_bps(symbol)
        if funding_bps is None:
            logger.warning(f"[skip] {symbol}: funding rate unavailable")
            continue
        signals = evaluate_close_signals(
            now=now,
            positions=syms_pos,
            funding_rate_bps=funding_bps,
            threshold_long_bps=threshold_long,
            threshold_short_bps=threshold_short,
            advance_minutes=advance_min,
        )
        for sig in signals:
            n_signals += 1
            mode_tag = "LIVE" if live_mode else "DRY-RUN"
            logger.info(
                f"{mode_tag} | {symbol} pos_id={sig['pos_id']} side={sig['side']} "
                f"funding={funding_bps:.2f}bps | {sig['reason']}"
            )
            audit.append({
                "event": "funding_close_decision",
                "symbol": symbol,
                "side": sig["side"],
                "pos_id": sig["pos_id"],
                "funding_rate_bps": funding_bps,
                "minutes_to_settlement": sig["minutes_to_settlement"],
                "reason": sig["reason"],
                "live_mode": live_mode,
            })
            if live_mode:
                # Trouve la position correspondante pour le close
                pos = next((p for p in syms_pos if p.get("id") == sig["pos_id"]), None)
                if pos is None:
                    logger.warning(f"position {sig['pos_id']} introuvable")
                    continue
                order_id = _execute_close(sig, pos, settings, trade_mode, audit)
                if order_id:
                    n_executed += 1

    logger.info(
        f"[done] signals={n_signals} executed={n_executed} mode="
        f"{'LIVE' if live_mode else 'DRY-RUN'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
