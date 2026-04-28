#!/usr/bin/env python3
"""P11 — Funding-aware close runner (systemd timer entrypoint).

Déclenché par hsbc-funding-close.timer toutes les minutes entre HH:55 et HH:59
pour HH ∈ {0, 8, 16} UTC. Il :

  1. Charge les positions ouvertes depuis state.json (BTC, ETH, SOL).
  2. Pour chaque symbole, fetch le funding rate courant via Binance.
  3. Évalue les signaux de close via funding_close.evaluate_close_signals().
  4. Log les décisions dans logs/funding_close.log (dry-run par défaut).

DRY-RUN par défaut : on log les décisions mais on N'EXÉCUTE PAS encore le
flat live. Pour activer l'exécution réelle, set env var
HSBC_FUNDING_CLOSE_LIVE=1 (et brancher l'OrderClient).

Cf. spec P11.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ajout binance_bot/ au path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.funding_close import evaluate_close_signals, is_settlement_window

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
    """Aplatit state.json multi-symbole en liste {id, side, symbol}."""
    out: list[dict] = []
    for symbol, blob in (state.get("symbols") or {}).items():
        for pos in blob.get("positions_long", []) or []:
            out.append({"id": pos.get("id"), "side": "long", "symbol": symbol})
        for pos in blob.get("positions_short", []) or []:
            out.append({"id": pos.get("id"), "side": "short", "symbol": symbol})
    return out


def _fetch_funding_rate_bps(symbol: str) -> float | None:
    """Tente de fetch le funding rate courant via Binance Futures premiumIndex.

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
        # lastFundingRate est un fraction (e.g. 0.0001 = 1 bp)
        rate = float(data.get("lastFundingRate", 0.0))
        return rate * 10000.0  # → bps
    except Exception as e:
        logger.warning(f"premiumIndex {sym} failed: {e}")
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

    # Évalue par symbole (funding rate symbol-spécifique)
    by_symbol: dict[str, list[dict]] = {}
    for p in positions:
        by_symbol.setdefault(p["symbol"], []).append(p)

    live_mode = os.environ.get("HSBC_FUNDING_CLOSE_LIVE", "0") == "1"
    all_signals: list[dict] = []
    for symbol, syms_pos in by_symbol.items():
        funding_bps = _fetch_funding_rate_bps(symbol)
        if funding_bps is None:
            logger.warning(f"[skip] {symbol}: funding rate unavailable")
            continue
        signals = evaluate_close_signals(
            now=now,
            positions=syms_pos,
            funding_rate_bps=funding_bps,
        )
        for s in signals:
            logger.info(
                f"{'LIVE' if live_mode else 'DRY-RUN'} | "
                f"{symbol} pos_id={s['pos_id']} side={s['side']} "
                f"funding={funding_bps:.2f}bps | {s['reason']}"
            )
            all_signals.append(s)

    if not all_signals:
        logger.info(f"[ok] aucun close signal à {now.isoformat()}")
    elif live_mode:
        # TODO Sky : brancher le client de close réel ici (out of P11 scope).
        logger.warning(f"LIVE MODE non encore branché — {len(all_signals)} signaux ignorés")

    return 0


if __name__ == "__main__":
    sys.exit(main())
