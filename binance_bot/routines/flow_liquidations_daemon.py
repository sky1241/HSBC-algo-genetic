#!/usr/bin/env python3
"""R3 — Flow liquidations WS daemon (P6.3) — systemd long-running service.

Maintient une connexion WebSocket persistante à wss://fstream.binance.com/ws/
!forceOrder@arr (toutes les liquidations Futures) et agrège par bucket 1min
× side. Écrit data/flow_liq_buckets.jsonl à chaque flush.

Lancé par hsbc-flow-liquidations.service (Type=simple, Restart=always).

Pas de timer : le service tourne 24/7. Reconnect auto avec backoff exponentiel
intégré dans LiquidationWSManager.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_liquidations import (
    LiquidationAggregator,
    LiquidationWSManager,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
STORE = ROOT / "data" / "flow_liq_buckets.jsonl"
HEARTBEAT_INTERVAL_SEC = 300  # log toutes les 5min

logger = logging.getLogger("flow_liquidations_daemon")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> int:
    aggregator = LiquidationAggregator(symbols=set(SYMBOLS))
    manager = LiquidationWSManager(aggregator=aggregator, store_path=STORE)

    stop_requested = {"flag": False}

    def _handle_sigterm(signum, frame):
        logger.info(f"received signal {signum} — shutting down WS")
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    logger.info(f"starting WS daemon for {SYMBOLS} → {STORE}")
    manager.start()

    last_heartbeat = time.time()
    try:
        while not stop_requested["flag"]:
            time.sleep(1.0)
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                logger.info(
                    f"heartbeat events_received={manager.events_received} "
                    f"buckets_flushed={manager.buckets_flushed} "
                    f"connection_errors={manager.connection_errors}"
                )
                last_heartbeat = now
    finally:
        manager.stop(drain_to_jsonl=True)
        logger.info("daemon exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
