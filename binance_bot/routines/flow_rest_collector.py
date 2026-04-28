#!/usr/bin/env python3
"""R3 — Flow REST collector (P6.1, P6.2, P6.4) — systemd oneshot every 5min.

Lit BTCUSDT/ETHUSDT/SOLUSDT × 3 services REST Binance Futures
(topLongShortPositionRatio, takerlongshortRatio, openInterestHist) et écrit
les nouveaux records en append-dedup dans data/flow_*.jsonl.

Lancé par hsbc-flow-rest.timer (oneshot, OnCalendar=*:0/5).

Le module est tolérant aux pannes par design :
- erreur réseau ou rate limit sur 1 endpoint → logue, ne casse pas les autres
- exit code 0 même en cas d'erreur partielle (le timer re-fire dans 5min)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_top_ls import poll_and_store as poll_top_ls
from services.flow_taker_ratio import poll_and_store as poll_taker
from services.flow_open_interest import poll_and_store as poll_oi


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

logger = logging.getLogger("flow_rest_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _data_path(name: str) -> Path:
    return ROOT / "data" / name


def _safe_poll(label: str, fn, symbol: str, path: Path) -> int:
    """Wrapper qui catch toute exception et continue les autres polls."""
    try:
        n = fn(symbol=symbol, store_path=path)
        logger.info(f"[{label}] {symbol} +{n} new records → {path.name}")
        return int(n)
    except Exception as e:
        logger.warning(f"[{label}] {symbol} failed: {type(e).__name__}: {e}")
        return 0


def main() -> int:
    total = 0
    for sym in SYMBOLS:
        total += _safe_poll(
            "top_ls", poll_top_ls, sym,
            _data_path(f"flow_top_ls_{sym}.jsonl"),
        )
        total += _safe_poll(
            "taker", poll_taker, sym,
            _data_path(f"flow_taker_{sym}.jsonl"),
        )
        total += _safe_poll(
            "open_interest", poll_oi, sym,
            _data_path(f"flow_oi_{sym}.jsonl"),
        )
    logger.info(f"flow_rest_collector done: {total} new records across "
                f"{len(SYMBOLS)} symbols × 3 services")
    return 0


if __name__ == "__main__":
    sys.exit(main())
