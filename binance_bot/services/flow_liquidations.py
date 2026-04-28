"""P6.3 — Binance Futures !forceOrder@arr WebSocket + agrégation buckets 1min.

WebSocket endpoint Binance USDM Futures (vérifié 27/04/2026) :
    wss://fstream.binance.com/ws/!forceOrder@arr

Stream de TOUTES les liquidations forcées tous symboles (push ~1000ms,
1 event par symbole et par fenêtre). Format event :
    {
      "e": "forceOrder",
      "E": <event_time_ms>,
      "o": {
        "s": "BTCUSDT",  # symbol
        "S": "SELL",     # side : BUY = liquidation des shorts, SELL = liquidation des longs
        "q": "0.014",    # quantity
        "p": "9910",     # price
        "ap": "9910",    # average price
        "T": <trade_time_ms>,
        ...
      }
    }

Implémentation:
    - Pure functions: parse_force_order_event, LiquidationAggregator (buckets 1min)
    - WSManager wrapper sync (pattern binance_bot/bot/user_stream.py),
      websocket-client lib + reconnect backoff exponentiel
    - Storage data/flow_liq_buckets.jsonl append-only (1 ligne par bucket complet)

Note sémantique :
    Sur Binance futures, un event side="SELL" = liquidation forcée d'une
    position LONG (le moteur force la vente). side="BUY" = liquidation d'une
    SHORT. On stocke `liq_long_notional` (= sum SELL events) et `liq_short_notional`
    (= sum BUY events) pour cohérence avec l'usage P6.5 composite_signal.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

logger = logging.getLogger(__name__)

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None  # tests mockent websocket.WebSocketApp


_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
_BUCKET_SIZE_MS = 60_000  # 1 minute
_RECONNECT_BASE_SEC = 1.0
_RECONNECT_MAX_SEC = 60.0


def parse_force_order_event(msg: Union[str, Mapping[str, Any]]) -> Optional[dict]:
    """Convertit un event Binance forceOrder en record normalisé.

    Args:
        msg: JSON string ou dict déjà parsé.

    Returns:
        dict {ts_ms, symbol, side, qty, price, notional} ou None si malformé.
        side ∈ {"long_liq", "short_liq"} (= traduction sémantique de
        SELL/BUY Binance, voir docstring module).
    """
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(msg, Mapping):
        return None
    if msg.get("e") != "forceOrder":
        return None
    o = msg.get("o")
    if not isinstance(o, Mapping):
        return None
    try:
        symbol = str(o["s"])
        side_raw = str(o["S"]).upper()
        qty = float(o["q"])
        price = float(o.get("ap", o.get("p", 0)))
        ts_ms = int(o.get("T", msg.get("E", 0)))
    except (KeyError, TypeError, ValueError):
        return None
    if qty <= 0 or price <= 0 or ts_ms <= 0:
        return None
    # SELL = liquidation d'un LONG (le moteur force vente du long)
    side_label = "long_liq" if side_raw == "SELL" else "short_liq" if side_raw == "BUY" else None
    if side_label is None:
        return None
    return {
        "ts_ms": ts_ms,
        "symbol": symbol,
        "side": side_label,
        "qty": qty,
        "price": price,
        "notional": float(qty) * float(price),
    }


def bucket_key_ms(ts_ms: int, bucket_size_ms: int = _BUCKET_SIZE_MS) -> int:
    """Floor du ts au début du bucket (1min par défaut)."""
    return int(ts_ms) - (int(ts_ms) % int(bucket_size_ms))


@dataclass
class LiquidationBucket:
    """Agrégat 1min de liquidations pour un symbole."""
    bucket_start_ms: int
    symbol: str
    long_liq_notional: float = 0.0
    short_liq_notional: float = 0.0
    long_liq_count: int = 0
    short_liq_count: int = 0

    def add_event(self, record: Mapping[str, Any]) -> None:
        notional = float(record.get("notional", 0.0))
        if record.get("side") == "long_liq":
            self.long_liq_notional += notional
            self.long_liq_count += 1
        elif record.get("side") == "short_liq":
            self.short_liq_notional += notional
            self.short_liq_count += 1

    def to_dict(self) -> dict:
        return {
            "bucket_start_ms": self.bucket_start_ms,
            "symbol": self.symbol,
            "long_liq_notional": self.long_liq_notional,
            "short_liq_notional": self.short_liq_notional,
            "long_liq_count": self.long_liq_count,
            "short_liq_count": self.short_liq_count,
        }


class LiquidationAggregator:
    """Stateful : accumule les events par (symbol, bucket_ms), flush quand
    on observe un event d'un bucket plus récent (= le précédent est complet).

    Thread-safe via lock interne (le WS callback peut être appelé depuis un
    autre thread).
    """

    def __init__(self, symbols: Optional[set[str]] = None,
                 bucket_size_ms: int = _BUCKET_SIZE_MS):
        self._buckets: dict[tuple[str, int], LiquidationBucket] = {}
        self._lock = threading.Lock()
        self._symbols_filter = set(s.upper() for s in symbols) if symbols else None
        self._bucket_size_ms = int(bucket_size_ms)

    def ingest(self, record: Mapping[str, Any]) -> list[dict]:
        """Ingest 1 event normalisé. Retourne les buckets devenus complets.

        Un bucket est considéré complet quand on observe un event d'un bucket
        strictement plus récent (= le previous a été closed). Approche simple
        et efficace; le drain final (sur shutdown) doit être appelé à part.
        """
        symbol = str(record.get("symbol", "")).upper()
        if not symbol:
            return []
        if self._symbols_filter is not None and symbol not in self._symbols_filter:
            return []
        ts_ms = int(record.get("ts_ms", 0))
        if ts_ms <= 0:
            return []
        bucket_start = bucket_key_ms(ts_ms, self._bucket_size_ms)
        completed: list[dict] = []
        with self._lock:
            # Flush tous les buckets strictement antérieurs à ce bucket pour ce symbole
            keys_to_flush = [
                k for k in self._buckets
                if k[0] == symbol and k[1] < bucket_start
            ]
            for k in keys_to_flush:
                completed.append(self._buckets.pop(k).to_dict())
            # Append au bucket courant
            current_key = (symbol, bucket_start)
            if current_key not in self._buckets:
                self._buckets[current_key] = LiquidationBucket(
                    bucket_start_ms=bucket_start, symbol=symbol
                )
            self._buckets[current_key].add_event(record)
        return completed

    def drain(self) -> list[dict]:
        """Flush tous les buckets en mémoire (à appeler en fin de session)."""
        with self._lock:
            out = [b.to_dict() for b in self._buckets.values()]
            self._buckets.clear()
        return out


def append_buckets_jsonl(buckets: list[dict], path: Path) -> int:
    """Append 1 ligne par bucket dans le jsonl. Retourne nb d'écritures."""
    if not buckets:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for b in buckets:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# WSManager wrapper (pattern user_stream.py)
# ---------------------------------------------------------------------------


class LiquidationWSManager:
    """WebSocket manager pour !forceOrder@arr avec reconnect.

    Pattern aligné sur binance_bot/bot/user_stream.py : thread daemon,
    websocket-client lib, backoff exponentiel sur déconnexion.

    Usage:
        agg = LiquidationAggregator(symbols={"BTCUSDT"})
        mgr = LiquidationWSManager(
            aggregator=agg,
            store_path=Path("data/flow_liq_buckets.jsonl"),
        )
        mgr.start()
        # ... runs in background thread ...
        mgr.stop()
    """

    def __init__(
        self,
        aggregator: LiquidationAggregator,
        store_path: Path,
        ws_url: str = _WS_URL,
    ):
        self.aggregator = aggregator
        self.store_path = store_path
        self.ws_url = ws_url
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.connection_errors = 0
        self.events_received = 0
        self.buckets_flushed = 0

    def _on_message(self, ws, raw: str) -> None:
        self.events_received += 1
        record = parse_force_order_event(raw)
        if record is None:
            return
        completed = self.aggregator.ingest(record)
        if completed:
            self.buckets_flushed += append_buckets_jsonl(completed, self.store_path)

    def _on_error(self, ws, err: Any) -> None:
        self.connection_errors += 1
        logger.warning("flow_liquidations WS error: %s", err)

    def _on_close(self, ws, code: Any, reason: Any) -> None:
        logger.info("flow_liquidations WS closed code=%s reason=%s", code, reason)

    def _run_loop(self) -> None:
        backoff = _RECONNECT_BASE_SEC
        while not self._stop_event.is_set():
            if websocket is None:
                logger.error("websocket-client not installed — flow_liquidations disabled")
                return
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever()
            except Exception as e:
                logger.error("flow_liquidations WS run_forever crashed: %s", e)
            if self._stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SEC)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, drain_to_jsonl: bool = True) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if drain_to_jsonl:
            remaining = self.aggregator.drain()
            self.buckets_flushed += append_buckets_jsonl(remaining, self.store_path)


__all__ = [
    "parse_force_order_event",
    "bucket_key_ms",
    "LiquidationBucket",
    "LiquidationAggregator",
    "LiquidationWSManager",
    "append_buckets_jsonl",
]
