"""P7-bis — Collecteur VPIN live (Binance Futures aggTrade + bookTicker WS).

Résout L-001 (VPIN data_fn placeholder None) en fournissant des
(vpin, obi) live au gate VPIN existant via WebSocket Binance Futures.

WS-001 mitigation
-----------------
@aggTrade ne fonctionne PAS sur fstream.binance.com (Binance close
frame post-SUBSCRIBE silent, cf BUGS.md). On utilise <symbol>@trade à
la place. Sémantiquement équivalent pour calcul VPIN — chaque trade
individuel au lieu d'agrégé par price+side+time, BVC fonctionne pareil.

Architecture
------------
- VPINLiveBuilder(symbol, bucket_size_v, window=50) : pure logic.
  Maintient buffer de trades + buckets via src/vpin.build_volume_buckets.
  Méthode add_trade(price, qty, ts_ms). current_vpin() returns last VPIN.

- OBIBuilder(symbol) : pure logic.
  Maintient bid_qty/ask_qty derniers reçus (bookTicker stream).
  Convention : OBI ∈ [0, 1] où 1 = parfaitement balanced, 0 = un côté
  domine 100%. Formule : OBI = 1 - |bid - ask| / (bid + ask).
  Edge case bid+ask=0 → 0.5 (default safe).

- VPINLiveCollectorWS : orchestre 2 WS (@trade + @bookTicker) par
  symbole via threading. Reconnect backoff (pattern
  LiquidationWSManager). Flush vers data/vpin_live.jsonl à intervalle
  configurable.

Référence
---------
Easley, López de Prado, O'Hara (2012). "The Volume Clock", JPM 39(1)
19-29. BVC + Volume Clock buckets V/N (N=50).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# WS lib (déjà dans requirements.txt R-FIX-FORGE)
try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False
    websocket = None  # type: ignore

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


_WS_BASE_URL = "wss://fstream.binance.com/ws"
_RECONNECT_BASE_SEC = 1.0
_RECONNECT_MAX_SEC = 60.0
_DEFAULT_FLUSH_INTERVAL_SEC = 30  # flush jsonl every 30s


# ---------------------------------------------------------------------------
# Pure logic : VPINLiveBuilder
# ---------------------------------------------------------------------------


class VPINLiveBuilder:
    """Maintient buffer de trades et calcule VPIN sur les buckets accumulés.

    Args:
        symbol: BTCUSDT / ETHUSDT / SOLUSDT (pour traçabilité).
        bucket_size_v: target volume par bucket en quote currency
            (USDT). Doit être V_daily_avg / N=50 selon Easley 2012.
        window: nombre de buckets dans la fenêtre VPIN (default 50).
    """

    def __init__(self, symbol: str, bucket_size_v: float, window: int = 50):
        self.symbol = str(symbol).upper()
        self.bucket_size_v = float(bucket_size_v)
        self.window = int(window)
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._ts_ms: list[int] = []
        self._cached_vpin: float = 0.0
        self._n_buckets_last: int = 0

    def add_trade(self, price: float, qty: float, ts_ms: int) -> None:
        """Ajoute un trade au buffer. ts_ms = timestamp Binance event time.

        BUG-FIX 2026-04-29 (L-007 unit mismatch) : `bucket_size_v` est
        documenté en QUOTE currency (USDT) — formule Easley 2012 V/N
        sur quote_volume klines. Mais le payload Binance @trade fournit
        `q` en BASE currency (BTC pour BTCUSDT). On stocke donc le
        notional `p * q` (USDT) pour cohérence d'unité avec le seuil
        de fermeture de bucket dans build_volume_buckets.
        VPIN value invariante (compute_vpin = mean(|buy-sell|/total),
        ratios indépendants de l'unité), seule la VITESSE de fermeture
        de bucket change (avant : ~28 jours/bucket, après : ~28 min/bucket
        sur BTC mainnet flow normal).
        """
        try:
            p = float(price)
            q = float(qty)
        except (TypeError, ValueError):
            return
        if p <= 0 or q <= 0:
            return
        self._prices.append(p)
        self._volumes.append(p * q)  # notional USDT (cf docstring)
        self._ts_ms.append(int(ts_ms))

    def current_vpin(self) -> float:
        """Calcule VPIN sur les buckets actuels via src.vpin.

        Retourne 0.0 si pas assez de buckets.
        """
        if not self._prices or self.bucket_size_v <= 0:
            return 0.0
        try:
            from src.vpin import build_volume_buckets, compute_vpin  # type: ignore
        except ImportError:
            try:
                from vpin import build_volume_buckets, compute_vpin  # type: ignore
            except ImportError:
                return 0.0
        prices = pd.Series(self._prices)
        volumes = pd.Series(self._volumes)
        buckets = build_volume_buckets(prices, volumes,
                                        bucket_size_v=self.bucket_size_v)
        complete = [b for b in buckets if not b.get("incomplete", False)]
        self._n_buckets_last = len(complete)
        vpin = compute_vpin(buckets, window=self.window)
        self._cached_vpin = float(vpin) if np.isfinite(vpin) else 0.0
        return self._cached_vpin

    def n_buckets(self) -> int:
        """Nombre de buckets COMPLETS au dernier appel current_vpin()."""
        return self._n_buckets_last

    def n_trades(self) -> int:
        return len(self._prices)


# ---------------------------------------------------------------------------
# Pure logic : OBIBuilder
# ---------------------------------------------------------------------------


class OBIBuilder:
    """Order Book Imbalance basé sur Binance bookTicker stream.

    Convention IMPOSÉE par src/vpin_gate.py existant :
        OBI ∈ [0, 1] où 1 = parfaitement balanced (bid_qty ≈ ask_qty),
        0 = un côté domine 100% (cascade imminente).
    Formule :
        OBI = 1 - |bid_qty - ask_qty| / (bid_qty + ask_qty)
    Edge case bid+ask=0 (impossible en pratique) → 0.5 (default safe).
    """

    def __init__(self, symbol: str):
        self.symbol = str(symbol).upper()
        self.bid_qty: float = 0.0
        self.ask_qty: float = 0.0
        self.last_update_ms: int = 0

    def update(self, bid_qty: float, ask_qty: float, ts_ms: int = 0) -> None:
        try:
            self.bid_qty = float(bid_qty)
            self.ask_qty = float(ask_qty)
        except (TypeError, ValueError):
            return
        if self.bid_qty < 0 or self.ask_qty < 0:
            self.bid_qty = max(0.0, self.bid_qty)
            self.ask_qty = max(0.0, self.ask_qty)
        if ts_ms:
            self.last_update_ms = int(ts_ms)

    def current_obi(self) -> float:
        total = self.bid_qty + self.ask_qty
        if total <= 0:
            return 0.5  # default safe (book vide ou non encore reçu)
        diff = abs(self.bid_qty - self.ask_qty)
        obi = 1.0 - (diff / total)
        # Clip [0, 1] strict pour cohérence vpin_gate
        return float(max(0.0, min(1.0, obi)))


# ---------------------------------------------------------------------------
# Payload parsers (Binance Futures @trade + @bookTicker)
# ---------------------------------------------------------------------------


def parse_trade_event(msg: Union[str, Mapping[str, Any]]) -> Optional[dict]:
    """Parse un message Binance Futures @trade.

    Payload attendu (testé live 2026-04-28T17:55Z) :
        {"e":"trade","E":<ms>,"T":<ms>,"s":"BTCUSDT","t":<id>,
         "p":"<price>","q":"<qty>","X":"MARKET","m":<bool>}
    Retour : {symbol, price, qty, ts_ms} ou None si invalide.
    """
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except (ValueError, json.JSONDecodeError):
            return None
    if not isinstance(msg, Mapping):
        return None
    if msg.get("e") != "trade":
        return None
    try:
        return {
            "symbol": str(msg["s"]).upper(),
            "price": float(msg["p"]),
            "qty": float(msg["q"]),
            "ts_ms": int(msg.get("T") or msg.get("E") or 0),
        }
    except (KeyError, TypeError, ValueError):
        return None


def parse_book_ticker_event(msg: Union[str, Mapping[str, Any]]) -> Optional[dict]:
    """Parse un message Binance Futures @bookTicker.

    Payload attendu (testé live 2026-04-28T17:55Z) :
        {"e":"bookTicker","u":<id>,"s":"BTCUSDT","b":"<bid_px>",
         "B":"<bid_qty>","a":"<ask_px>","A":"<ask_qty>",
         "T":<ms>,"E":<ms>}
    Retour : {symbol, bid_qty, ask_qty, ts_ms} ou None.
    """
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except (ValueError, json.JSONDecodeError):
            return None
    if not isinstance(msg, Mapping):
        return None
    # bookTicker can have e='bookTicker' or no 'e' field (older format)
    if msg.get("e") not in (None, "bookTicker"):
        return None
    if "B" not in msg or "A" not in msg:
        return None
    try:
        return {
            "symbol": str(msg["s"]).upper(),
            "bid_qty": float(msg["B"]),
            "ask_qty": float(msg["A"]),
            "ts_ms": int(msg.get("T") or msg.get("E") or 0),
        }
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Persistance JSONL
# ---------------------------------------------------------------------------


def append_vpin_record(path: Path, record: dict) -> int:
    """Append 1 record au jsonl. Retourne 1 si écrit, 0 sinon."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return 1
    except Exception as e:
        logger.warning(f"append_vpin_record failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# WS Collector orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _WSState:
    """État interne pour un symbole."""
    vpin_builder: VPINLiveBuilder
    obi_builder: OBIBuilder
    last_flush_sec: float = 0.0
    events_received: int = 0
    connection_errors: int = 0
    trade_thread: Optional[threading.Thread] = None
    book_thread: Optional[threading.Thread] = None
    trade_ws: Any = None
    book_ws: Any = None


class VPINLiveCollectorWS:
    """Orchestre 2 WS (@trade + @bookTicker) par symbole, flush jsonl.

    Pattern reconnect aligné sur LiquidationWSManager. Threading par
    symbol × stream (2 threads par symbol). SIGTERM stop drain proper.
    """

    def __init__(
        self,
        symbols: list[str],
        bucket_sizes: dict[str, float],
        store_path: Path,
        ws_base_url: str = _WS_BASE_URL,
        flush_interval_sec: int = _DEFAULT_FLUSH_INTERVAL_SEC,
    ):
        self.symbols = [str(s).upper() for s in symbols]
        self.bucket_sizes = {str(k).upper(): float(v) for k, v in bucket_sizes.items()}
        self.store_path = Path(store_path)
        self.ws_base_url = ws_base_url.rstrip("/")
        self.flush_interval_sec = int(flush_interval_sec)
        self._stop_event = threading.Event()
        self._states: dict[str, _WSState] = {}
        for sym in self.symbols:
            bs = self.bucket_sizes.get(sym, 0.0)
            self._states[sym] = _WSState(
                vpin_builder=VPINLiveBuilder(sym, bucket_size_v=bs),
                obi_builder=OBIBuilder(sym),
            )

    # ---- Per-stream callbacks ----

    def _on_trade_message(self, sym: str, raw: Any) -> None:
        st = self._states[sym]
        st.events_received += 1
        rec = parse_trade_event(raw)
        if rec is None:
            return
        st.vpin_builder.add_trade(rec["price"], rec["qty"], rec["ts_ms"])
        # Flush si intervalle écoulé
        now_s = time.time()
        if now_s - st.last_flush_sec >= self.flush_interval_sec:
            st.last_flush_sec = now_s
            self._flush_symbol(sym)

    def _on_book_message(self, sym: str, raw: Any) -> None:
        rec = parse_book_ticker_event(raw)
        if rec is None:
            return
        self._states[sym].obi_builder.update(rec["bid_qty"], rec["ask_qty"],
                                              rec["ts_ms"])

    def _flush_symbol(self, sym: str) -> None:
        st = self._states[sym]
        if st.vpin_builder.n_trades() < 5:
            return  # pas assez de trades pour calculer
        vpin = st.vpin_builder.current_vpin()
        obi = st.obi_builder.current_obi()
        record = {
            "ts_ms": int(time.time() * 1000),
            "symbol": sym,
            "vpin": float(vpin),
            "obi": float(obi),
            "n_buckets": int(st.vpin_builder.n_buckets()),
            "n_trades": int(st.vpin_builder.n_trades()),
        }
        append_vpin_record(self.store_path, record)

    # ---- WS run loops ----

    def _run_trade_loop(self, sym: str) -> None:
        if not HAS_WS:
            logger.error("websocket-client not installed")
            return
        url = f"{self.ws_base_url}/{sym.lower()}@trade"
        backoff = _RECONNECT_BASE_SEC
        while not self._stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    url,
                    on_message=lambda ws, raw, s=sym: self._on_trade_message(s, raw),
                    on_error=lambda ws, err, s=sym: self._on_ws_error(s, err),
                    on_close=lambda ws, c, r, s=sym: self._on_ws_close(s, c, r),
                )
                self._states[sym].trade_ws = ws_app
                ws_app.run_forever()
            except Exception as e:
                logger.error(f"[{sym}] trade WS run_forever crashed: {e}")
                self._states[sym].connection_errors += 1
            if self._stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SEC)

    def _run_book_loop(self, sym: str) -> None:
        if not HAS_WS:
            return
        url = f"{self.ws_base_url}/{sym.lower()}@bookTicker"
        backoff = _RECONNECT_BASE_SEC
        while not self._stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    url,
                    on_message=lambda ws, raw, s=sym: self._on_book_message(s, raw),
                    on_error=lambda ws, err, s=sym: self._on_ws_error(s, err),
                    on_close=lambda ws, c, r, s=sym: self._on_ws_close(s, c, r),
                )
                self._states[sym].book_ws = ws_app
                ws_app.run_forever()
            except Exception as e:
                logger.error(f"[{sym}] book WS run_forever crashed: {e}")
            if self._stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SEC)

    def _on_ws_error(self, sym: str, err: Any) -> None:
        self._states[sym].connection_errors += 1
        logger.warning(f"[{sym}] WS error: {err}")

    def _on_ws_close(self, sym: str, code: Any, reason: Any) -> None:
        logger.info(f"[{sym}] WS closed code={code} reason={reason}")

    # ---- Lifecycle ----

    def start(self) -> None:
        self._stop_event.clear()
        for sym in self.symbols:
            t = threading.Thread(target=self._run_trade_loop, args=(sym,), daemon=True)
            b = threading.Thread(target=self._run_book_loop, args=(sym,), daemon=True)
            self._states[sym].trade_thread = t
            self._states[sym].book_thread = b
            t.start()
            b.start()

    def stop(self, drain: bool = True) -> None:
        self._stop_event.set()
        for sym, st in self._states.items():
            if drain:
                self._flush_symbol(sym)
            for ws in (st.trade_ws, st.book_ws):
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    # ---- Diagnostics ----

    def stats(self) -> dict:
        return {
            sym: {
                "events_received": st.events_received,
                "connection_errors": st.connection_errors,
                "n_trades": st.vpin_builder.n_trades(),
                "n_buckets": st.vpin_builder.n_buckets(),
                "last_vpin": st.vpin_builder._cached_vpin,
                "last_obi": st.obi_builder.current_obi(),
            }
            for sym, st in self._states.items()
        }


__all__ = [
    "VPINLiveBuilder",
    "OBIBuilder",
    "VPINLiveCollectorWS",
    "parse_trade_event",
    "parse_book_ticker_event",
    "append_vpin_record",
]
