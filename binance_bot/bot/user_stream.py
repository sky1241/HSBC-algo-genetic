#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UserStreamListener — Binance USDM Futures user data stream (BUG-B13).

WebSocket listener synchrone (thread daemon + websocket-client) pour les
events `ORDER_TRADE_UPDATE`, `ACCOUNT_UPDATE`, `MARGIN_CALL`,
`listenKeyExpired` en temps réel sur testnet.binancefuture.com.

Architecture :
  - Thread principal (start) : crée le listenKey, ouvre la WS, boucle de
    réception. Auto-reconnect avec backoff exponentiel sur déconnexion.
  - Sous-thread keepalive : PUT listenKey toutes les KEEPALIVE_INTERVAL_SEC
    (30 min — marge sur les 60 min officiels).
  - Sur `listenKeyExpired` → recréer un nouveau listenKey + reconnecter.
  - stop() : close WS + DELETE listenKey + join threads (idempotent).

Tous les events parsés sont dispatchés via `on_event_callback(dict)`.

Endpoints (USDM Futures testnet) :
  - POST   /fapi/v1/listenKey   (signed)         → {"listenKey": "..."}
  - PUT    /fapi/v1/listenKey?listenKey=...      (signed) keepalive
  - DELETE /fapi/v1/listenKey?listenKey=...      (signed) close
  - WS     wss://stream.binancefuture.com/ws/{listenKey}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.parse
from typing import Any, Callable, Dict, Optional

import requests

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover
    websocket = None  # gracieux : tests mockent websocket.WebSocketApp

logger = logging.getLogger(__name__)

# Marge sur les 60 min officiels Binance — on PUT toutes les 30 min.
KEEPALIVE_INTERVAL_SEC = 30 * 60
# Backoff borné pour reconnect WS
RECONNECT_BACKOFF_BASE = 2.0
RECONNECT_BACKOFF_MAX = 60.0

DEFAULT_BASE_REST = "https://testnet.binancefuture.com"
DEFAULT_BASE_WS = "wss://stream.binancefuture.com/ws"


EventCallback = Callable[[Dict[str, Any]], None]


class UserStreamListener:
    """Listener WebSocket Binance USDM Futures user data stream.

    Args:
        api_key, api_secret : clés Binance (signed REST pour listenKey).
        base_rest          : ex. https://testnet.binancefuture.com
        base_ws            : ex. wss://stream.binancefuture.com/ws
        on_event_callback  : callable invoqué pour chaque event (dict).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_rest: str = DEFAULT_BASE_REST,
        base_ws: str = DEFAULT_BASE_WS,
        on_event_callback: Optional[EventCallback] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_rest = base_rest.rstrip("/")
        self.base_ws = base_ws.rstrip("/")
        self.on_event_callback: Optional[EventCallback] = on_event_callback

        self._listen_key: Optional[str] = None
        self._ws = None  # websocket.WebSocketApp
        self._ws_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_requested = threading.Event()

        # Pour debug/tests
        self.connection_errors = 0
        self.events_received = 0

    # ------------------------------------------------------------------
    # REST helpers (signature HMAC-SHA256)
    # ------------------------------------------------------------------

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return query + "&signature=" + sig

    def _signed_request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 5000)
        qs = self._sign(params)
        url = f"{self.base_rest}{path}?{qs}"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            resp = requests.request(method.upper(), url, headers=headers, timeout=10)
            try:
                return resp.json()
            except Exception:
                return {"error": "invalid_json", "status": getattr(resp, "status_code", 0)}
        except Exception as e:
            logger.error("UserStream signed_request %s %s failed: %s", method, path, e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # listenKey lifecycle
    # ------------------------------------------------------------------

    def create_listen_key(self) -> Optional[str]:
        """POST /fapi/v1/listenKey → retourne listenKey (str) ou None."""
        data = self._signed_request("POST", "/fapi/v1/listenKey")
        lk = data.get("listenKey") if isinstance(data, dict) else None
        if lk:
            self._listen_key = lk
            logger.info("listenKey créé: %s...", lk[:8])
        else:
            logger.error("Échec création listenKey: %s", data)
        return lk

    def keepalive_listen_key(self) -> bool:
        """PUT /fapi/v1/listenKey?listenKey=... → True si OK."""
        if not self._listen_key:
            return False
        data = self._signed_request(
            "PUT", "/fapi/v1/listenKey", {"listenKey": self._listen_key}
        )
        ok = isinstance(data, dict) and data.get("error") is None
        if ok:
            logger.debug("listenKey keepalive OK")
        else:
            logger.warning("listenKey keepalive failed: %s", data)
        return ok

    def close_listen_key(self) -> bool:
        """DELETE /fapi/v1/listenKey?listenKey=... → True si OK."""
        if not self._listen_key:
            return False
        data = self._signed_request(
            "DELETE", "/fapi/v1/listenKey", {"listenKey": self._listen_key}
        )
        ok = isinstance(data, dict) and data.get("error") is None
        if ok:
            logger.info("listenKey fermé")
        else:
            logger.warning("listenKey close failed: %s", data)
        self._listen_key = None
        return ok

    # ------------------------------------------------------------------
    # WS callbacks
    # ------------------------------------------------------------------

    def _on_message(self, ws, raw: str) -> None:
        try:
            event = json.loads(raw)
        except (ValueError, TypeError) as e:
            logger.warning("UserStream JSON parse error: %s (raw=%r)", e, raw[:100])
            return
        if not isinstance(event, dict):
            return
        self.events_received += 1
        ev_type = event.get("e")
        logger.debug("UserStream event type=%s", ev_type)

        # Cas spécial : listenKeyExpired → recréer + reconnect
        if ev_type == "listenKeyExpired":
            logger.warning("listenKey expired → reconnect")
            self._reconnect_requested.set()
            try:
                ws.close()
            except Exception:
                pass

        # Dispatch via callback utilisateur (toujours, même listenKeyExpired)
        if self.on_event_callback is not None:
            try:
                self.on_event_callback(event)
            except Exception as e:
                logger.error("on_event_callback raised: %s", e)

    def _on_error(self, ws, error: Any) -> None:
        self.connection_errors += 1
        logger.warning("UserStream WS error: %s", error)

    def _on_close(self, ws, status_code: Any = None, msg: Any = None) -> None:
        logger.info("UserStream WS closed (status=%s msg=%s)", status_code, msg)

    def _on_open(self, ws) -> None:
        logger.info("UserStream WS open")

    # ------------------------------------------------------------------
    # Threads loops
    # ------------------------------------------------------------------

    def _keepalive_loop(self) -> None:
        """PUT listenKey toutes les KEEPALIVE_INTERVAL_SEC."""
        while not self._stop_event.is_set():
            # Wait avec interruption rapide en cas de stop()
            if self._stop_event.wait(timeout=KEEPALIVE_INTERVAL_SEC):
                return
            try:
                self.keepalive_listen_key()
            except Exception as e:
                logger.error("Keepalive loop error: %s", e)

    def _ws_loop(self) -> None:
        """Boucle principale : (re)crée listenKey, ouvre WS, reconnecte."""
        attempt = 0
        while not self._stop_event.is_set():
            # (re)crée listenKey si absent ou si expiré
            if self._listen_key is None or self._reconnect_requested.is_set():
                self._reconnect_requested.clear()
                lk = self.create_listen_key()
                if lk is None:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "create_listen_key failed → backoff %.1fs", delay
                    )
                    if self._stop_event.wait(delay):
                        return
                    attempt += 1
                    continue

            ws_url = f"{self.base_ws}/{self._listen_key}"
            if websocket is None:
                logger.error("websocket-client non installé — WS impossible")
                return

            self._ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            try:
                # run_forever bloque jusqu'à fermeture
                self._ws.run_forever(ping_interval=180, ping_timeout=10)
            except Exception as e:
                logger.error("ws.run_forever exception: %s", e)
                self.connection_errors += 1

            if self._stop_event.is_set():
                break

            # Reconnect avec backoff
            delay = self._backoff_delay(attempt)
            logger.info("UserStream reconnect in %.1fs (attempt %d)", delay, attempt)
            if self._stop_event.wait(delay):
                return
            attempt += 1

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(RECONNECT_BACKOFF_BASE ** min(attempt, 6), RECONNECT_BACKOFF_MAX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Lance les 2 threads daemon (WS + keepalive). Idempotent."""
        if self._ws_thread is not None and self._ws_thread.is_alive():
            logger.warning("UserStream déjà démarré — ignore start()")
            return
        self._stop_event.clear()
        self._reconnect_requested.clear()

        self._ws_thread = threading.Thread(
            target=self._ws_loop, name="UserStream-WS", daemon=True
        )
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, name="UserStream-Keepalive", daemon=True
        )
        self._ws_thread.start()
        self._keepalive_thread.start()
        logger.info("UserStreamListener started")

    def stop(self, join_timeout: float = 5.0) -> None:
        """Arrête proprement : close WS, DELETE listenKey, join threads."""
        self._stop_event.set()
        # Close WS pour débloquer run_forever
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        # DELETE listenKey
        try:
            self.close_listen_key()
        except Exception as e:
            logger.warning("close_listen_key during stop: %s", e)
        # Join threads
        for t in (self._ws_thread, self._keepalive_thread):
            if t is not None and t.is_alive():
                t.join(timeout=join_timeout)
        self._ws_thread = None
        self._keepalive_thread = None
        logger.info("UserStreamListener stopped")
