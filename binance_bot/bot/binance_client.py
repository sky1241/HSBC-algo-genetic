#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BinanceClient — middleware HTTP commun pour les endpoints Binance Futures.

Couvre :
  - BUG-B11 (time sync) : synchro `serverTime` (GET /fapi/v1/time), offset stocké,
    re-sync auto toutes les 600s, application sur `params['timestamp']`.
  - BUG-B9  (rate limits + backoff) :
      * parsing de `X-MBX-USED-WEIGHT-1m` et `X-MBX-ORDER-COUNT-1m` à chaque réponse,
      * throttle préventif si used_weight > 0.8 * 2400,
      * respect du `Retry-After` sur HTTP 429,
      * log CRITICAL + ban auto sur HTTP 418,
      * backoff exponentiel + jitter sur 5xx / connection errors (5 tentatives max).

API publique (toutes retournent un dict, [] sur listes, ou {} en cas d'échec) :
  - public_get(path, params=None)
  - signed_get(path, params=None)
  - signed_post(path, params=None)
  - signed_delete(path, params=None)
  - time_offset_ms() -> int
  - is_banned() -> bool
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import random
import time
import urllib.parse
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Limites Binance Futures (UM)
WEIGHT_LIMIT_PER_MIN = 2400
WEIGHT_THROTTLE_THRESHOLD = int(0.8 * WEIGHT_LIMIT_PER_MIN)  # 1920
TIME_RESYNC_INTERVAL_SEC = 600  # 10 min
MAX_BACKOFF_RETRIES = 5  # 5 tentatives (attempt 0..4)
MAX_BACKOFF_DELAY = 60.0


class BinanceClient:
    """Wrapper requests pour Binance Futures avec time-sync, rate-limit, backoff."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://testnet.binancefuture.com",
        recv_window: int = 5000,
        auto_sync: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.recv_window = int(recv_window)

        # Time sync state (B11)
        self._time_offset_ms: int = 0
        self._last_time_sync: float = 0.0  # epoch seconds

        # Rate limit state (B9)
        self._used_weight_1m: int = 0
        self._order_count_1m: int = 0
        self._weight_window_start: float = time.time()
        self._banned_until: float = 0.0  # epoch seconds

        if auto_sync:
            try:
                self._sync_time()
            except Exception as e:
                logger.warning("Initial time sync failed: %s", e)

    # ---------------------------------------------------------------------
    # Time sync (B11)
    # ---------------------------------------------------------------------

    def _sync_time(self) -> int:
        """Appelle GET /fapi/v1/time et calcule l'offset serveur - local.

        Returns:
            offset en ms (peut être négatif).
        """
        url = f"{self.base_url}/fapi/v1/time"
        local_before = int(time.time() * 1000)
        resp = requests.get(url, timeout=10)
        local_after = int(time.time() * 1000)
        data = resp.json()
        server_time = int(data["serverTime"])
        # Compense la latence aller-retour (RTT/2)
        local_mid = (local_before + local_after) // 2
        offset = server_time - local_mid
        self._time_offset_ms = offset
        self._last_time_sync = time.time()
        if abs(offset) > 1000:
            logger.warning(
                "Clock skew detected: |offset| = %d ms (> 1000 ms)", abs(offset)
            )
        else:
            logger.debug("Time sync OK: offset=%d ms", offset)
        return offset

    def _maybe_resync(self) -> None:
        """Re-sync si la dernière a > TIME_RESYNC_INTERVAL_SEC."""
        if time.time() - self._last_time_sync > TIME_RESYNC_INTERVAL_SEC:
            try:
                self._sync_time()
            except Exception as e:
                logger.warning("Periodic time resync failed: %s", e)

    def time_offset_ms(self) -> int:
        """Offset serveur - local (en ms). Positif = horloge locale en retard."""
        return self._time_offset_ms

    def _now_ms(self) -> int:
        """Timestamp courant corrigé de l'offset (pour signer les requêtes)."""
        return int(time.time() * 1000) + self._time_offset_ms

    # ---------------------------------------------------------------------
    # Rate limit (B9)
    # ---------------------------------------------------------------------

    def is_banned(self) -> bool:
        """True si on est sous coup d'un HTTP 418 (ban) qui n'a pas expiré."""
        return time.time() < self._banned_until

    def used_weight_1m(self) -> int:
        return self._used_weight_1m

    def order_count_1m(self) -> int:
        return self._order_count_1m

    def _update_rate_headers(self, headers: Any) -> None:
        """Parse les headers Binance et met à jour le compteur local."""
        if headers is None:
            return
        try:
            uw = headers.get("X-MBX-USED-WEIGHT-1m") or headers.get("x-mbx-used-weight-1m")
            if uw is not None:
                self._used_weight_1m = int(uw)
        except (ValueError, TypeError):
            pass
        try:
            oc = headers.get("X-MBX-ORDER-COUNT-1m") or headers.get("x-mbx-order-count-1m")
            if oc is not None:
                self._order_count_1m = int(oc)
        except (ValueError, TypeError):
            pass

    def _maybe_throttle(self) -> None:
        """Throttle préventif si used_weight > WEIGHT_THROTTLE_THRESHOLD (1920)."""
        if self._used_weight_1m > WEIGHT_THROTTLE_THRESHOLD:
            # Sleep jusqu'à la prochaine minute (au sens horloge unix)
            elapsed_in_min = time.time() % 60
            sleep_for = max(0.0, 60.0 - elapsed_in_min)
            logger.warning(
                "Rate limit guard: used_weight_1m=%d > %d → sleep %.2fs",
                self._used_weight_1m, WEIGHT_THROTTLE_THRESHOLD, sleep_for,
            )
            time.sleep(sleep_for)
            self._used_weight_1m = 0  # nouvelle minute

    # ---------------------------------------------------------------------
    # Signature
    # ---------------------------------------------------------------------

    def _sign(self, params: Dict[str, Any]) -> str:
        """HMAC-SHA256 sur la query encodée. Retourne `query&signature=...`."""
        if not self.api_secret:
            raise RuntimeError("api_secret manquant pour signer la requête")
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return query + "&signature=" + sig

    # ---------------------------------------------------------------------
    # HTTP core avec backoff
    # ---------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Exécute une requête HTTP avec gestion 429/418/5xx.

        Returns:
            dict (ou liste enveloppée). En cas d'échec total, retourne {} ou
            la dernière erreur (`{"error": ...}`).
        """
        if self.is_banned():
            logger.critical(
                "Refus d'appel: banned_until=%.0f (now=%.0f)",
                self._banned_until, time.time(),
            )
            return {"error": "banned", "banned_until": self._banned_until}

        self._maybe_resync()
        self._maybe_throttle()

        params = dict(params or {})
        headers = {}
        if signed:
            params["timestamp"] = self._now_ms()
            params.setdefault("recvWindow", self.recv_window)
            qs = self._sign(params)
            url = f"{self.base_url}{path}?{qs}"
            headers["X-MBX-APIKEY"] = self.api_key or ""
            body_params = None
        else:
            qs = urllib.parse.urlencode(params) if params else ""
            url = f"{self.base_url}{path}" + (f"?{qs}" if qs else "")
            body_params = None

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_BACKOFF_RETRIES):
            try:
                resp = requests.request(
                    method.upper(), url, headers=headers, timeout=10
                )
                # Always parse rate headers (même sur erreur)
                self._update_rate_headers(getattr(resp, "headers", {}) or {})
                status = getattr(resp, "status_code", 200)

                # 418 — IP ban
                if status == 418:
                    retry_after = self._parse_retry_after(resp)
                    self._banned_until = time.time() + retry_after
                    logger.critical(
                        "HTTP 418 IP banned. Retry-After=%ds → banned_until=%.0f",
                        retry_after, self._banned_until,
                    )
                    return {"error": "banned", "retry_after": retry_after}

                # 429 — rate limited
                if status == 429:
                    retry_after = self._parse_retry_after(resp)
                    logger.warning(
                        "HTTP 429 rate limited. Retry-After=%ds (attempt %d)",
                        retry_after, attempt,
                    )
                    time.sleep(retry_after)
                    continue

                # 5xx — backoff exponentiel
                if 500 <= status < 600:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "HTTP %d (attempt %d) → backoff %.2fs", status, attempt, delay
                    )
                    time.sleep(delay)
                    continue

                # OK ou 4xx logique métier → on rend le JSON tel quel
                try:
                    return resp.json()
                except Exception:
                    return {"error": "invalid_json", "status": status}

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Connection error (attempt %d): %s → backoff %.2fs",
                    attempt, e, delay,
                )
                time.sleep(delay)
                continue
            except Exception as e:
                last_exc = e
                logger.error("Unexpected request error: %s", e)
                break

        logger.error("Request failed after %d attempts: %s", MAX_BACKOFF_RETRIES, last_exc)
        return {"error": str(last_exc) if last_exc else "max_retries_exceeded"}

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """delay = min(2^attempt + uniform(0, 1), 60)."""
        return min(float(2 ** attempt) + random.uniform(0.0, 1.0), MAX_BACKOFF_DELAY)

    @staticmethod
    def _parse_retry_after(resp: Any) -> int:
        """Lit Retry-After (en secondes), default 60."""
        try:
            headers = getattr(resp, "headers", {}) or {}
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra is not None:
                return max(1, int(float(ra)))
        except (ValueError, TypeError):
            pass
        return 60

    # ---------------------------------------------------------------------
    # API publique simple
    # ---------------------------------------------------------------------

    def public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params, signed=False)

    def signed_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params, signed=True)

    def signed_post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, params=params, signed=True)

    def signed_delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("DELETE", path, params=params, signed=True)
