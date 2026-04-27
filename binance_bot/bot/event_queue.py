#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventQueue — buffer thread-safe pour les events user data stream (BUG-B13).

Stocke les events dict en mémoire (deque protégée par lock) et les persiste
en JSONL append-only sous `binance_bot/data/user_stream_events.jsonl` pour
replay/debug.

API publique :
  - push(event_dict)           : appelé par UserStreamListener (thread WS)
  - pop_all() -> list[dict]    : appelé par le runner pour drainer
  - size() -> int              : nombre d'events buffer
  - close()                    : flush et ferme le fichier
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EventQueue:
    """Buffer thread-safe + persistance JSONL pour les events user-stream."""

    def __init__(
        self,
        persist_path: Optional[Union[str, Path]] = None,
        max_buffer: int = 100_000,
    ):
        self._lock = threading.Lock()
        self._buffer: deque = deque()
        self._max_buffer = int(max_buffer)
        self._persist_path: Optional[Path] = (
            Path(persist_path) if persist_path is not None else None
        )
        self._persist_fp = None
        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            # touch pour création (append-only)
            self._persist_fp = open(self._persist_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------

    def push(self, event: Dict[str, Any]) -> None:
        """Ajoute un event au buffer ET persiste en JSONL."""
        if not isinstance(event, dict):
            logger.warning("EventQueue.push: skip non-dict event (%r)", type(event))
            return
        with self._lock:
            self._buffer.append(event)
            # Garde le buffer borné (drop les plus anciens si overflow)
            while len(self._buffer) > self._max_buffer:
                self._buffer.popleft()
            if self._persist_fp is not None:
                try:
                    self._persist_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                    self._persist_fp.flush()
                except Exception as e:
                    logger.error("EventQueue persist failed: %s", e)

    def pop_all(self) -> List[Dict[str, Any]]:
        """Retourne et vide tous les events accumulés."""
        with self._lock:
            out = list(self._buffer)
            self._buffer.clear()
            return out

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def close(self) -> None:
        """Ferme le fichier de persistance (idempotent)."""
        with self._lock:
            if self._persist_fp is not None:
                try:
                    self._persist_fp.flush()
                    self._persist_fp.close()
                except Exception:
                    pass
                self._persist_fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
