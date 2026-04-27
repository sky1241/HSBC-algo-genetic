#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit log hash-chain — B15 (compliance / post-mortem).

Append-only JSONL où chaque ligne est:
    {"seq": N, "ts": "...", "prev_hash": "...", "payload": {...}, "hash": "..."}

Le hash de la ligne N est SHA256(prev_hash + canonical(payload)).

Garantit l'intégrité: une mutation/suppression rompt la chaîne et est détectée
par `verify_chain()` au démarrage.

Pas de chiffrement — c'est l'INTÉGRITÉ qu'on garantit, pas la confidentialité.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

GENESIS_HASH = "0" * 64


def _canonical(payload: dict) -> str:
    """Sérialise le payload de manière déterministe (clés triées)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash(prev: str, canonical: str) -> str:
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


@dataclass
class ChainStatus:
    valid: bool
    n_entries: int
    last_hash: str
    error: Optional[str] = None
    error_seq: Optional[int] = None


class AuditLog:
    """Append-only hash chain JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_state(self) -> tuple[int, str]:
        """Retourne (next_seq, last_hash). (0, GENESIS) si fichier absent/vide."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0, GENESIS_HASH
        last = None
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = line
        if last is None:
            return 0, GENESIS_HASH
        try:
            entry = json.loads(last)
            return int(entry["seq"]) + 1, entry["hash"]
        except Exception:
            return 0, GENESIS_HASH

    def append(self, payload: dict) -> dict:
        """Append un payload, retourne l'entry créée (avec hash + seq).

        fsync forcé pour durabilité.
        """
        seq, prev = self._last_state()
        canonical = _canonical(payload)
        h = _hash(prev, canonical)
        entry = {
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "prev_hash": prev,
            "payload": payload,
            "hash": h,
        }
        line = json.dumps(entry, sort_keys=True, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        return entry

    def iter_entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def verify_chain(self) -> ChainStatus:
        """Re-calcule la chaîne et valide qu'aucune ligne n'a été altérée."""
        prev = GENESIS_HASH
        n = 0
        last_hash = GENESIS_HASH
        for entry in self.iter_entries():
            try:
                seq = entry["seq"]
                if seq != n:
                    return ChainStatus(
                        valid=False, n_entries=n, last_hash=last_hash,
                        error=f"seq mismatch (expected {n}, got {seq})", error_seq=seq,
                    )
                if entry["prev_hash"] != prev:
                    return ChainStatus(
                        valid=False, n_entries=n, last_hash=last_hash,
                        error=f"prev_hash mismatch at seq {seq}", error_seq=seq,
                    )
                expected_hash = _hash(prev, _canonical(entry["payload"]))
                if entry["hash"] != expected_hash:
                    return ChainStatus(
                        valid=False, n_entries=n, last_hash=last_hash,
                        error=f"hash mismatch at seq {seq} (tampered payload?)",
                        error_seq=seq,
                    )
                prev = entry["hash"]
                last_hash = entry["hash"]
                n += 1
            except Exception as e:
                return ChainStatus(
                    valid=False, n_entries=n, last_hash=last_hash,
                    error=f"parse error at seq {n}: {e}", error_seq=n,
                )
        return ChainStatus(valid=True, n_entries=n, last_hash=last_hash)


__all__ = ["AuditLog", "ChainStatus", "GENESIS_HASH"]
