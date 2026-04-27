#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery WAL — Write-Ahead Log + clientOrderId déterministe (BUG-B12).

Objectif: garantir l'idempotence après un crash entre `create_market_*_order`
et `_place_algo_order` (SL/TP). Le runner enregistre un `intent` AVANT chaque
appel API, et le marque `completed` après réception confirmée. Au boot, les
`pending_intents` sont rejoués avec le **même** clientOrderId — Binance dédoublonne
nativement, donc rejouer ne crée pas de doublon.

Format JSONL append-only (durabilité fsync) : une ligne = un événement
(`pending` / `completed` / `failed`). La compaction est explicite via
`clear_completed`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Limite Binance pour clientOrderId (newClientOrderId) sur futures USDM.
_MAX_CLIENT_ORDER_ID_LEN = 36


def _utcnow_iso() -> str:
    """Retourne timestamp ISO-8601 UTC (suffixe 'Z')."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def deterministic_client_order_id(strategy: str, bar_ts_ms: int, leg: str) -> str:
    """Génère un clientOrderId déterministe pour idempotence Binance.

    Format: `{strategy}-{bar_ts_ms}-{leg}` tronqué à 36 chars.

    Caractéristiques:
        - Pure fonction des inputs → mêmes inputs → même sortie (replay-safe).
        - Binance USDM rejette les `newClientOrderId` doublons → un retry après
          crash retourne la **même** réponse au lieu de créer un second ordre.

    Args:
        strategy: identifiant de stratégie (ex: "ich2h", "phase1long").
        bar_ts_ms: timestamp ms de la bougie de signal (déterministe par bougie).
        leg: jambe de l'intent (ex: "entry", "sl", "tp", "close").

    Returns:
        str de longueur ≤ 36, ASCII-friendly.
    """
    raw = f"{strategy}-{int(bar_ts_ms)}-{leg}"
    return raw[:_MAX_CLIENT_ORDER_ID_LEN]


@dataclass
class WALEntry:
    """Représentation interne d'une ligne JSONL du WAL."""

    intent_id: str
    status: str  # "pending" | "completed" | "failed"
    ts_iso: str
    action: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "intent_id": self.intent_id,
            "status": self.status,
            "ts_iso": self.ts_iso,
        }
        if self.action is not None:
            d["action"] = self.action
        if self.payload is not None:
            d["payload"] = self.payload
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


class WAL:
    """Write-Ahead Log JSONL append-only avec fsync.

    Chaque méthode `record_intent` / `mark_completed` / `mark_failed` ajoute
    une ligne et force `fsync` pour garantir la durabilité avant retour
    (resistance crash kernel/disque).
    """

    def __init__(self, wal_path: Path):
        """Initialise le WAL.

        Crée le répertoire parent et un fichier vide si absent.
        """
        self.wal_path = Path(wal_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.wal_path.exists():
            # Crée un fichier vide ; `touch` puis fsync sur le directory pour
            # garantir la visibilité de l'inode après crash.
            self.wal_path.touch()
            self._fsync_dir(self.wal_path.parent)

    # ------------------------------------------------------------------ writes

    def record_intent(self, intent_id: str, payload: Dict[str, Any]) -> None:
        """Append `pending` entry et fsync.

        `payload` peut contenir `action`, args API, clientOrderId, etc.
        L'`action` est extrait de `payload` si présent (sinon "unknown").
        """
        entry = WALEntry(
            intent_id=intent_id,
            status="pending",
            ts_iso=_utcnow_iso(),
            action=str(payload.get("action", "unknown")),
            payload=dict(payload),
        )
        self._append(entry)

    def mark_completed(self, intent_id: str, result: Dict[str, Any]) -> None:
        """Append `completed` entry et fsync."""
        entry = WALEntry(
            intent_id=intent_id,
            status="completed",
            ts_iso=_utcnow_iso(),
            result=dict(result),
        )
        self._append(entry)

    def mark_failed(self, intent_id: str, error: str) -> None:
        """Append `failed` entry et fsync."""
        entry = WALEntry(
            intent_id=intent_id,
            status="failed",
            ts_iso=_utcnow_iso(),
            error=str(error),
        )
        self._append(entry)

    # ------------------------------------------------------------------- reads

    def pending_intents(self) -> List[Dict[str, Any]]:
        """Retourne les intents dont la dernière entry est `pending`.

        Algorithme: parcourt toutes les lignes, retient pour chaque `intent_id`
        son dernier statut. Ne retourne que ceux finissant en `pending` —
        avec leur payload original.
        """
        last_status: Dict[str, str] = {}
        first_pending: Dict[str, Dict[str, Any]] = {}
        for entry in self._read_all():
            iid = entry.intent_id
            last_status[iid] = entry.status
            if entry.status == "pending" and iid not in first_pending:
                first_pending[iid] = entry.to_dict()

        return [
            first_pending[iid]
            for iid, st in last_status.items()
            if st == "pending" and iid in first_pending
        ]

    # -------------------------------------------------------------- compaction

    def clear_completed(self, before_iso: str) -> None:
        """Compacte le WAL en supprimant les lignes terminées et anciennes.

        Conserve une ligne si:
            - `ts_iso >= before_iso` (récente, qu'importe le statut), OU
            - son `intent_id` n'a pas de statut final (`completed`/`failed`)
              dans tout le fichier (toujours pending → on garde tout son trail).

        Réécriture atomique via `tmp + os.replace`.
        """
        all_entries = self._read_all()

        # Statut final éventuel par intent_id
        final_status: Dict[str, str] = {}
        for e in all_entries:
            if e.status in ("completed", "failed"):
                final_status[e.intent_id] = e.status

        kept: List[WALEntry] = []
        for e in all_entries:
            keep_recent = e.ts_iso >= before_iso
            keep_unfinished = e.intent_id not in final_status
            if keep_recent or keep_unfinished:
                kept.append(e)

        tmp = self.wal_path.with_suffix(self.wal_path.suffix + ".compact")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in kept:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.wal_path)
        self._fsync_dir(self.wal_path.parent)

    # --------------------------------------------------------------- internals

    def _append(self, entry: WALEntry) -> None:
        """Append + flush + fsync (durabilité)."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        # `open` en mode append pour atomicité POSIX sur écritures < PIPE_BUF.
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Certains filesystems (tmpfs) ne supportent pas fsync — best effort.
                pass

    def _read_all(self) -> List[WALEntry]:
        """Lit toutes les entries, ignore les lignes corrompues silencieusement."""
        if not self.wal_path.exists():
            return []
        out: List[WALEntry] = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    # Ligne tronquée par un crash mid-write → skip
                    continue
                try:
                    out.append(
                        WALEntry(
                            intent_id=str(obj["intent_id"]),
                            status=str(obj["status"]),
                            ts_iso=str(obj["ts_iso"]),
                            action=obj.get("action"),
                            payload=obj.get("payload"),
                            result=obj.get("result"),
                            error=obj.get("error"),
                        )
                    )
                except KeyError:
                    continue
        return out

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """fsync sur le directory pour persister l'entrée d'inode (POSIX)."""
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
