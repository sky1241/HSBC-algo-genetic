"""Tests pour recovery_wal — BUG-B12 (WAL + clientOrderId déterministe)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.recovery_wal import WAL, deterministic_client_order_id


# ---------------------------------------------------------------- utilitaires

def _read_lines(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            out.append(json.loads(raw))
    return out


# ============================================================ WAL: structure

def test_wal_creates_file_if_absent(tmp_path: Path):
    wal_path = tmp_path / "subdir" / "wal.jsonl"
    assert not wal_path.exists()
    wal = WAL(wal_path)
    assert wal_path.exists()
    assert wal_path.read_text(encoding="utf-8") == ""


# ===================================================== WAL: record_intent

def test_record_intent_appends_pending_entry(tmp_path: Path):
    wal = WAL(tmp_path / "wal.jsonl")
    wal.record_intent(
        "intent-1",
        {"action": "open_long_sl", "symbol": "BTC/USDT", "stop": 50000.0},
    )

    lines = _read_lines(tmp_path / "wal.jsonl")
    assert len(lines) == 1
    entry = lines[0]
    assert entry["intent_id"] == "intent-1"
    assert entry["status"] == "pending"
    assert entry["action"] == "open_long_sl"
    assert entry["payload"]["symbol"] == "BTC/USDT"
    assert entry["payload"]["stop"] == 50000.0
    assert "ts_iso" in entry


# ===================================================== WAL: mark_completed

def test_record_intent_then_mark_completed_pending_empty(tmp_path: Path):
    wal = WAL(tmp_path / "wal.jsonl")
    wal.record_intent("intent-1", {"action": "open_long_sl"})
    wal.mark_completed("intent-1", {"orderId": 12345, "status": "FILLED"})

    pending = wal.pending_intents()
    assert pending == []

    lines = _read_lines(tmp_path / "wal.jsonl")
    assert len(lines) == 2
    assert lines[1]["status"] == "completed"
    assert lines[1]["result"]["orderId"] == 12345


# ===================================================== WAL: mark_failed

def test_record_intent_then_mark_failed_pending_empty(tmp_path: Path):
    wal = WAL(tmp_path / "wal.jsonl")
    wal.record_intent("intent-1", {"action": "open_long_sl"})
    wal.mark_failed("intent-1", "binance error -2010 insufficient balance")

    pending = wal.pending_intents()
    assert pending == []

    lines = _read_lines(tmp_path / "wal.jsonl")
    assert len(lines) == 2
    assert lines[1]["status"] == "failed"
    assert "insufficient balance" in lines[1]["error"]


# ===================================================== WAL: pending_intents

def test_pending_intents_returns_only_pending(tmp_path: Path):
    wal = WAL(tmp_path / "wal.jsonl")
    wal.record_intent("intent-A", {"action": "open_long_sl"})
    wal.record_intent("intent-B", {"action": "open_long_tp"})
    wal.mark_completed("intent-A", {"orderId": 1})
    # intent-B reste pending
    pending = wal.pending_intents()
    assert len(pending) == 1
    assert pending[0]["intent_id"] == "intent-B"
    assert pending[0]["status"] == "pending"
    assert pending[0]["action"] == "open_long_tp"


def test_pending_intents_handles_multiple_intents(tmp_path: Path):
    wal = WAL(tmp_path / "wal.jsonl")
    # 5 intents, 2 completed, 1 failed, 2 pending
    wal.record_intent("i-1", {"action": "open_long_sl"})
    wal.record_intent("i-2", {"action": "open_long_tp"})
    wal.record_intent("i-3", {"action": "close_short"})
    wal.record_intent("i-4", {"action": "open_short_sl"})
    wal.record_intent("i-5", {"action": "open_short_tp"})

    wal.mark_completed("i-1", {"orderId": 1})
    wal.mark_completed("i-3", {"orderId": 3})
    wal.mark_failed("i-5", "rejected")

    pending = wal.pending_intents()
    pending_ids = sorted(p["intent_id"] for p in pending)
    assert pending_ids == ["i-2", "i-4"]


# ===================================================== WAL: clear_completed

def test_clear_completed_removes_old_completed(tmp_path: Path):
    wal_path = tmp_path / "wal.jsonl"
    wal = WAL(wal_path)
    wal.record_intent("old-1", {"action": "open_long_sl"})
    wal.mark_completed("old-1", {"orderId": 1})

    # Cutoff dans le futur → tout est "before"
    future_iso = (datetime.now(timezone.utc) + timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    wal.clear_completed(before_iso=future_iso)

    lines = _read_lines(wal_path)
    # old-1 a un statut final ET son ts < future_iso → supprimé entièrement
    ids = [ln["intent_id"] for ln in lines]
    assert "old-1" not in ids


def test_clear_completed_keeps_pending(tmp_path: Path):
    wal_path = tmp_path / "wal.jsonl"
    wal = WAL(wal_path)
    wal.record_intent("done-1", {"action": "open_long_sl"})
    wal.mark_completed("done-1", {"orderId": 1})
    wal.record_intent("pending-1", {"action": "open_long_tp"})

    future_iso = (datetime.now(timezone.utc) + timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    wal.clear_completed(before_iso=future_iso)

    lines = _read_lines(wal_path)
    ids = [ln["intent_id"] for ln in lines]
    # pending-1 conservé (pas de statut final), done-1 supprimé
    assert "pending-1" in ids
    assert "done-1" not in ids

    # pending_intents le voit toujours
    pending = wal.pending_intents()
    assert len(pending) == 1
    assert pending[0]["intent_id"] == "pending-1"


# ============================================ deterministic_client_order_id

def test_deterministic_client_order_id_format(tmp_path: Path):
    coid = deterministic_client_order_id("ich2h", 1700000000000, "sl")
    assert coid == "ich2h-1700000000000-sl"
    assert len(coid) <= 36


def test_deterministic_client_order_id_truncated_at_36(tmp_path: Path):
    long_strategy = "very-long-strategy-name-with-many-chars"
    coid = deterministic_client_order_id(long_strategy, 1700000000000, "entry")
    assert len(coid) == 36
    # Doit rester un préfixe du raw
    raw = f"{long_strategy}-1700000000000-entry"
    assert coid == raw[:36]


def test_deterministic_client_order_id_idempotent(tmp_path: Path):
    a = deterministic_client_order_id("ich2h", 1700000000000, "sl")
    b = deterministic_client_order_id("ich2h", 1700000000000, "sl")
    c = deterministic_client_order_id("ich2h", 1700000000000, "tp")
    assert a == b
    assert a != c
