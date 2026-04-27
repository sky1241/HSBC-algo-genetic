"""Tests pour EventQueue (BUG-B13 — buffer thread-safe + persist JSONL)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.event_queue import EventQueue


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            out.append(json.loads(raw))
    return out


# ============================================================ basics

def test_push_then_pop_all_returns_and_clears(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl")
    q.push({"e": "ORDER_TRADE_UPDATE", "x": 1})
    q.push({"e": "ACCOUNT_UPDATE", "x": 2})
    assert q.size() == 2
    out = q.pop_all()
    assert len(out) == 2
    assert out[0]["e"] == "ORDER_TRADE_UPDATE"
    assert out[1]["e"] == "ACCOUNT_UPDATE"
    # Vidé
    assert q.size() == 0
    assert q.pop_all() == []
    q.close()


def test_pop_all_on_empty_returns_empty_list(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl")
    assert q.pop_all() == []
    q.close()


def test_push_skips_non_dict(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl")
    q.push("not a dict")  # type: ignore[arg-type]
    q.push(42)  # type: ignore[arg-type]
    q.push({"e": "OK"})
    assert q.size() == 1
    q.close()


# ============================================================ persistance JSONL

def test_persistance_appends_to_jsonl(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    q = EventQueue(persist_path=p)
    q.push({"e": "A", "n": 1})
    q.push({"e": "B", "n": 2})
    q.close()

    rows = _read_jsonl(p)
    assert len(rows) == 2
    assert rows[0] == {"e": "A", "n": 1}
    assert rows[1] == {"e": "B", "n": 2}


def test_persistance_creates_parent_dir(tmp_path: Path):
    p = tmp_path / "deep" / "sub" / "events.jsonl"
    assert not p.parent.exists()
    q = EventQueue(persist_path=p)
    q.push({"e": "X"})
    q.close()
    assert p.exists()
    assert _read_jsonl(p) == [{"e": "X"}]


def test_no_persist_path_works_in_memory_only():
    q = EventQueue(persist_path=None)
    q.push({"e": "MEM"})
    assert q.size() == 1
    out = q.pop_all()
    assert out == [{"e": "MEM"}]
    q.close()


# ============================================================ thread-safe

def test_concurrent_push_does_not_lose_events(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl")
    N_THREADS = 8
    PER_THREAD = 200

    def worker(tid: int):
        for i in range(PER_THREAD):
            q.push({"tid": tid, "i": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert q.size() == N_THREADS * PER_THREAD
    drained = q.pop_all()
    assert len(drained) == N_THREADS * PER_THREAD
    q.close()


# ============================================================ buffer borné

def test_buffer_bounded_by_max(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl", max_buffer=5)
    for i in range(20):
        q.push({"i": i})
    # Buffer ne doit pas dépasser 5 ; les plus anciens sont droppés
    assert q.size() == 5
    out = q.pop_all()
    assert [e["i"] for e in out] == [15, 16, 17, 18, 19]
    # Mais la persistance JSONL doit avoir TOUS les events (append-only)
    rows = _read_jsonl(tmp_path / "events.jsonl")
    assert len(rows) == 20
    q.close()


def test_close_is_idempotent(tmp_path: Path):
    q = EventQueue(persist_path=tmp_path / "events.jsonl")
    q.push({"e": "OK"})
    q.close()
    q.close()  # ne lève pas
