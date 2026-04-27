#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency monitor — B22.

Mesure la latence des appels Binance API et stocke dans un JSONL.
Le watchdog peut lire le snapshot pour alerter si p95 > seuil.

Usage:
    monitor = LatencyMonitor(log_path)
    with monitor.time("fetch_balance"):
        ex.fetch_balance()
    with monitor.time("create_order", status_field="ok"):
        ex.create_order(...)

    # Plus tard:
    snap = monitor.snapshot(window_minutes=15)
    print(snap.p50, snap.p95, snap.p99)
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class LatencySnapshot:
    operation: str
    n_calls: int
    n_errors: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    avg_ms: float
    error_rate: float

    def to_dict(self) -> dict:
        return {
            "operation": self.operation, "n_calls": self.n_calls, "n_errors": self.n_errors,
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms, "p99_ms": self.p99_ms,
            "max_ms": self.max_ms, "avg_ms": self.avg_ms, "error_rate": self.error_rate,
        }


def _percentile(values: list[float], q: float) -> float:
    """Percentile linéaire (équivalent np.percentile sans dépendance numpy)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    weight = rank - lo
    return s[lo] * (1 - weight) + s[hi] * weight


class LatencyMonitor:
    """Persistance JSONL append-only des latences API."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, operation: str, duration_ms: float, status: str = "ok",
               error: Optional[str] = None):
        entry = {
            "ts": self._now_iso(),
            "operation": operation,
            "duration_ms": float(duration_ms),
            "status": status,
        }
        if error:
            entry["error"] = error[:200]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @contextmanager
    def time(self, operation: str):
        """Context manager qui mesure la durée et logge.

        Tag automatiquement status='error' si une exception est levée.
        """
        t0 = time.perf_counter()
        status = "ok"
        error = None
        try:
            yield self
        except Exception as e:
            status = "error"
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.record(operation, dt_ms, status=status, error=error)

    def _read_recent(self, window_minutes: int) -> list[dict]:
        if not self.log_path.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        out = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    ts = datetime.fromisoformat(e["ts"])
                    if ts >= cutoff:
                        out.append(e)
                except Exception:
                    continue
        return out

    def snapshot(self, window_minutes: int = 15, operation: Optional[str] = None) -> LatencySnapshot:
        """Aggrégat sur une fenêtre récente."""
        entries = self._read_recent(window_minutes)
        if operation:
            entries = [e for e in entries if e.get("operation") == operation]
        durations = [e["duration_ms"] for e in entries]
        n = len(entries)
        n_err = sum(1 for e in entries if e.get("status") == "error")
        return LatencySnapshot(
            operation=operation or "ALL",
            n_calls=n,
            n_errors=n_err,
            p50_ms=_percentile(durations, 0.50),
            p95_ms=_percentile(durations, 0.95),
            p99_ms=_percentile(durations, 0.99),
            max_ms=max(durations) if durations else float("nan"),
            avg_ms=sum(durations) / n if n else float("nan"),
            error_rate=n_err / n if n else 0.0,
        )

    def is_degraded(self, window_minutes: int = 15, p95_threshold_ms: float = 200.0,
                    error_rate_threshold: float = 0.05, min_calls: int = 5) -> tuple[bool, str]:
        """Helper for watchdog: True si latence ou error rate dégradés."""
        snap = self.snapshot(window_minutes)
        if snap.n_calls < min_calls:
            return False, f"insufficient calls ({snap.n_calls} < {min_calls})"
        reasons = []
        if snap.p95_ms > p95_threshold_ms:
            reasons.append(f"p95={snap.p95_ms:.0f}ms > {p95_threshold_ms}ms")
        if snap.error_rate > error_rate_threshold:
            reasons.append(f"error_rate={snap.error_rate:.1%} > {error_rate_threshold:.0%}")
        return bool(reasons), "; ".join(reasons)


__all__ = ["LatencyMonitor", "LatencySnapshot"]
