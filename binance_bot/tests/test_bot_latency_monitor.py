"""Tests pour LatencyMonitor — B22."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.latency_monitor import LatencyMonitor, LatencySnapshot


@pytest.fixture
def lm(tmp_path):
    return LatencyMonitor(tmp_path / "latency.jsonl")


def test_record_appends_line(lm):
    lm.record("fetch_balance", 50.0)
    snap = lm.snapshot(window_minutes=60)
    assert snap.n_calls == 1
    assert snap.p50_ms == pytest.approx(50.0)


def test_time_context_manager_records_duration(lm):
    with lm.time("test_op"):
        time.sleep(0.01)  # ~10ms
    snap = lm.snapshot(60, operation="test_op")
    assert snap.n_calls == 1
    assert 5 < snap.p50_ms < 200  # tolérance large système


def test_time_context_records_error_on_exception(lm):
    with pytest.raises(ValueError):
        with lm.time("failing_op"):
            raise ValueError("boom")
    snap = lm.snapshot(60, operation="failing_op")
    assert snap.n_calls == 1
    assert snap.n_errors == 1
    assert snap.error_rate == 1.0


def test_snapshot_percentiles_correct():
    """100 records linéaires → p50=50, p95=95, p99=99."""
    lm = LatencyMonitor("/tmp/test_latency_p.jsonl")
    Path(lm.log_path).unlink(missing_ok=True)
    for i in range(1, 101):
        lm.record("op", float(i))
    snap = lm.snapshot(60, operation="op")
    assert snap.n_calls == 100
    assert 49 <= snap.p50_ms <= 51
    assert 94 <= snap.p95_ms <= 96
    assert 98 <= snap.p99_ms <= 100
    assert snap.max_ms == 100.0


def test_snapshot_filters_by_operation(lm):
    lm.record("op_a", 100)
    lm.record("op_a", 200)
    lm.record("op_b", 1000)
    snap_a = lm.snapshot(60, operation="op_a")
    assert snap_a.n_calls == 2
    snap_b = lm.snapshot(60, operation="op_b")
    assert snap_b.n_calls == 1
    snap_all = lm.snapshot(60)
    assert snap_all.n_calls == 3


def test_is_degraded_below_threshold(lm):
    for i in range(20):
        lm.record("fast", 50.0)
    degraded, reason = lm.is_degraded(window_minutes=60, p95_threshold_ms=200, min_calls=5)
    assert degraded is False


def test_is_degraded_above_p95_threshold(lm):
    for _ in range(20):
        lm.record("slow", 500.0)
    degraded, reason = lm.is_degraded(window_minutes=60, p95_threshold_ms=200, min_calls=5)
    assert degraded is True
    assert "p95" in reason


def test_is_degraded_above_error_rate(lm):
    for _ in range(15):
        lm.record("ok_op", 50.0, status="ok")
    for _ in range(5):
        lm.record("err_op", 50.0, status="error")
    degraded, reason = lm.is_degraded(window_minutes=60, error_rate_threshold=0.10, min_calls=5)
    assert degraded is True
    assert "error_rate" in reason


def test_is_degraded_insufficient_calls(lm):
    lm.record("op", 50.0)
    degraded, reason = lm.is_degraded(window_minutes=60, min_calls=5)
    assert degraded is False
    assert "insufficient" in reason


def test_snapshot_empty_when_no_records(lm):
    snap = lm.snapshot(60)
    assert snap.n_calls == 0
    assert snap.error_rate == 0.0


def test_snapshot_to_dict_serializable(lm):
    lm.record("op", 50.0)
    snap = lm.snapshot(60)
    d = snap.to_dict()
    import json
    s = json.dumps(d)  # doit être sérialisable
    assert "n_calls" in s
