"""Tests pour watchdog (health checks + B10 liquidation monitor)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.watchdog import (
    _DAILY_TIMESTAMP_RE,
    _LOG_TIMESTAMP_RE,
    HealthReport,
    check_balance_drop,
    check_kill_flag,
    check_liquidation_safety,
    last_log_timestamp,
    run_health_checks,
)


# ============================================================
# HealthReport
# ============================================================

def test_health_report_starts_ok():
    r = HealthReport(timestamp_iso="2026-04-26T20:00:00Z")
    assert r.status == "OK"
    assert r.anomalies == []


def test_health_report_warn_overrides_ok():
    r = HealthReport(timestamp_iso="x")
    r.add("WARN", "test warn")
    assert r.status == "WARN"


def test_health_report_critical_overrides_warn():
    r = HealthReport(timestamp_iso="x")
    r.add("WARN", "warn1")
    r.add("CRITICAL", "crit")
    r.add("WARN", "warn2")
    assert r.status == "CRITICAL"
    assert len(r.anomalies) == 3


# ============================================================
# Log timestamp parsing
# ============================================================

def test_last_log_timestamp_returns_none_for_missing(tmp_path):
    assert last_log_timestamp(tmp_path / "missing.log", _LOG_TIMESTAMP_RE) is None


def test_last_log_timestamp_extracts_intraday(tmp_path):
    log = tmp_path / "intraday.log"
    log.write_text(
        "Some preamble\n"
        "🔄 INTRADAY RUN — 2026-04-26 18:01:13\n"
        "blah\n"
        "🔄 INTRADAY RUN — 2026-04-26 20:01:13\n"
        "trail\n"
    )
    ts = last_log_timestamp(log, _LOG_TIMESTAMP_RE)
    assert ts is not None
    assert ts.year == 2026 and ts.month == 4 and ts.day == 26
    assert ts.hour == 20 and ts.minute == 1


def test_last_log_timestamp_handles_daily(tmp_path):
    log = tmp_path / "daily.log"
    log.write_text("📅 DAILY PHASE UPDATE — 2026-04-26 00:05:30\n")
    ts = last_log_timestamp(log, _DAILY_TIMESTAMP_RE)
    assert ts is not None
    assert ts.hour == 0 and ts.minute == 5


# ============================================================
# Kill flag
# ============================================================

def test_check_kill_flag_no_flag(tmp_path):
    flag = tmp_path / ".killed"
    killed, reason = check_kill_flag(flag)
    assert killed is False
    assert reason == ""


def test_check_kill_flag_present(tmp_path):
    flag = tmp_path / ".killed"
    flag.write_text("global_stop reached at 2026-04-26")
    killed, reason = check_kill_flag(flag)
    assert killed is True
    assert "global_stop" in reason


# ============================================================
# Liquidation safety (B10)
# ============================================================

def test_liq_safety_no_positions():
    assert check_liquidation_safety([]) == []


def test_liq_safety_zero_contracts_ignored():
    pos = [{"symbol": "BTC/USDT", "contracts": 0, "liquidationPrice": 60000, "markPrice": 78000, "side": "long"}]
    assert check_liquidation_safety(pos) == []


def test_liq_safety_far_from_liq_no_alarm():
    pos = [{
        "symbol": "BTC/USDT", "side": "long", "contracts": 0.001,
        "liquidationPrice": 50000, "markPrice": 78000,
    }]
    assert check_liquidation_safety(pos) == []


def test_liq_safety_close_to_liq_with_atr_alerts():
    """Distance < 1.5 ATR ⇒ alerte (B10)."""
    pos = [{
        "symbol": "BTC/USDT", "side": "long", "contracts": 0.001,
        "liquidationPrice": 76500, "markPrice": 78000,
    }]
    # ATR = 1500, distance = 78000-76500 = 1500 = 1×ATR < 1.5
    anomalies = check_liquidation_safety(pos, atr=1500)
    assert len(anomalies) == 1
    assert "BTC/USDT" in anomalies[0]


def test_liq_safety_close_to_liq_without_atr_uses_pct():
    """Sans ATR, on alerte si distance < 5%."""
    pos = [{
        "symbol": "BTC/USDT", "side": "long", "contracts": 0.001,
        "liquidationPrice": 75500, "markPrice": 78000,
    }]
    # 78000-75500 = 2500, 2500/78000 = 3.2% < 5%
    anomalies = check_liquidation_safety(pos)
    assert len(anomalies) == 1


def test_liq_safety_missing_data_alerts():
    pos = [{"symbol": "BTC/USDT", "side": "long", "contracts": 0.001}]
    anomalies = check_liquidation_safety(pos)
    assert len(anomalies) == 1
    assert "manquant" in anomalies[0].lower() or "missing" in anomalies[0].lower()


# ============================================================
# Balance drop
# ============================================================

def test_balance_drop_first_call_no_history(tmp_path):
    hist = tmp_path / "balance_history.jsonl"
    msg = check_balance_drop(5000.0, hist)
    assert msg is None
    # Et le file a été créé avec 1 ligne
    assert hist.exists()
    assert len(hist.read_text().splitlines()) == 1


def test_balance_drop_under_threshold(tmp_path):
    hist = tmp_path / "balance_history.jsonl"
    check_balance_drop(5000.0, hist)
    msg = check_balance_drop(4900.0, hist)  # -2%
    assert msg is None


def test_balance_drop_over_threshold(tmp_path):
    hist = tmp_path / "balance_history.jsonl"
    check_balance_drop(5000.0, hist)
    msg = check_balance_drop(4000.0, hist, threshold=0.10)  # -20%
    assert msg is not None
    assert "drop" in msg.lower()


# ============================================================
# Integration run_health_checks
# ============================================================

def test_run_health_checks_minimal(tmp_path, monkeypatch):
    """Tous les checks externes mockés ⇒ pas de WARN/CRITICAL inattendu."""
    repo = tmp_path
    log_dir = repo / "binance_bot" / "logs"
    log_dir.mkdir(parents=True)
    data_dir = repo / "binance_bot" / "data"
    data_dir.mkdir(parents=True)

    # Logs récents simulés
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    (log_dir / "intraday.log").write_text(f"🔄 INTRADAY RUN — {now_str}\n")
    (log_dir / "daily.log").write_text(f"📅 DAILY PHASE UPDATE — {now_str}\n")

    # Pas de kill flag
    # Mocker systemctl + dashboard
    monkeypatch.setattr("bot.watchdog.check_systemd_timer_active",
                        lambda unit="hsbc-intraday.timer": (True, "active"))
    monkeypatch.setattr("bot.watchdog.check_dashboard_up",
                        lambda url=None, timeout=5: (True, "HTTP 200"))

    report = run_health_checks(repo_root=repo, binance_client=None)
    # On n'a pas de client Binance ⇒ on skip les checks Binance, mais le reste doit passer
    assert report.status == "OK", f"Expected OK, got {report.status}: {report.anomalies}"


def test_run_health_checks_kill_flag_critical(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "binance_bot" / "logs").mkdir(parents=True)
    (repo / "binance_bot" / "data").mkdir(parents=True)
    (repo / "binance_bot" / "data" / ".killed").write_text("test kill")

    monkeypatch.setattr("bot.watchdog.check_systemd_timer_active",
                        lambda unit="hsbc-intraday.timer": (True, "active"))
    monkeypatch.setattr("bot.watchdog.check_dashboard_up",
                        lambda url=None, timeout=5: (True, "HTTP 200"))

    report = run_health_checks(repo_root=repo)
    assert report.status == "CRITICAL"
    assert any("kill" in a.lower() for a in report.anomalies)


def test_run_health_checks_with_binance_client_balance_drop(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "binance_bot" / "logs").mkdir(parents=True)
    (repo / "binance_bot" / "data").mkdir(parents=True)

    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    (repo / "binance_bot" / "logs" / "intraday.log").write_text(f"🔄 INTRADAY RUN — {now_str}\n")

    # Pré-écrire un balance history avec 5000
    hist = repo / "binance_bot" / "data" / "balance_history.jsonl"
    hist.write_text(json.dumps({"ts": "2026-04-26T19:00:00Z", "balance": 5000.0}) + "\n")

    monkeypatch.setattr("bot.watchdog.check_systemd_timer_active",
                        lambda unit="hsbc-intraday.timer": (True, "active"))
    monkeypatch.setattr("bot.watchdog.check_dashboard_up",
                        lambda url=None, timeout=5: (True, "HTTP 200"))

    fake_client = MagicMock()
    fake_client.fetch_balance.return_value = {"USDT": {"total": 4000.0}}  # -20%
    fake_client.fetch_positions.return_value = []

    report = run_health_checks(repo_root=repo, binance_client=fake_client)
    assert report.status == "WARN"
    assert any("drop" in a.lower() for a in report.anomalies)
