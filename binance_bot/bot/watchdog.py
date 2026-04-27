#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watchdog: health checks périodiques du bot HSBC.

Vérifie :
- Le timer systemd intraday est actif
- Le dernier run intraday a eu lieu il y a moins que `max_silence_seconds`
- Le dashboard répond sur localhost:8080
- Aucun kill flag actif
- Liquidation price loin du markPrice (B10)
- Balance n'a pas plongé brutalement entre deux ticks

Renvoie un `HealthReport` avec status OK / WARN / CRITICAL et liste des anomalies.
À runner via un service systemd timer toutes les 5 minutes.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Seuils par défaut
INTRADAY_MAX_SILENCE_S = 3 * 3600       # 3h sans run intraday → WARN (timer 2h)
DAILY_MAX_SILENCE_S = 26 * 3600         # 26h sans run daily → WARN
DASHBOARD_URL = "http://127.0.0.1:8080/api/status"
DASHBOARD_TIMEOUT_S = 5
LIQUIDATION_DISTANCE_ATR_MIN = 1.5      # alerte si markPrice <1.5×ATR de liq
BALANCE_DROP_THRESHOLD = 0.10           # alerte si balance baisse >10% entre 2 checks


# Severity ordering: CRITICAL > WARN > OK
SEVERITY_RANK = {"OK": 0, "WARN": 1, "CRITICAL": 2}


@dataclass
class HealthReport:
    timestamp_iso: str
    status: str = "OK"                  # OK / WARN / CRITICAL
    anomalies: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add(self, severity: str, message: str):
        if severity not in SEVERITY_RANK:
            severity = "WARN"
        self.anomalies.append(f"[{severity}] {message}")
        if SEVERITY_RANK[severity] > SEVERITY_RANK[self.status]:
            self.status = severity

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_systemd_timer_active(unit: str = "hsbc-intraday.timer") -> tuple[bool, str]:
    """Retourne (is_active, raw_status_str)."""
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "is-active", unit],
            text=True, stderr=subprocess.STDOUT, timeout=5,
        ).strip()
        return out == "active", out
    except subprocess.CalledProcessError as e:
        return False, e.output.strip() if e.output else "unknown"
    except Exception as e:
        return False, str(e)


_LOG_TIMESTAMP_RE = re.compile(r"INTRADAY RUN — (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_DAILY_TIMESTAMP_RE = re.compile(r"DAILY PHASE UPDATE — (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def last_log_timestamp(log_path: Path, regex: re.Pattern) -> Optional[datetime]:
    """Parse le dernier timestamp matchant `regex` dans le fichier log."""
    if not log_path.exists():
        return None
    try:
        # Lire en remontant (le fichier peut être gros)
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk_size = min(65536, size)
            f.seek(max(0, size - chunk_size))
            tail = f.read().decode("utf-8", errors="replace")
        matches = regex.findall(tail)
        if not matches:
            return None
        last = matches[-1]
        # Heure locale présumée — on convertit à UTC en ASSUMANT le système est en UTC ou heure locale.
        # Pour rester safe, on assume "naive" et on marque tz=UTC pour comparaison cohérente.
        return datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def check_dashboard_up(url: str = DASHBOARD_URL, timeout: float = DASHBOARD_TIMEOUT_S) -> tuple[bool, str]:
    """Renvoie (is_up, detail). Tente une requête HTTP GET."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if 200 <= resp.status < 400:
                return True, f"HTTP {resp.status}"
            return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)[:120]


def check_kill_flag(flag_path: Path) -> tuple[bool, str]:
    """Renvoie (killed, reason)."""
    if not flag_path.exists():
        return False, ""
    try:
        return True, flag_path.read_text(encoding="utf-8")[:300]
    except Exception as e:
        return True, f"(unreadable: {e})"


def check_liquidation_safety(
    positions: list[dict],
    min_distance_atr: float = LIQUIDATION_DISTANCE_ATR_MIN,
    atr: Optional[float] = None,
) -> list[str]:
    """Retourne anomalies si une position est trop proche du liq price (B10).

    Args:
        positions: liste retournée par fetch_positions (ccxt format).
        min_distance_atr: seuil — si markPrice à moins de N×ATR de liq → WARN.
        atr: ATR courant en USDT (estimé via signal engine ou ohlcv). Si None,
             on retourne les distances en %.
    """
    anomalies = []
    for p in positions or []:
        contracts = float(p.get("contracts") or 0)
        if contracts == 0:
            continue
        liq = p.get("liquidationPrice")
        mark = p.get("markPrice") or p.get("info", {}).get("markPrice")
        try:
            liq = float(liq) if liq else None
            mark = float(mark) if mark else None
        except (TypeError, ValueError):
            liq, mark = None, None
        if liq is None or mark is None or liq <= 0 or mark <= 0:
            anomalies.append(
                f"Position {p.get('symbol')} {p.get('side')}: liquidationPrice ou markPrice manquant ({liq=}, {mark=})"
            )
            continue
        distance_pct = abs(mark - liq) / mark
        if atr and atr > 0:
            distance_atr = abs(mark - liq) / atr
            if distance_atr < min_distance_atr:
                anomalies.append(
                    f"Position {p.get('symbol')} {p.get('side')}: liq={liq:.2f}, mark={mark:.2f}, "
                    f"distance={distance_atr:.2f} ATR (<{min_distance_atr})"
                )
        else:
            if distance_pct < 0.05:  # <5% = critique sans ATR
                anomalies.append(
                    f"Position {p.get('symbol')} {p.get('side')}: liq={liq:.2f}, mark={mark:.2f}, "
                    f"distance={distance_pct*100:.1f}% (<5%)"
                )
    return anomalies


def check_balance_drop(
    current_balance: float,
    history_path: Path,
    threshold: float = BALANCE_DROP_THRESHOLD,
) -> Optional[str]:
    """Compare balance à la dernière valeur loggée. Retourne message si chute > threshold.

    Side-effect: append (timestamp, balance) à history_path.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    drop_msg = None
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last = json.loads(lines[-1])
                last_balance = float(last.get("balance", 0))
                if last_balance > 0:
                    drop = (last_balance - current_balance) / last_balance
                    if drop > threshold:
                        drop_msg = (
                            f"Balance drop: {last_balance:.2f} → {current_balance:.2f} "
                            f"({drop*100:.1f}%, seuil {threshold*100:.0f}%)"
                        )
        except Exception:
            pass
    # Append current
    try:
        with open(history_path, "a") as f:
            f.write(json.dumps({"ts": _now_iso(), "balance": current_balance}) + "\n")
    except Exception:
        pass
    return drop_msg


def run_health_checks(
    repo_root: Path,
    intraday_log: Optional[Path] = None,
    daily_log: Optional[Path] = None,
    kill_flag: Optional[Path] = None,
    balance_history: Optional[Path] = None,
    binance_client=None,                  # objet exposant fetch_balance/fetch_positions
    intraday_max_silence_s: float = INTRADAY_MAX_SILENCE_S,
    daily_max_silence_s: float = DAILY_MAX_SILENCE_S,
) -> HealthReport:
    """Tous les checks. binance_client peut être None pour skip les checks Binance."""
    report = HealthReport(timestamp_iso=_now_iso())
    repo_root = Path(repo_root)
    intraday_log = intraday_log or (repo_root / "binance_bot" / "logs" / "intraday.log")
    daily_log = daily_log or (repo_root / "binance_bot" / "logs" / "daily.log")
    kill_flag = kill_flag or (repo_root / "binance_bot" / "data" / ".killed")
    balance_history = balance_history or (repo_root / "binance_bot" / "data" / "balance_history.jsonl")

    # 1. Timers systemd
    is_active, raw = check_systemd_timer_active("hsbc-intraday.timer")
    report.metrics["intraday_timer_active"] = is_active
    if not is_active:
        report.add("CRITICAL", f"Timer hsbc-intraday inactive (status='{raw}')")

    is_active_daily, raw_daily = check_systemd_timer_active("hsbc-daily.timer")
    report.metrics["daily_timer_active"] = is_active_daily
    if not is_active_daily:
        report.add("WARN", f"Timer hsbc-daily inactive (status='{raw_daily}')")

    # 2. Last run intraday
    last_intraday = last_log_timestamp(intraday_log, _LOG_TIMESTAMP_RE)
    if last_intraday is None:
        report.add("WARN", f"Aucun timestamp dans {intraday_log}")
    else:
        age = (datetime.now(timezone.utc) - last_intraday).total_seconds()
        report.metrics["last_intraday_age_s"] = age
        if age > intraday_max_silence_s:
            report.add("CRITICAL", f"Intraday silencieux depuis {age/3600:.1f}h (last: {last_intraday.isoformat()})")

    # 3. Last run daily
    last_daily = last_log_timestamp(daily_log, _DAILY_TIMESTAMP_RE)
    if last_daily is None:
        report.add("WARN", f"Aucun timestamp dans {daily_log}")
    else:
        age = (datetime.now(timezone.utc) - last_daily).total_seconds()
        report.metrics["last_daily_age_s"] = age
        if age > daily_max_silence_s:
            report.add("WARN", f"Daily silencieux depuis {age/3600:.1f}h")

    # 4. Dashboard
    dash_up, dash_detail = check_dashboard_up()
    report.metrics["dashboard_up"] = dash_up
    if not dash_up:
        report.add("WARN", f"Dashboard down: {dash_detail}")

    # 5. Kill flag
    killed, reason = check_kill_flag(kill_flag)
    report.metrics["killed"] = killed
    if killed:
        report.add("CRITICAL", f"Kill flag actif: {reason[:200]}")

    # 6. Binance state (si client fourni)
    if binance_client is not None:
        try:
            balance = binance_client.fetch_balance()
            usdt_total = float(balance.get("USDT", {}).get("total") or 0)
            report.metrics["balance_usdt_total"] = usdt_total
            drop = check_balance_drop(usdt_total, balance_history)
            if drop:
                report.add("WARN", drop)
        except Exception as e:
            report.add("WARN", f"fetch_balance failed: {e}")

        try:
            symbol = "BTC/USDT"
            positions = binance_client.fetch_positions([symbol])
            n_open = sum(1 for p in positions if float(p.get("contracts") or 0) > 0)
            report.metrics["open_positions"] = n_open
            for a in check_liquidation_safety(positions):
                report.add("CRITICAL", a)
        except Exception as e:
            report.add("WARN", f"fetch_positions failed: {e}")

    return report


__all__ = [
    "HealthReport",
    "run_health_checks",
    "check_systemd_timer_active",
    "last_log_timestamp",
    "check_dashboard_up",
    "check_kill_flag",
    "check_liquidation_safety",
    "check_balance_drop",
    "_LOG_TIMESTAMP_RE",
    "_DAILY_TIMESTAMP_RE",
]
