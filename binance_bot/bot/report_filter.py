"""Helper de filtrage pour rapports : exclut tout ce qui précède le baseline.

Les trades de setup (bleed wallet $4998->$103, tests round-trip de validation)
ne font PAS partie de la stratégie K3 et doivent être exclus de tout P&L report,
audit perf, ou tracking error live vs backtest.

Usage:
    from bot.report_filter import load_baseline, filter_after_baseline

    baseline = load_baseline()
    trades_clean = filter_after_baseline(ex.fetch_my_trades('BTC/USDT'), baseline)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_BASELINE_PATH = Path(__file__).resolve().parents[1] / "data" / "report_baseline.json"


def load_baseline(path: Optional[Path] = None) -> Optional[dict]:
    """Charge le baseline depuis report_baseline.json. None si absent (= pas de cutoff)."""
    p = Path(path) if path else DEFAULT_BASELINE_PATH
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def baseline_ts_ms(baseline: Optional[dict]) -> Optional[int]:
    """Retourne le timestamp baseline en ms epoch, ou None si pas de baseline."""
    if not baseline:
        return None
    ts_str = baseline.get("baseline_ts_utc")
    if not ts_str:
        return None
    if ts_str.endswith("Z"):
        ts_str = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def filter_after_baseline(records: Iterable[dict], baseline: Optional[dict],
                          ts_key: str = "timestamp") -> list[dict]:
    """Filtre une liste de records (trades, orders) pour ne garder que ts >= baseline.

    Args:
        records: itérable de dicts (format ccxt: {timestamp: ms_epoch, ...})
        baseline: dict chargé via load_baseline() ; si None, retourne tout.
        ts_key: nom du champ timestamp (défaut "timestamp" pour ccxt).

    Returns:
        Liste filtrée.
    """
    cutoff_ms = baseline_ts_ms(baseline)
    if cutoff_ms is None:
        return list(records)
    return [r for r in records if r.get(ts_key, 0) >= cutoff_ms]
