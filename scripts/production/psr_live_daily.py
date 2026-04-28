#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 — Cron daily PSR live alpha decay detection.

Lit les returns live depuis `binance_bot/data/paper_log.csv` (ou autre source
selon état de la stack), compute PSR vs benchmark backtest, log dans
`binance_bot/data/psr_history.jsonl`, alerte Telegram si PSR < 0.3.

Usage:
    python scripts/production/psr_live_daily.py
    SR_BENCHMARK=0.0 python scripts/production/psr_live_daily.py

Variables d'environnement:
    SR_BENCHMARK : Sharpe benchmark NON annualisé à dépasser (default 0).
    PSR_WINDOW   : nombre de returns récents à considérer (default 30 jours
                   = 30 returns daily, 360 si bars 2h).

Branchement systemd (futur):
    Ajouter au timer hsbc-daily.timer ou créer hsbc-psr-daily.timer
    qui fire à 00:10 UTC quotidiennement.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binance_bot"))

from src.psr_live import compute_psr, classify_psr_alert  # type: ignore


PAPER_LOG_PATH = ROOT / "binance_bot" / "data" / "paper_log.csv"
PSR_HISTORY_PATH = ROOT / "binance_bot" / "data" / "psr_history.jsonl"
DEFAULT_WINDOW = 360  # 30 jours × 12 bars H2


def _load_recent_returns(path: Path, window: int) -> pd.Series:
    """Charge les returns nets des trades clos depuis paper_log.csv.

    paper_log.csv format minimal: colonnes timestamp_iso, action, live_pnl_usdt
    On computes returns par close (action contient 'close') sur capital de
    référence (initial_capital_usdt depuis state.json, fallback 100).
    """
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    if df.empty or "live_pnl_usdt" not in df.columns:
        return pd.Series(dtype=float)
    closes = df[df["action"].astype(str).str.startswith("close_", na=False)].copy()
    if closes.empty:
        return pd.Series(dtype=float)
    # Capital de référence pour normaliser pnl en return
    state_path = ROOT / "binance_bot" / "data" / "state.json"
    capital = 100.0
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            capital = float(state.get("initial_capital_usdt", 100.0))
        except Exception:
            pass
    pnl = closes["live_pnl_usdt"].astype(float)
    returns = (pnl / max(capital, 1e-9)).tail(window)
    return returns.dropna()


def _append_history(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    sr_benchmark = float(os.environ.get("SR_BENCHMARK", "0.0"))
    window = int(os.environ.get("PSR_WINDOW", DEFAULT_WINDOW))

    returns = _load_recent_returns(PAPER_LOG_PATH, window)
    n = len(returns)
    if n < 5:
        print(f"PSR live: only {n} closed trades, need >=5 — skipping daily report")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_obs": n,
            "psr": None,
            "alert": "insufficient_data",
            "sr_benchmark": sr_benchmark,
        }
        _append_history(record, PSR_HISTORY_PATH)
        return 0

    psr = compute_psr(returns, sr_benchmark=sr_benchmark)
    alert_label = classify_psr_alert(psr)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_obs": n,
        "psr": float(psr),
        "alert": alert_label,
        "sr_benchmark": sr_benchmark,
        "window": window,
    }
    _append_history(record, PSR_HISTORY_PATH)
    print(f"PSR daily report: PSR={psr:.4f} (n={n}, sr_bench={sr_benchmark}) → {alert_label}")

    # Telegram alert si seuil franchi (alpha_decay ou pire)
    if alert_label in ("alpha_decay", "kill_system"):
        try:
            from binance_bot.bot.notifier import TelegramNotifier  # type: ignore
            notifier = TelegramNotifier()
            level = "critical" if alert_label == "kill_system" else "warn"
            msg = (
                f"PSR LIVE ALERT [{alert_label}]: PSR={psr:.4f} (n={n}, sr_bench={sr_benchmark}) "
                f"-- alpha decay suspecté, investiguer divergence live vs backtest"
            )
            getattr(notifier, level)(msg)
        except Exception as e:
            print(f"⚠️ Telegram notification failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
