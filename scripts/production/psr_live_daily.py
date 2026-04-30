#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 — Cron daily PSR live alpha decay detection (PSR-001 branchement systemd).

Lit les returns live depuis `binance_bot/data/paper_log.csv` (ou autre source
selon état de la stack), compute PSR vs benchmark backtest, log dans
`binance_bot/data/psr_history.jsonl`, alerte Telegram si PSR < 0.3.

Usage:
    python scripts/production/psr_live_daily.py
    SR_BENCHMARK=0.0 PSR_MIN_N=20 python scripts/production/psr_live_daily.py

Configuration (binance_bot/configs/bot_settings.yaml):
    psr_benchmark_sr           : Sharpe benchmark à dépasser (default 1.0).
    psr_min_track_record_length: minimum returns pour PSR significatif
                                 (default 15 ; Bailey-LdP 2012 MinTRL).

Variables d'environnement (overrident le yaml, pour debug ad-hoc):
    SR_BENCHMARK : remplace psr_benchmark_sr.
    PSR_WINDOW   : fenêtre de returns à considérer (default 360 = 30j × 12 bars H2).
    PSR_MIN_N    : remplace psr_min_track_record_length.

Branchement systemd (PSR-001 — branché 2026-04-30):
    `~/.config/systemd/user/hsbc-psr.timer` fire 23:50 UTC quotidiennement,
    déclenche `hsbc-psr.service` qui exécute ce script via le wrapper
    `binance_bot/systemd/hsbc-bot-runner.sh scripts/production/psr_live_daily.py`.
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
SETTINGS_PATH = ROOT / "binance_bot" / "configs" / "bot_settings.yaml"
DEFAULT_WINDOW = 360  # 30 jours × 12 bars H2
DEFAULT_MIN_N = 15  # Bailey-LdP MinTRL ~7-15 jours pour SR_hat=1, alpha=0.05
DEFAULT_SR_BENCHMARK = 1.0


def _load_psr_settings() -> tuple[float, int]:
    """Lit bot_settings.yaml ; env vars prennent priorité.

    Returns: (sr_benchmark, min_n_returns)
    """
    cfg_sr: float = DEFAULT_SR_BENCHMARK
    cfg_n: int = DEFAULT_MIN_N
    try:
        import yaml  # type: ignore

        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            cfg_sr = float(settings.get("psr_benchmark_sr", DEFAULT_SR_BENCHMARK))
            cfg_n = int(settings.get("psr_min_track_record_length", DEFAULT_MIN_N))
    except Exception as e:  # pragma: no cover (yaml optional, fallback safe)
        print(f"⚠️ PSR settings: yaml load failed ({e}), using defaults", file=sys.stderr)

    sr_bench = float(os.environ.get("SR_BENCHMARK", str(cfg_sr)))
    min_n = int(os.environ.get("PSR_MIN_N", str(cfg_n)))
    return sr_bench, min_n


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
    sr_benchmark, min_n = _load_psr_settings()
    window = int(os.environ.get("PSR_WINDOW", DEFAULT_WINDOW))

    returns = _load_recent_returns(PAPER_LOG_PATH, window)
    n = len(returns)
    if n < min_n:
        print(f"PSR live: {n} closed trades, need >={min_n} — insufficient_data")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_obs": n,
            "min_required": min_n,
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
