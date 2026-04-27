#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSBC bot dashboard.

Flask app exposing a local UI on http://localhost:8080 to monitor the
Ichimoku K3 bot running on Binance USDM testnet:
 - Bot state (mode, leverage, kill switch, phase, params)
 - Live USDT balance, BTC/USDT price
 - Open futures positions + algo orders (SL/TP) via TradeManager
 - P&L vs initial capital
 - Tail of intraday/daily logs
 - Button to force an intraday run via systemctl --user

Server-side only Binance calls — API keys never leave the host.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, jsonify, render_template, request

# Ensure we can import binance_bot.* when launched as `python -m binance_bot.dashboard.app`
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy imports so that importing the module never crashes if a dep is missing
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


CONFIG_PATH = ROOT / "configs" / "bot_settings.yaml"
STATE_PATH = ROOT / "data" / "state.json"
KILL_FLAG = ROOT / "data" / ".killed"
INTRADAY_LOG = ROOT / "logs" / "intraday.log"
DAILY_LOG = ROOT / "logs" / "daily.log"

# Cache to avoid hammering Binance every refresh tick
_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_CACHE_TTL = 8.0  # seconds — UI refreshes every 30s anyway
_CACHE_LOCK = threading.Lock()


app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


# -----------------------------
# Helpers (filesystem)
# -----------------------------

def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        return {"_error": f"cannot read {path}: {e}"}


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {"_error": f"cannot read {path}: {e}"}


def _read_log_tail(path: Path, max_blocks: int = 10) -> List[str]:
    """Return the last `max_blocks` runs from a log file (split on '======...').

    Each "block" is delimited by the line of '=' chars used by the runners.
    Returns most-recent-first.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [f"(log read error: {e})"]

    # Split blocks on the equals separator. Keep non-empty trimmed blocks.
    sep = "======================================================================"
    raw_blocks = [b.strip() for b in text.split(sep) if b.strip()]
    # The runner alternates separator/header/separator/body — a block is "header + body"
    # We just take the last N raw chunks; UI displays them as-is.
    blocks = raw_blocks[-max_blocks:]
    return list(reversed(blocks))


def _kill_status() -> Dict[str, Any]:
    if KILL_FLAG.exists():
        try:
            reason = KILL_FLAG.read_text(encoding="utf-8").strip()
        except Exception:
            reason = "(unreadable)"
        return {"killed": True, "reason": reason}
    return {"killed": False, "reason": ""}


# -----------------------------
# Helpers (Binance via ccxt)
# -----------------------------

def _build_exchange_and_trade_mgr():
    """Build a fresh ccxt.binanceusdm instance + TradeManager (testnet aware).

    Returns (exchange, trade_mgr, symbol, settings) or raises.
    """
    import ccxt  # local import to avoid hard dep at module-load time
    from bot.trade_manager import TradeManager

    settings = _safe_load_yaml(CONFIG_PATH)
    symbol = settings.get("symbol", "BTC/USDT")
    leverage = float(settings.get("max_leverage", 1.0))
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        "timeout": 8000,
    })
    if testnet:
        exchange.set_sandbox_mode(True)

    algo_base = (
        "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    )
    trade_mgr = TradeManager(
        exchange=exchange,
        symbol=symbol,
        mode="live",  # we are only READING here, but live unlocks the algo endpoint
        leverage=leverage,
        algo_base_url=algo_base,
    )
    return exchange, trade_mgr, symbol, settings


def _fetch_binance_snapshot() -> Dict[str, Any]:
    """One-shot snapshot of balance/positions/algoOrders/last_price.

    Each sub-call is wrapped — partial failure is fine, we surface it as 'error'
    on that field instead of crashing the whole endpoint.
    """
    snap: Dict[str, Any] = {
        "balance": None,
        "positions": [],
        "algo_orders": [],
        "last_price": None,
        "errors": [],
        "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
    }
    try:
        exchange, trade_mgr, symbol, settings = _build_exchange_and_trade_mgr()
    except Exception as e:
        snap["errors"].append(f"init exchange failed: {e}")
        return snap

    snap["symbol"] = symbol

    # Balance (USDT)
    try:
        bal = exchange.fetch_balance()
        usdt = bal.get("USDT", {}) if isinstance(bal, dict) else {}
        snap["balance"] = {
            "free": float(usdt.get("free") or 0.0),
            "used": float(usdt.get("used") or 0.0),
            "total": float(usdt.get("total") or 0.0),
        }
    except Exception as e:
        snap["errors"].append(f"fetch_balance: {e}")

    # Last price
    try:
        ticker = exchange.fetch_ticker(symbol)
        snap["last_price"] = float(ticker.get("last") or 0.0) or None
    except Exception as e:
        snap["errors"].append(f"fetch_ticker: {e}")

    # Open positions
    try:
        positions_raw = exchange.fetch_positions([symbol])
        positions: List[Dict[str, Any]] = []
        for p in positions_raw or []:
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                continue
            positions.append({
                "symbol": p.get("symbol"),
                "side": (p.get("side") or "").lower(),
                "contracts": contracts,
                "entry": float(p.get("entryPrice") or 0.0),
                "mark": float(p.get("markPrice") or 0.0),
                "unrealized": float(p.get("unrealizedPnl") or 0.0),
                "leverage": float(p.get("leverage") or 0.0),
                "liquidation": float(p.get("liquidationPrice") or 0.0) or None,
            })
        snap["positions"] = positions
    except Exception as e:
        snap["errors"].append(f"fetch_positions: {e}")

    # Algo orders (SL/TP)
    try:
        algo = trade_mgr.fetch_open_algo_orders()
        # Normalise fields for the UI
        norm: List[Dict[str, Any]] = []
        for o in algo or []:
            norm.append({
                "algo_id": o.get("algoId") or o.get("algoid"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("orderType") or o.get("type"),
                "trigger_price": o.get("triggerPrice") or o.get("triggerprice"),
                "quantity": o.get("origQty") or o.get("quantity"),
                "position_side": o.get("positionSide"),
                "status": o.get("algoStatus") or o.get("status"),
            })
        snap["algo_orders"] = norm
    except Exception as e:
        snap["errors"].append(f"fetch_open_algo_orders: {e}")

    return snap


def _get_cached_snapshot(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    with _CACHE_LOCK:
        if (
            not force
            and _CACHE["data"] is not None
            and (now - _CACHE["ts"]) < _CACHE_TTL
        ):
            return _CACHE["data"]
    snap = _fetch_binance_snapshot()
    with _CACHE_LOCK:
        _CACHE["ts"] = now
        _CACHE["data"] = snap
    return snap


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    settings = _safe_load_yaml(CONFIG_PATH)
    state = _safe_load_json(STATE_PATH)
    trade_mode = os.getenv("TRADE_MODE") or settings.get("trade_mode") or "simulation"

    snap = _get_cached_snapshot(force=request.args.get("force") == "1")

    initial_capital = float(state.get("initial_capital_usdt") or 0.0)
    starting_reference = 5000.0  # per the user's brief
    total_balance = (snap.get("balance") or {}).get("total") if snap.get("balance") else None
    pnl_vs_start = None
    pnl_pct_vs_start = None
    if total_balance is not None:
        pnl_vs_start = total_balance - starting_reference
        if starting_reference > 0:
            pnl_pct_vs_start = pnl_vs_start / starting_reference * 100.0

    payload = {
        "now": datetime.now().isoformat(timespec="seconds"),
        "trade_mode": trade_mode,
        "leverage": settings.get("max_leverage"),
        "symbol": settings.get("symbol", "BTC/USDT"),
        "testnet": snap.get("testnet"),
        "kill": _kill_status(),
        "state": {
            "date": state.get("date"),
            "phase_today": state.get("phase_today"),
            "params_today": state.get("params_today"),
            "equity": state.get("equity"),
            "initial_capital_usdt": initial_capital,
            "max_drawdown": state.get("max_drawdown"),
            "daily_loss": state.get("daily_loss"),
            "total_trades": state.get("total_trades"),
            "last_update": state.get("last_update"),
        },
        "binance": {
            "balance": snap.get("balance"),
            "last_price": snap.get("last_price"),
            "positions": snap.get("positions"),
            "algo_orders": snap.get("algo_orders"),
            "errors": snap.get("errors"),
        },
        "pnl": {
            "starting_reference": starting_reference,
            "pnl_vs_start": pnl_vs_start,
            "pnl_pct_vs_start": pnl_pct_vs_start,
        },
        "logs": {
            "intraday": _read_log_tail(INTRADAY_LOG, 10),
            "daily": _read_log_tail(DAILY_LOG, 5),
        },
        "settings": {
            "position_size_pct": settings.get("position_size_pct"),
            "max_positions_per_side": settings.get("max_positions_per_side"),
            "stop_global_equity": settings.get("stop_global_equity"),
            "atr_period": settings.get("atr_period"),
            "atr_trailing_multiplier": settings.get("atr_trailing_multiplier"),
        },
    }
    return jsonify(payload)


@app.route("/api/run-intraday", methods=["POST"])
def api_run_intraday():
    """Triggers `systemctl --user start hsbc-intraday.service`.

    Returns stdout/stderr so the UI can surface failures.
    """
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "start", "hsbc-intraday.service"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return jsonify({
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": f"systemctl not found: {e}"}), 500
    except subprocess.TimeoutExpired as e:
        return jsonify({"ok": False, "error": f"systemctl timeout: {e}"}), 504
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def main():
    host = os.environ.get("HSBC_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("HSBC_DASHBOARD_PORT", "8080"))
    # Disable the reloader so it doesn't double-import on boot under systemd
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
