"""P-MTF-4 — Job systemd cyclic toutes les 2h : calcul tendance H2 par symbole.

Persiste pour chaque symbole un snapshot {direction, last_close, cloud_top,
cloud_bottom, distance_pct, computed_at_iso, reason} dans
state.json::symbols.<sym>.h2_trend.

Lu par execution_runner_15m (P-MTF-5) via state_mgr.get_symbol_data(sym, "h2_trend").

Exécution :
    python -m routines.h2_trend_runner

Trigger systemd : hsbc-h2-trend.timer (P-MTF-6) à *:01:30 UTC toutes les 2h.

Robustesse : si un symbole échoue (fetch error, NaN cloud, etc.), on
continue avec les autres symboles. Le DataFetcher.get_ohlcv_multi_tf() est
robuste à l'échec d'un TF particulier.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# sys.path injection (cf intraday_runner.py pattern)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binance_bot"))

from binance_bot.bot.state_manager import StateManager
from binance_bot.services.data_fetcher import DataFetcher
from binance_bot.services.ichimoku_engine import calculate_ichimoku
from src.trend_filter_h2 import classify_h2_trend


SETTINGS_PATH = ROOT / "binance_bot" / "configs" / "bot_settings.yaml"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_H2_PARAMS = {  # fallback si state.json::params_today absent
    "tenkan": 27,
    "kijun": 100,
    "senkou_b": 180,
    "shift": 93,
}


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_symbols(settings: dict) -> list[str]:
    """Récupère la liste de symboles à processer (mêmes que intraday).

    Format yaml supporté :
      symbols:
        - { pair: "BTC/USDT", leverage: 125 }    # format réel bot_settings.yaml
        - { symbol: "ETH/USDT" }                  # format alternatif (compat)
        - "SOL/USDT"                              # string brut (compat)
    """
    syms_cfg = settings.get("symbols")
    if isinstance(syms_cfg, list) and syms_cfg:
        out = []
        for s in syms_cfg:
            if isinstance(s, str):
                out.append(s)
            elif isinstance(s, dict):
                # Le format réel utilise "pair" (cf bot_settings.yaml)
                pair = s.get("pair") or s.get("symbol")
                if pair:
                    out.append(pair)
        if out:
            return out
    fallback = settings.get("symbol")
    if fallback:
        return [fallback]
    return DEFAULT_SYMBOLS


def _get_h2_params(state_mgr: StateManager, settings: dict) -> dict:
    """Récupère les params Ichimoku H2.

    Priorité :
        1. state.json::params_today (set par daily_phase_job depuis K3)
        2. settings.yaml::params_h2 (si présent — pas dans default config)
        3. DEFAULT_H2_PARAMS (phase 0 du backtest)
    """
    p = state_mgr.get("params_today")
    if isinstance(p, dict) and "tenkan" in p:
        return p
    p = settings.get("params_h2")
    if isinstance(p, dict) and "tenkan" in p:
        return p
    return DEFAULT_H2_PARAMS


def compute_and_persist_h2_trend(symbol: str, state_mgr: StateManager,
                                 params_h2: dict, margin_pct: float = 0.0) -> dict:
    """Calcule + persiste le snapshot H2 trend pour un symbole.

    Returns le snapshot pour le caller (logging).
    Lève une exception si fetch ou calcul fail (caller décide).
    """
    fetcher = DataFetcher(symbol, "2h")
    # On utilise multi_tf avec un seul TF pour bénéficier du cache class-level
    # (h2_trend_runner appelé toutes les 2h, cache TTL 1h → resync mid-bar OK)
    dfs = fetcher.get_ohlcv_multi_tf(["2h"], limit=300, use_cache=False)
    if "2h" not in dfs:
        raise RuntimeError(f"DataFetcher returned no '2h' data for {symbol}")
    df_h2 = dfs["2h"]
    df_ich = calculate_ichimoku(
        df_h2,
        tenkan=int(params_h2["tenkan"]),
        kijun=int(params_h2["kijun"]),
        senkou_b=int(params_h2["senkou_b"]),
        shift=int(params_h2["shift"]),
    )
    snapshot = classify_h2_trend(df_ich, margin_pct=margin_pct)
    state_mgr.set_symbol_data(symbol, "h2_trend", snapshot)
    return snapshot


def main() -> int:
    settings = _load_settings()
    state_file = ROOT / "binance_bot" / settings.get("state_file", "data/state.json")
    state_mgr = StateManager(str(state_file))

    symbols = _resolve_symbols(settings)
    params_h2 = _get_h2_params(state_mgr, settings)
    margin_pct = float(settings.get("h2_trend_margin_pct", 0.0))

    print(f"=== h2_trend_runner — {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Symbols: {symbols}")
    print(f"Params H2: {params_h2}")
    print(f"Margin pct: {margin_pct}")

    successes = 0
    failures: list[tuple[str, str]] = []
    for symbol in symbols:
        try:
            snap = compute_and_persist_h2_trend(symbol, state_mgr, params_h2, margin_pct)
            close_str = (
                f"{snap['last_close']:.2f}" if snap.get('last_close') is not None else "N/A"
            )
            print(
                f"  [{symbol}] direction={snap['direction']:5s} "
                f"close={close_str} reason={snap['reason']}"
            )
            successes += 1
        except Exception as e:
            print(f"  [{symbol}] FAIL: {type(e).__name__}: {e}")
            failures.append((symbol, str(e)))

    print(f"\nDone: {successes} ok, {len(failures)} fail")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
