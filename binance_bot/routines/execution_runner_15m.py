"""P-MTF-5 — Execution runner 15m indexé tendance H2.

Job systemd cyclic toutes les 15 min (déclenché par hsbc-execution-15m.timer
en P-MTF-6). Pour chaque symbole :
  1. Lit le snapshot tendance H2 depuis state.json::symbols.<sym>.h2_trend
     (persisté par h2_trend_runner toutes les 2h en P-MTF-4).
  2. Skip si stale (> h2_trend_max_age_hours, default 3h).
  3. Fetch bougies 15m via DataFetcher.get_ohlcv_multi_tf (cache TTL 60s).
  4. Calcule Ichimoku 15m avec params_15m du yaml.
  5. Construit un trend_gate_fn closure pour SignalEngine.
  6. Réutilise toutes les factories d'intraday_runner pour les autres
     callbacks (portfolio_scale_fn, var_gate_fn, regime_gate_fn,
     vpin_data_fn, composite_log_fn).
  7. Détecte signaux 15m + applique trend_gate H2 + autres gates.
  8. Exécute les signaux via TradeManager (mêmes positions persistées
     dans state.json::symbols.<sym>.positions_{long,short}).

Inclut la logique flat EOD (P-MTF-9) : si flat_eod_enabled=true et
HH:flat_eod_minute_utc UTC atteint, ferme toutes les positions ouvertes
avant minuit.

Exécution :
    python -m routines.execution_runner_15m

Garde-fous :
- Skip propre si multi_tf_enabled=false dans yaml (rollback en config).
- Skip propre si kill_switch présent (data/.killed).
- Skip par symbole si h2_trend stale ou flat (pas de trade autorisé).
- Toutes les exceptions par symbole sont loguées et n'arrêtent pas le run.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# sys.path injection (cf intraday_runner.py pattern)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "binance_bot"))

from binance_bot.bot.state_manager import StateManager
from binance_bot.services.data_fetcher import DataFetcher
from binance_bot.services.ichimoku_engine import calculate_ichimoku
from binance_bot.services.signal_engine import SignalEngine
from src.trend_filter_h2 import is_h2_trend_stale

# Réutilise les factories d'intraday_runner (DRY)
from binance_bot.routines.intraday_runner import (
    _load_portfolio_features,
    _make_portfolio_scale_fn,
    _make_regime_gate_fn,
    _make_var_gate_fn,
    _make_vpin_data_fn,
    _make_vpin_gate_config,
    _make_composite_log_fn,
)
from binance_bot.routines.h2_trend_runner import _resolve_symbols


SETTINGS_PATH = ROOT / "binance_bot" / "configs" / "bot_settings.yaml"
DEFAULT_15M_PARAMS = {
    "tenkan": 9,
    "kijun": 26,
    "senkou_b": 52,
    "shift": 26,
    "atr_mult": 5.0,
    "tp_mult": 3.0,
}


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_trend_gate_fn(state_mgr: StateManager, symbol: str, max_age_hours: float):
    """Closure : retourne le snapshot H2 trend pour `symbol`, ou flat si stale.

    Note importante : default-arg trick pour capturer `symbol` proprement
    en boucle (sinon toutes les closures partagent la dernière valeur).
    """
    def gate(s=symbol, sm=state_mgr, max_age=max_age_hours):
        h2 = sm.get_symbol_data(s, "h2_trend")
        if is_h2_trend_stale(h2, max_age_hours=max_age):
            return {"direction": "flat", "reason": "h2_stale"}
        return h2 or {"direction": "flat", "reason": "h2_missing"}
    return gate


def _maybe_flat_eod(
    settings: dict,
    state_mgr: StateManager,
    symbols: list[str],
    now_utc: Optional[datetime] = None,
) -> list[dict]:
    """P-MTF-9 — Flat all positions à HH:flat_eod_minute_utc UTC.

    Returns la liste des signaux close émis (pour audit/log).
    """
    if not settings.get("flat_eod_enabled", True):
        return []
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    minute_threshold = int(settings.get("flat_eod_minute_utc", 45))
    if not (now_utc.hour == 23 and now_utc.minute >= minute_threshold):
        return []

    closed: list[dict] = []
    for symbol in symbols:
        for side in ("long", "short"):
            for pos in list(state_mgr.get_positions(side, symbol=symbol)):
                state_mgr.remove_position(side, pos["id"], symbol=symbol)
                closed.append({
                    "symbol": symbol,
                    "side": side,
                    "pos_id": pos["id"],
                    "reason": "flat_eod",
                    "ts_iso": now_utc.isoformat(),
                })
                print(f"  [{symbol}] FLAT EOD: close {side} pos {pos['id']}")
    return closed


def _resolve_capital(state_mgr: StateManager, settings: dict) -> tuple[float, float]:
    """Retourne (capital_usdt, rolling_equity_high) depuis state.json."""
    capital = float(state_mgr.get("initial_capital_usdt", 100.0))
    high = float(state_mgr.get("rolling_equity_high_usdt", capital))
    return capital, high


def main() -> int:
    settings = _load_settings()
    if not settings.get("multi_tf_enabled", True):
        print("multi_tf_enabled=false → execution_runner_15m skipped")
        return 0

    state_file = ROOT / "binance_bot" / settings.get("state_file", "data/state.json")
    state_mgr = StateManager(str(state_file))

    # Kill switch global
    kill_flag = ROOT / "binance_bot" / "data" / ".killed"
    if kill_flag.exists():
        print(f"⛔ Kill switch active ({kill_flag}) → skip cycle")
        return 0

    symbols = _resolve_symbols(settings)
    params_15m = {**DEFAULT_15M_PARAMS, **(settings.get("params_15m") or {})}
    max_age_h = float(settings.get("h2_trend_max_age_hours", 3.0))
    max_pos_per_symbol = int(settings.get("max_positions_per_symbol",
                                          settings.get("max_positions_per_side", 3)))
    capital_usdt, rolling_equity_high_usdt = _resolve_capital(state_mgr, settings)

    # P0/P3 — portfolio features (corr + vol)
    pf_path = ROOT / "binance_bot" / "data" / "portfolio_features.json"
    portfolio_corr_df, portfolio_vol_dict = _load_portfolio_features(pf_path, symbols)
    max_portfolio_risk = float(settings.get("max_portfolio_risk", 0.06))

    # P2 — drawdown scale (anti-martingale)
    def _drawdown_scale_fn():
        try:
            from src.risk_sizing import drawdown_size_multiplier
            cur_equity = float(state_mgr.get("rolling_equity_high_usdt", capital_usdt))
            return float(drawdown_size_multiplier(cur_equity, rolling_equity_high_usdt))
        except Exception:
            return 1.0

    print(f"=== execution_runner_15m — {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Symbols: {symbols}")
    print(f"Params 15m: {params_15m}")
    print(f"H2 trend max age: {max_age_h}h")

    n_signals_total = 0
    n_blocked_by_h2 = 0

    for symbol in symbols:
        try:
            # 1) Check h2_trend (skip si stale)
            h2 = state_mgr.get_symbol_data(symbol, "h2_trend")
            if is_h2_trend_stale(h2, max_age_hours=max_age_h):
                print(f"  [{symbol}] h2_trend stale or missing → skip")
                n_blocked_by_h2 += 1
                continue
            print(f"  [{symbol}] h2_trend={h2['direction']} (close={h2.get('last_close')}) — proceed")

            # 2) Fetch 15m + Ichimoku
            fetcher = DataFetcher(symbol, "15m")
            dfs = fetcher.get_ohlcv_multi_tf(["15m"], limit=300)
            if "15m" not in dfs:
                print(f"  [{symbol}] no 15m data → skip")
                continue
            df_15m = dfs["15m"]
            df_ich = calculate_ichimoku(
                df_15m,
                tenkan=int(params_15m["tenkan"]),
                kijun=int(params_15m["kijun"]),
                senkou_b=int(params_15m["senkou_b"]),
                shift=int(params_15m["shift"]),
            )
            current_price = float(df_ich.iloc[-1]["close"])

            # 3) Build callbacks (réutilise intraday_runner factories)
            portfolio_scale_fn = _make_portfolio_scale_fn(
                state_mgr=state_mgr,
                current_symbol=symbol,
                total_capital=capital_usdt,
                corr_df=portfolio_corr_df,
                vol_dict=portfolio_vol_dict,
                max_portfolio_risk=max_portfolio_risk,
                current_price=current_price,
            )
            var_gate_fn = _make_var_gate_fn(
                state_mgr=state_mgr,
                current_symbol=symbol,
                corr_df=portfolio_corr_df,
                vol_dict=portfolio_vol_dict,
                threshold=float(settings.get("portfolio_var_threshold", 0.08)),
            )
            # Regime gate : besoin de returns 1h pour HAR-RV
            returns_1h_series = None
            try:
                candles_1h = fetcher.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=720)
                if candles_1h:
                    import pandas as _pd
                    _df = _pd.DataFrame(
                        candles_1h, columns=["ts", "open", "high", "low", "close", "volume"]
                    )
                    returns_1h_series = _df["close"].astype(float).pct_change().dropna()
            except Exception as _e:
                print(f"   ⚠️ HAR-RV fetch 1h failed for {symbol}: {_e}")
            regime_gate_fn = _make_regime_gate_fn(returns_1h_series)

            # P-MTF-3 trend gate
            trend_gate_fn = _make_trend_gate_fn(state_mgr, symbol, max_age_h)

            # 4) SignalEngine
            engine = SignalEngine(
                max_positions=max_pos_per_symbol,
                daily_loss_threshold=0.10,
                atr_trailing_mult=float(settings.get("atr_trailing_multiplier", 2.0)),
                portfolio_scale_fn=portfolio_scale_fn,
                daily_loss_soft_cap_pct=float(settings.get("daily_loss_soft_cap_pct", 0.0)),
                daily_gain_soft_cap_pct=float(settings.get("daily_gain_soft_cap_pct", 0.0)),
                daily_loss_hard_cap_pct=float(settings.get("daily_loss_hard_cap_pct", 0.0)),
                kill_switch_path=kill_flag,
                drawdown_scale_fn=_drawdown_scale_fn,
                var_gate_fn=var_gate_fn,
                regime_gate_fn=regime_gate_fn,
                composite_log_fn=_make_composite_log_fn(symbol, ROOT / "binance_bot" / "data"),
                composite_log_path=(ROOT / "binance_bot" / "data" / "flow_composite_log.jsonl"),
                vpin_data_fn=_make_vpin_data_fn(symbol),
                vpin_gate_config=_make_vpin_gate_config(settings),
                vpin_state_dict=state_mgr.get_symbol_data(symbol, "vpin_state"),
                vpin_event_log_path=(ROOT / "binance_bot" / "data" / "vpin_events.jsonl"),
                trend_gate_fn=trend_gate_fn,  # P-MTF-3
            )
            positions_long = state_mgr.get_positions("long", symbol=symbol)
            positions_short = state_mgr.get_positions("short", symbol=symbol)
            engine.load_state(
                positions_long=positions_long,
                positions_short=positions_short,
                daily_loss=float(state_mgr.get("daily_loss", 0.0)),
                daily_pnl_pct=float(state_mgr.get("daily_pnl_pct", 0.0)),
                block_until_iso=state_mgr.get("block_until_iso"),
            )

            # 5) Detect signals 15m
            signals = engine.detect_signals(
                df_ich,
                current_price=current_price,
                params=params_15m,
            )
            n_signals_total += len(signals)
            if signals:
                print(f"  [{symbol}] {len(signals)} signal(s) émis : {[s['action'] for s in signals]}")
            else:
                print(f"  [{symbol}] 0 signal (gates ou pas de cassure)")

            # NOTE : l'exécution réelle (TradeManager) n'est pas faite ici dans
            # P-MTF-5. Elle sera ajoutée en P-MTF-11 lors du déploiement testnet
            # après validation du pipeline en mode dry-run pendant 24h.

        except Exception as e:
            print(f"  [{symbol}] EXCEPTION: {type(e).__name__}: {e}")

    # 6) Flat EOD (P-MTF-9)
    closed_eod = _maybe_flat_eod(settings, state_mgr, symbols)
    if closed_eod:
        print(f"\n=== FLAT EOD : {len(closed_eod)} positions fermées ===")

    print(f"\nDone: {n_signals_total} signaux émis, {n_blocked_by_h2} symboles skip (h2 stale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
