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

# P-MTF-12 : TradeManager + audit + paper + reconcile pour exécution réelle
from binance_bot.bot.trade_manager import TradeManager
from binance_bot.bot.audit_log import AuditLog
from binance_bot.bot.paper_trader import PaperTrader
from binance_bot.bot.reconciler import reconcile_positions
from binance_bot.bot.notifier import TelegramNotifier

# Réutilise les factories d'intraday_runner (DRY)
from binance_bot.routines.intraday_runner import (
    _load_portfolio_features,
    _make_portfolio_scale_fn,
    _make_regime_gate_fn,
    _make_var_gate_fn,
    _make_vpin_data_fn,
    _make_vpin_gate_config,
    _make_composite_log_fn,
    _build_features_snapshot_at_open,
    _build_meta_context,
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

    # P-MTF-12 : trade_mode + notifier + audit + paper_log + meta_logger
    trade_mode = (os.environ.get("TRADE_MODE")
                  or settings.get("trade_mode") or "simulation")
    notifier = TelegramNotifier()
    audit = AuditLog(ROOT / "binance_bot" / "data" / "trades_audit.jsonl")
    audit.append({
        "event": "boot_execution_runner_15m",
        "ts": datetime.now(timezone.utc).isoformat(),
        "trade_mode": trade_mode,
        "n_symbols": len(symbols),
    })
    try:
        from binance_bot.bot.trade_meta import MetaLabelLogger as _MLL
        meta_logger = _MLL(path=ROOT / "binance_bot" / "data" / "trades_meta.jsonl")
    except Exception:
        meta_logger = None
    paper = PaperTrader(
        ROOT / "binance_bot" / "data" / "paper_log.csv",
        meta_logger=meta_logger,
    )
    sym_cfgs = settings.get("symbols", []) or []
    leverage_by_pair = {
        c["pair"]: int(c.get("leverage", 10))
        for c in sym_cfgs
        if isinstance(c, dict) and "pair" in c
    }

    print(f"=== execution_runner_15m — {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Trade mode: {trade_mode}")
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

            # P-MTF-12 : init TradeManager + reconcile (live only)
            leverage = leverage_by_pair.get(symbol, 10)
            try:
                exch_info = fetcher.exchange.fapiPrivateGetPositionsideDual()
                is_hedge = str(exch_info.get("dualSidePosition", False)).lower() == "true"
            except Exception:
                is_hedge = False
            trade_mgr = TradeManager(
                exchange=fetcher.exchange,
                symbol=symbol,
                mode=trade_mode,
                leverage=leverage,
                position_mode="hedge" if is_hedge else "oneway",
            )
            try:
                trade_mgr.apply_leverage_on_exchange()
            except Exception as _le:
                print(f"   ⚠️ apply_leverage failed for {symbol}: {_le}")
            if trade_mode == "live":
                try:
                    reconcile_positions(state_mgr, fetcher.exchange, symbol)
                except Exception as _re:
                    print(f"   ⚠️ reconcile_positions failed for {symbol}: {_re}")

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

            # P7 / R5 — persist VPIN state per symbole
            try:
                vpin_state_dict = engine.get_vpin_state_dict()
                if vpin_state_dict is not None:
                    state_mgr.set_symbol_data(symbol, "vpin_state", vpin_state_dict)
            except Exception:
                pass

            if not signals:
                print(f"  [{symbol}] 0 signal (gates ou pas de cassure)")
                # Persister positions (trailing stops mis à jour)
                try:
                    pl_new, ps_new = engine.get_positions_state()
                    state_mgr.ensure_symbol(symbol)
                    state_mgr.state["symbols"][symbol]["positions_long"] = pl_new
                    state_mgr.state["symbols"][symbol]["positions_short"] = ps_new
                    state_mgr.save()
                except Exception:
                    pass
                continue

            n_signals_total += len(signals)
            print(f"  [{symbol}] {len(signals)} signal(s) émis : {[s['action'] for s in signals]}")

            # P-MTF-12 — Exécution réelle via TradeManager
            for sig in signals:
                signal_id = f"{datetime.now(timezone.utc).isoformat()}_{symbol}_15m_{sig['action']}"
                audit.append({
                    "event": "signal_detected",
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "tf": "15m",
                    "signal": sig,
                })
                try:
                    order_id = trade_mgr.execute_signal(sig, capital_usdt)
                except Exception as _ee:
                    print(f"   ❌ execute_signal failed: {_ee}")
                    audit.append({
                        "event": "order_exception",
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "error": str(_ee),
                    })
                    continue
                audit.append({
                    "event": "order_result",
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "tf": "15m",
                    "order_id": order_id,
                    "action": sig["action"],
                })
                if sig["action"] in ("open_long", "open_short") and order_id:
                    side = "long" if sig["action"] == "open_long" else "short"
                    try:
                        snap = _build_features_snapshot_at_open(
                            symbol=symbol,
                            df_ichimoku=df_ich,
                            returns_1h_series=returns_1h_series,
                            regime_gate_fn=regime_gate_fn,
                            composite_log_fn=engine.composite_log_fn,
                            vpin_data_fn=engine.vpin_data_fn,
                        )
                    except Exception:
                        snap = None
                    state_mgr.add_position(
                        side, sig["entry"], sig["stop"], sig["tp"],
                        sig.get("size", 0.01),
                        symbol=symbol, features_snapshot=snap,
                    )
                    qty = capital_usdt * sig.get("size", 0.01) * trade_mgr.leverage / sig["entry"]
                    paper.log_open(
                        sig["action"], qty=qty, price=sig["entry"],
                        live_order_id=order_id, signal_id=signal_id,
                    )
                    notifier.info(f"[{symbol} 15m] {sig['action']} {qty:.4f} @ {sig['entry']:.4f}")
                elif "close" in sig["action"]:
                    side = "long" if "long" in sig["action"] else "short"
                    existing = next(
                        (p for p in state_mgr.get_positions(side, symbol=symbol)
                         if p.get("id") == sig.get("pos_id", "")),
                        None,
                    )
                    entry_price = float(existing.get("entry", 0)) if existing else 0.0
                    entry_qty = float(existing.get("size", 0)) if existing else 0.0
                    opened_at_str = existing.get("opened_at") if existing else None
                    state_mgr.remove_position(side, sig.get("pos_id", ""), symbol=symbol)
                    exit_p = float(sig.get("exit", current_price))
                    held_s = 0.0
                    if opened_at_str:
                        try:
                            opened_at = datetime.fromisoformat(
                                opened_at_str.replace("Z", "+00:00") if "Z" in opened_at_str else opened_at_str
                            )
                            held_s = max(0.0, (datetime.now(opened_at.tzinfo or None) - opened_at).total_seconds())
                        except Exception:
                            held_s = 0.0
                    if entry_price > 0 and entry_qty > 0:
                        qty_native = capital_usdt * entry_qty * trade_mgr.leverage / entry_price
                        try:
                            _meta_ctx = _build_meta_context(
                                signal_id=signal_id, symbol=symbol, sig=sig,
                                opened_at_iso=opened_at_str,
                                phase_K3=state_mgr.get("phase_today"),
                                atr_at_entry=float(df_ich.iloc[-1].get("ATR", 0.0)) or None,
                                regime_har=(regime_gate_fn() if regime_gate_fn is not None else None),
                                composite_score=None,
                                entry_features=(existing.get("features_snapshot") if existing else None),
                            )
                        except Exception:
                            _meta_ctx = None
                        paper.log_close(
                            sig["action"], qty=qty_native, exit_price=exit_p, entry_price=entry_price,
                            live_order_id=order_id, signal_id=signal_id, held_seconds=held_s,
                            notes=f"{symbol} 15m {sig.get('reason', '')}",
                            meta_context=_meta_ctx,
                        )
                    notifier.info(f"[{symbol} 15m] {sig['action']} reason={sig.get('reason')} @ {exit_p:.4f}")

        except Exception as e:
            print(f"  [{symbol}] EXCEPTION: {type(e).__name__}: {e}")
            audit.append({
                "event": "symbol_loop_exception",
                "symbol": symbol,
                "error": f"{type(e).__name__}: {e}",
            })

    # 6) Flat EOD (P-MTF-9)
    closed_eod = _maybe_flat_eod(settings, state_mgr, symbols)
    if closed_eod:
        print(f"\n=== FLAT EOD : {len(closed_eod)} positions fermées ===")

    print(f"\nDone: {n_signals_total} signaux émis, {n_blocked_by_h2} symboles skip (h2 stale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
