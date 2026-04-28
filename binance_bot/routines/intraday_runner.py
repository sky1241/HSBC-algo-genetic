#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intraday Runner: exécuté toutes les 2h pour détecter signaux et passer ordres.

Workflow:
1. Charge state.json (phase, params, positions)
2. Récupère dernière bougie H2 depuis Binance
3. Calcule Ichimoku avec params du jour
4. Détecte signaux (open/close, TP, SL)
5. Exécute ordres via TradeManager
6. Met à jour state (positions, equity)
7. Log actions
"""
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Ajouter binance_bot au path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.data_fetcher import DataFetcher  # CCXT (simple et fiable)
from services.ichimoku_engine import calculate_ichimoku
from services.signal_engine import SignalEngine
from services.regime_filter import RegimeFilterConfig, filter_entry_signals_inplace
from bot.state_manager import StateManager
from bot.trade_manager import TradeManager
from bot.risk_manager import RiskManager
from bot.reconciler import reconcile_positions
from bot.kill_switch import is_killed, read_kill_reason, trigger_kill
from bot.notifier import TelegramNotifier
from bot.audit_log import AuditLog
from bot.paper_trader import PaperTrader
from bot.recovery_wal import WAL


# P0 (2026-04-28) — Portfolio-aware sizing helpers ----------------------------

# Defaults conservateurs MDPI 2025 ("Cryptocurrency Market Maturation and
# Evolving Risk Profiles") : en stress regime, BTC/ETH co-corr ~0.85,
# BTC/SOL ~0.80, vol annualisée 60-95%. Utilisés tant que pas de cron
# de calcul live des corrélations rolling 30j.
_DEFAULT_PORTFOLIO_CORR = {
    ("BTC/USDT", "ETH/USDT"): 0.85,
    ("BTC/USDT", "SOL/USDT"): 0.80,
    ("ETH/USDT", "SOL/USDT"): 0.85,
}
_DEFAULT_PORTFOLIO_VOL = {
    "BTC/USDT": 0.60,
    "ETH/USDT": 0.75,
    "SOL/USDT": 0.95,
}


def _load_portfolio_features(path, symbols):
    """Charge corrélations + volatilités, fallback sur defaults conservateurs.

    Returns:
        (corr_df, vol_dict) — corr_df: DataFrame symbol×symbol Pearson;
        vol_dict: {symbol: vol_ann_fraction}.
    """
    import json
    import numpy as np
    import pandas as pd

    # Toujours initialiser corr complète et symétrique avec defaults
    n = len(symbols)
    corr_df = pd.DataFrame(np.eye(n), index=symbols, columns=symbols)
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i == j:
                continue
            key = (si, sj) if (si, sj) in _DEFAULT_PORTFOLIO_CORR else (sj, si)
            corr_df.iloc[i, j] = float(_DEFAULT_PORTFOLIO_CORR.get(key, 0.7))

    vol_dict = {s: float(_DEFAULT_PORTFOLIO_VOL.get(s, 0.80)) for s in symbols}

    # Override par fichier persistent si présent (calcul live futur)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            file_corr = data.get("correlations", {})
            for si in symbols:
                for sj in symbols:
                    if si in file_corr and sj in file_corr[si]:
                        corr_df.loc[si, sj] = float(file_corr[si][sj])
            file_vol = data.get("volatilities", {})
            for s in symbols:
                if s in file_vol:
                    vol_dict[s] = float(file_vol[s])
        except Exception as e:
            print(f"⚠️ portfolio_features.json invalid ({e}) — using defaults")

    return corr_df, vol_dict


def _make_portfolio_scale_fn(state_mgr,
                             current_symbol,
                             total_capital,
                             corr_df,
                             vol_dict,
                             max_portfolio_risk,
                             current_price):
    """Construit le callback `portfolio_scale_fn` pour un symbole donné.

    Le callback retourne un facteur [0, 1] basé sur le risk agrégé déjà utilisé
    par les positions ouvertes des AUTRES symboles (le symbole courant est exclu
    pour éviter le double-comptage avant ouverture nouvelle position).
    """
    def _scale_fn():
        try:
            from src.vol_targeting import compute_aggregated_var  # type: ignore
        except ImportError:
            try:
                from vol_targeting import compute_aggregated_var  # type: ignore
            except ImportError:
                return 1.0  # fallback safe si import échoue

        # Construire open_positions agrégé (tous symboles SAUF current_symbol)
        open_positions = {}
        symbols_dict = state_mgr.state.get("symbols", {}) or {}
        for sym, sym_state in symbols_dict.items():
            if sym == current_symbol:
                continue  # éviter double-comptage
            longs = sym_state.get("positions_long", []) or []
            shorts = sym_state.get("positions_short", []) or []
            # Notional USDT approximé : `size` est stocké en fraction du capital
            # (0.01 = 1%), donc notional ≈ size × total_capital (hypothèse 1x).
            # Approximation OK pour P0; un futur chunk pourra raffiner avec leverage.
            cap_factor = float(total_capital) if total_capital > 0 else 100.0
            for p in longs:
                notional = float(p.get("size", 0)) * cap_factor
                if notional > 0:
                    open_positions.setdefault(sym, {"side": "long", "notional": 0.0})
                    open_positions[sym]["notional"] += notional
            for p in shorts:
                notional = float(p.get("size", 0)) * cap_factor
                if notional > 0:
                    open_positions.setdefault(sym, {"side": "short", "notional": 0.0})
                    open_positions[sym]["notional"] += notional

        if not open_positions or total_capital <= 0:
            return 1.0  # pas d'autres positions → pas de pénalisation

        sigma_pf = compute_aggregated_var(
            open_positions=open_positions,
            rolling_correlations=corr_df,
            rolling_volatilities=vol_dict,
            total_capital=float(total_capital),
        )
        if max_portfolio_risk <= 0:
            return 0.0
        remaining = max(0.0, max_portfolio_risk - sigma_pf)
        return float(min(1.0, remaining / max_portfolio_risk))

    return _scale_fn


def _make_regime_gate_fn(returns_1h_series):
    """P4 — Factory du callback regime_gate_fn() -> "low"/"mid"/"high".

    Fit HAR-RV à chaque appel (lazy). Si returns_1h_series None ou trop court,
    retourne "mid" (no block). Caching basique : on ne re-fit pas si appelé
    plusieurs fois — la closure capture le résultat.
    """
    cached_regime = {"value": None}

    def _gate():
        if cached_regime["value"] is not None:
            return cached_regime["value"]
        if returns_1h_series is None or len(returns_1h_series) < 200:
            cached_regime["value"] = "mid"
            return "mid"
        try:
            try:
                from src.har_rv import (  # type: ignore
                    fit_har_rv, predict_rv, classify_regime, realized_volatility,
                )
            except ImportError:
                from har_rv import (  # type: ignore
                    fit_har_rv, predict_rv, classify_regime, realized_volatility,
                )
        except ImportError:
            cached_regime["value"] = "mid"
            return "mid"
        try:
            params = fit_har_rv(returns_1h_series)
            rv_h = float(realized_volatility(returns_1h_series, window=1).iloc[-1])
            rv_d = float(realized_volatility(returns_1h_series, window=5).iloc[-1])
            rv_w = float(realized_volatility(returns_1h_series, window=22).iloc[-1])
            rv_pred = predict_rv(params, rv_h, rv_d, rv_w)
            historical_rv_h = realized_volatility(returns_1h_series, window=1).dropna()
            label = classify_regime(rv_pred, historical_rv_h)
            cached_regime["value"] = label
            return label
        except Exception:
            cached_regime["value"] = "mid"
            return "mid"
    return _gate


def _make_var_gate_fn(state_mgr, current_symbol, corr_df, vol_dict, threshold):
    """P3 — Factory du callback var_gate_fn(side, notional_pct) -> (allowed, reason).

    Lit l'état actuel des positions de tous les autres symboles + ajoute la
    position projetée pour current_symbol, calcule VaR95, compare au seuil.
    """
    def _gate(side, notional_pct):
        try:
            try:
                from src.portfolio_risk_gate import can_enter_new_position  # type: ignore
            except ImportError:
                from portfolio_risk_gate import can_enter_new_position  # type: ignore
        except ImportError:
            return True, ""  # safe fallback: si module absent, autoriser

        # Construire open_positions agrégé (tous symboles, current inclus si déjà ouvert)
        open_positions = {}
        symbols_dict = state_mgr.state.get("symbols", {}) or {}
        for sym, sym_state in symbols_dict.items():
            longs = sym_state.get("positions_long", []) or []
            shorts = sym_state.get("positions_short", []) or []
            for p in longs:
                np_ = float(p.get("size", 0))
                if np_ > 0:
                    if sym not in open_positions:
                        open_positions[sym] = {"side": "long", "notional_pct": 0.0}
                    open_positions[sym]["notional_pct"] += np_
            for p in shorts:
                np_ = float(p.get("size", 0))
                if np_ > 0:
                    if sym not in open_positions:
                        open_positions[sym] = {"side": "short", "notional_pct": 0.0}
                    open_positions[sym]["notional_pct"] += np_

        return can_enter_new_position(
            symbol=current_symbol,
            side=side,
            notional_pct=float(notional_pct),
            current_state={
                "open_positions": open_positions,
                "rolling_correlations": corr_df,
                "rolling_volatilities": vol_dict,
            },
            threshold=float(threshold),
        )
    return _gate


# -----------------------------------------------------------------------------


def main():
    print("="*70)
    print(f"🔄 INTRADAY RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # B14/B15 — Init notifier + audit log (no-op si pas configurés)
    notifier = TelegramNotifier()
    audit = AuditLog(ROOT / "data" / "trades_audit.jsonl")
    chain_status = audit.verify_chain()
    if not chain_status.valid:
        print(f"⚠️ Audit chain INVALID: {chain_status.error}")
        notifier.critical(f"Audit chain corruption — {chain_status.error}")
    audit.append({"event": "boot", "ts": datetime.now().isoformat()})

    # B12 — WAL: vérifier les intents en attente (crash recovery diagnostic, pas de replay auto)
    wal = WAL(ROOT / "data" / "intents.wal.jsonl")
    pending = wal.pending_intents()
    if pending:
        print(f"⚠️ {len(pending)} intent(s) WAL pending au boot — possible crash mid-trade")
        for p in pending[:5]:
            print(f"   intent_id={p.get('intent_id')} action={p.get('action')}")
        notifier.warn(f"{len(pending)} pending WAL intent(s) — investiguer trades_audit.jsonl")
        audit.append({"event": "wal_pending_at_boot", "n": len(pending),
                       "ids": [p.get('intent_id') for p in pending[:10]]})

    # B7 kill switch: refuser de tourner si le bot a été killed
    kill_flag = ROOT / "data" / ".killed"
    if is_killed(kill_flag):
        reason = read_kill_reason(kill_flag)
        print(f"🛑 BOT KILLED — refus de démarrer.")
        print(f"   Raison: {reason}")
        print(f"   Pour réactiver: rm {kill_flag}")
        notifier.critical(f"Boot refusé — kill flag actif: {reason[:200]}")
        audit.append({"event": "boot_refused_killed", "reason": reason[:200]})
        return 99

    # Charger settings
    settings_path = ROOT / "configs" / "bot_settings.yaml"
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    timeframe = settings['timeframe']
    state_file = ROOT / settings['state_file']

    # Multi-symbole : settings['symbols'] = liste de {pair, leverage}
    # Fallback legacy mono : utilise settings['symbol'] + settings['max_leverage']
    sym_cfgs = settings.get('symbols')
    if not sym_cfgs:
        sym_cfgs = [{
            "pair": settings['symbol'],
            "leverage": float(settings.get('max_leverage', 1.0)),
        }]

    max_pos_per_symbol = int(settings.get('max_positions_per_symbol',
                                          settings.get('max_positions_per_side', 3)))

    # Mode trade global
    import os
    trade_mode = (
        os.environ.get('TRADE_MODE')
        or settings.get('trade_mode')
        or "simulation"
    )
    if trade_mode not in ("simulation", "live"):
        print(f"❌ trade_mode invalide: {trade_mode!r}, fallback simulation")
        trade_mode = "simulation"
    print(f"⚙️  trade_mode = {trade_mode}")

    # Init state + risk (partagé entre symboles)
    try:
        state_mgr = StateManager(str(state_file))
        risk_mgr = RiskManager(
            initial_capital=state_mgr.get('initial_capital_usdt', 100.0),
            stop_global_pct=settings['stop_global_equity'],
            position_size_pct=settings['position_size_pct']
        )
    except Exception as e:
        print(f"❌ Erreur initialisation state/risk: {e}")
        return 1
    
    # Vérifier que daily_phase_job a tourné aujourd'hui
    state_date = state_mgr.get('date')
    today_str = datetime.now().date().isoformat()
    if state_date != today_str:
        print(f"⚠️ Daily phase job n'a pas tourné aujourd'hui!")
        print(f"   State date: {state_date} vs today: {today_str}")
        print(f"   Exécuter: python routines/daily_phase_job.py")
        return 2

    phase = state_mgr.get('phase_today')
    params = state_mgr.get('params_today')

    if phase is None or params is None:
        print(f"❌ Phase ou params manquants dans state.json")
        return 1

    print(f"\n🎯 Phase du jour: {phase}")
    print(f"📊 Paramètres: tenkan={params['tenkan']}, shift={params['shift']}, atr_mult={params['atr_mult']}, tp_mult={params['tp_mult']}")
    print(f"📦 Symboles à traiter: {', '.join(c['pair'] for c in sym_cfgs)}")
    print(f"   (mêmes phase/params appliqués à tous — option A: réutilisation labels K3 BTC)")

    current_equity = state_mgr.get('equity', 1.0)
    capital_usdt = state_mgr.get('initial_capital_usdt', 100.0) * current_equity

    # Stop global AVANT toute exécution (vérifié au niveau compte, pas par-symbole)
    if risk_mgr.check_global_stop(capital_usdt):
        msg = (
            f"global_stop atteint: equity={capital_usdt:.2f} USDT "
            f"<= seuil={risk_mgr.stop_global_threshold:.2f}"
        )
        print(f"🛑 STOP GLOBAL ATTEINT — {msg}")
        actions = trigger_kill(
            kill_flag, msg,
            state_mgr=state_mgr, trade_mgr=None,
            current_price=0.0, capital_usdt=capital_usdt,
        )
        for a in actions:
            print(f"   - {a}")
        print(f"   Bot KILLED. Pour réactiver: rm {kill_flag}")
        notifier.critical(f"KILL SWITCH FIRED — {msg}\\n" + "\\n".join(actions[:5]))
        audit.append({"event": "kill_switch", "reason": msg, "actions": actions, "equity": capital_usdt})
        return 3

    paper = PaperTrader(ROOT / "data" / "paper_log.csv")
    daily_loss = state_mgr.get('daily_loss', 0.0)
    rf_cfg = RegimeFilterConfig(**(settings.get('regime_filter') or {}))

    # P0 — Portfolio features pour Kelly portfolio-aware (corrélations + volatilités).
    # Sources :
    #   1. data/portfolio_features.json si présent (calculé par cron quotidien futur)
    #   2. Sinon defaults conservateurs MDPI 2025 (stress regime BTC+ETH ~0.85, +SOL ~0.80)
    portfolio_features_path = ROOT / "data" / "portfolio_features.json"
    portfolio_corr_df, portfolio_vol_dict = _load_portfolio_features(portfolio_features_path,
                                                                     symbols=[c['pair'] for c in sym_cfgs])
    max_portfolio_risk = float(settings.get('max_portfolio_risk', 0.06))

    # P2 — Anti-martingale: track rolling_equity_high + drawdown circuit breaker.
    # Update à chaque run: max(rolling_equity_high, capital_usdt). Refinement
    # futur: vraie fenêtre 90j rolling (current = lifetime high simplifié).
    rolling_equity_high_usdt = float(state_mgr.get('rolling_equity_high_usdt', 0.0) or 0.0)
    if capital_usdt > rolling_equity_high_usdt:
        rolling_equity_high_usdt = capital_usdt
        state_mgr.set('rolling_equity_high_usdt', rolling_equity_high_usdt)

    # Construire drawdown_scale_fn (callback sans args, lazy import risk_sizing)
    def _make_drawdown_scale_fn(_capital_usdt, _rolling_high):
        def _fn():
            try:
                from src.risk_sizing import drawdown_size_multiplier  # type: ignore
            except ImportError:
                try:
                    from risk_sizing import drawdown_size_multiplier  # type: ignore
                except ImportError:
                    return 1.0
            return float(drawdown_size_multiplier(_capital_usdt, _rolling_high))
        return _fn

    drawdown_scale_fn = _make_drawdown_scale_fn(capital_usdt, rolling_equity_high_usdt)

    # P2 — Vérifier le drawdown circuit breaker UNE FOIS au boot (avant la
    # boucle symbols). Si dd < -15% sur le compte global, on déclenche
    # kill_switch + Telegram CRITICAL + audit (raison spécifique drawdown).
    if rolling_equity_high_usdt > 0:
        _dd_pct = (capital_usdt - rolling_equity_high_usdt) / rolling_equity_high_usdt
        if _dd_pct < -0.15:
            _msg = (
                f"DRAWDOWN CIRCUIT BREAKER: equity={capital_usdt:.2f} USDT "
                f"vs rolling_high={rolling_equity_high_usdt:.2f} USDT "
                f"(dd={_dd_pct*100:.2f}% < -15.00%)"
            )
            print(f"🛑 {_msg}")
            audit.append({
                "event": "drawdown_circuit_breaker",
                "current_equity_usdt": float(capital_usdt),
                "rolling_equity_high_usdt": float(rolling_equity_high_usdt),
                "dd_pct": float(_dd_pct),
            })
            try:
                notifier.critical(_msg)
            except Exception:
                pass
            from bot.kill_switch import trigger_kill as _trigger_kill
            _trigger_kill(kill_flag, _msg)
            print(f"   Bot KILLED. Pour réactiver: rm {kill_flag}")
            return 4

    overall_signals = 0
    for sym_cfg in sym_cfgs:
        symbol = sym_cfg['pair']
        leverage = float(sym_cfg.get('leverage', settings.get('max_leverage', 1.0)))
        print(f"\n{'='*70}\n📍 Symbole: {symbol} (leverage={leverage:.0f}x)\n{'='*70}")

        try:
            data_fetcher = DataFetcher(symbol=symbol, timeframe=timeframe)
            # Détection hedge/oneway côté compte (idem pour tous les symboles)
            try:
                dual = data_fetcher.exchange.fapiPrivateGetPositionSideDual()
                is_hedge = dual.get('dualSidePosition') in (True, 'true')
            except Exception as e:
                print(f"⚠️ Détection position_mode failed: {e} → fallback hedge")
                is_hedge = True
            pos_mode = "hedge" if is_hedge else "oneway"

            trade_mgr = TradeManager(
                exchange=data_fetcher.exchange,
                symbol=symbol,
                mode=trade_mode,
                leverage=leverage,
                position_mode=pos_mode,
            )
            trade_mgr.apply_leverage_on_exchange()

            if trade_mode == "live":
                # reconcile_positions ne supporte pas encore symbol scoping, on garde la signature legacy
                # (positions reconciliées vont dans state.symbols[symbol])
                reconcile_positions(state_mgr, data_fetcher.exchange, symbol)
        except Exception as e:
            print(f"❌ Init {symbol} failed: {e} — skip")
            audit.append({"event": "symbol_init_failed", "symbol": symbol, "error": str(e)})
            continue

        # Récupérer bougies + Ichimoku
        df_ohlc = data_fetcher.get_ohlcv(limit=300)
        df_ichimoku = calculate_ichimoku(
            df_ohlc,
            tenkan=params['tenkan'],
            kijun=params['kijun'],
            senkou_b=params['senkou_b'],
            shift=params['shift']
        )
        current_price = float(df_ichimoku.iloc[-1]['close'])
        print(f"   Prix: {current_price:.4f}, bougies: {len(df_ohlc)}")

        rf_info = filter_entry_signals_inplace(df_ichimoku, rf_cfg)
        if rf_cfg.enabled:
            audit.append({"event": "regime_filter", "symbol": symbol, **rf_info})

        # Charger positions par symbole (state.symbols[symbol])
        positions_long = state_mgr.get_positions('long', symbol=symbol)
        positions_short = state_mgr.get_positions('short', symbol=symbol)

        # P0 — Construire le callback portfolio_scale_fn (Kelly portfolio-aware)
        # pour ce symbole, en excluant ses propres positions du calcul agrégé.
        portfolio_scale_fn = _make_portfolio_scale_fn(
            state_mgr=state_mgr,
            current_symbol=symbol,
            total_capital=capital_usdt,
            corr_df=portfolio_corr_df,
            vol_dict=portfolio_vol_dict,
            max_portfolio_risk=max_portfolio_risk,
            current_price=current_price,
        )

        # P3 — Construire var_gate_fn(side, notional_pct) -> (allowed, reason)
        # qui BLOQUE les nouvelles entrées si VaR95 projetée > seuil.
        var_gate_fn = _make_var_gate_fn(
            state_mgr=state_mgr,
            current_symbol=symbol,
            corr_df=portfolio_corr_df,
            vol_dict=portfolio_vol_dict,
            threshold=float(settings.get('portfolio_var_threshold', 0.08)),
        )

        # P4 — Construire regime_gate_fn() -> "low"/"mid"/"high" via HAR-RV.
        # On récupère 720 bars 1h = 30 jours via ccxt fetch_ohlcv direct (timeframe
        # override car DataFetcher est lié au timeframe H2 du bot).
        # Si fail (rate limit, etc.), regime="mid" (no block).
        returns_1h_series = None
        try:
            candles_1h = data_fetcher.exchange.fetch_ohlcv(
                symbol, timeframe="1h", limit=720
            )
            if candles_1h:
                import pandas as _pd
                _df = _pd.DataFrame(
                    candles_1h, columns=['ts', 'open', 'high', 'low', 'close', 'volume']
                )
                returns_1h_series = _df['close'].astype(float).pct_change().dropna()
        except Exception as _e:
            print(f"   ⚠️ HAR-RV fetch 1h returns failed for {symbol}: {_e}")
            returns_1h_series = None
        regime_gate_fn = _make_regime_gate_fn(returns_1h_series)

        # Une instance SignalEngine par symbole (state isolé)
        signal_engine = SignalEngine(
            max_positions=max_pos_per_symbol,
            daily_loss_threshold=0.10,
            atr_trailing_mult=float(settings.get('atr_trailing_multiplier', 2.0)),
            portfolio_scale_fn=portfolio_scale_fn,
            # P1 — daily caps + kill_switch + notifier
            daily_loss_soft_cap_pct=float(settings.get('daily_loss_soft_cap_pct', 0.0)),
            daily_gain_soft_cap_pct=float(settings.get('daily_gain_soft_cap_pct', 0.0)),
            daily_loss_hard_cap_pct=float(settings.get('daily_loss_hard_cap_pct', 0.0)),
            kill_switch_path=kill_flag,
            notifier=notifier,
            # P2 — anti-martingale drawdown scaling
            drawdown_scale_fn=drawdown_scale_fn,
            # P3 — VaR95 gate (bloque les entrées au-dessus du seuil)
            var_gate_fn=var_gate_fn,
            # P4 — HAR-RV regime gate (bloque entrées en regime=low)
            regime_gate_fn=regime_gate_fn,
        )
        # P1 — daily_pnl_pct + block_until_iso depuis state global (partagés multi-symbole)
        daily_pnl_pct = float(state_mgr.get('daily_pnl_pct', 0.0))
        block_until_iso = state_mgr.get('block_until_iso', None)
        signal_engine.load_state(
            positions_long, positions_short, daily_loss,
            daily_pnl_pct=daily_pnl_pct,
            block_until_iso=block_until_iso,
        )

        signals = signal_engine.detect_signals(df_ichimoku, params, current_price)

        if not signals:
            print(f"   ✅ Aucun signal ({len(positions_long)}L / {len(positions_short)}S)")
            # Persist quand même les trailing stops mis à jour
            pl_new, ps_new = signal_engine.get_positions_state()
            state_mgr.ensure_symbol(symbol)
            state_mgr.state['symbols'][symbol]['positions_long'] = pl_new
            state_mgr.state['symbols'][symbol]['positions_short'] = ps_new
            state_mgr.save()
            continue

        print(f"   🔔 {len(signals)} signal(s) sur {symbol}")
        overall_signals += len(signals)

        for sig in signals:
            print(f"\n      Action: {sig['action']}")
            signal_id = f"{datetime.now().isoformat()}_{symbol}_{sig['action']}"
            audit.append({"event": "signal_detected", "signal_id": signal_id, "symbol": symbol, "signal": sig})
            order_id = trade_mgr.execute_signal(sig, capital_usdt)
            audit.append({"event": "order_result", "signal_id": signal_id,
                          "symbol": symbol, "order_id": order_id, "action": sig['action']})

            if sig['action'] == 'open_long' and order_id:
                state_mgr.add_position('long', sig['entry'], sig['stop'], sig['tp'],
                                       sig.get('size', 0.01), symbol=symbol)
                qty = capital_usdt * sig.get('size', 0.01) * trade_mgr.leverage / sig['entry']
                paper.log_open('open_long', qty=qty, price=sig['entry'],
                               live_order_id=order_id, signal_id=signal_id)
                notifier.info(f"[{symbol}] open_long {qty:.4f} @ {sig['entry']:.4f}")
            elif sig['action'] == 'open_short' and order_id:
                state_mgr.add_position('short', sig['entry'], sig['stop'], sig['tp'],
                                       sig.get('size', 0.01), symbol=symbol)
                qty = capital_usdt * sig.get('size', 0.01) * trade_mgr.leverage / sig['entry']
                paper.log_open('open_short', qty=qty, price=sig['entry'],
                               live_order_id=order_id, signal_id=signal_id)
                notifier.info(f"[{symbol}] open_short {qty:.4f} @ {sig['entry']:.4f}")
            elif 'close' in sig['action']:
                side = 'long' if 'long' in sig['action'] else 'short'
                existing = next(
                    (p for p in state_mgr.get_positions(side, symbol=symbol) if p.get('id') == sig.get('pos_id', '')),
                    None,
                )
                entry_price = float(existing.get('entry', 0)) if existing else 0.0
                entry_qty = float(existing.get('size', 0)) if existing else 0.0
                opened_at_str = existing.get('opened_at') if existing else None

                state_mgr.remove_position(side, sig.get('pos_id', ''), symbol=symbol)
                exit_p = float(sig.get('exit', current_price))

                held_s = 0.0
                if opened_at_str:
                    try:
                        opened_at = datetime.fromisoformat(opened_at_str.replace('Z', '+00:00') if 'Z' in opened_at_str else opened_at_str)
                        held_s = max(0.0, (datetime.now(opened_at.tzinfo or None) - opened_at).total_seconds())
                    except Exception:
                        held_s = 0.0

                if entry_price > 0 and entry_qty > 0:
                    qty_native = capital_usdt * entry_qty * trade_mgr.leverage / entry_price
                    paper.log_close(
                        sig['action'], qty=qty_native, exit_price=exit_p, entry_price=entry_price,
                        live_order_id=order_id, signal_id=signal_id, held_seconds=held_s,
                        notes=f"{symbol} {sig.get('reason', '')}",
                    )
                notifier.info(f"[{symbol}] {sig['action']} reason={sig.get('reason')} @ {exit_p:.4f}")

        # P1 — Si SignalEngine a déclenché un soft cap, persister le block_until_iso
        # global (partagé multi-symbole : un cap déclenché par n'importe quel symbol
        # bloque toutes les nouvelles entrées).
        if signal_engine.block_until_iso is not None:
            existing = state_mgr.get('block_until_iso', None)
            # Garder le timestamp le plus tardif (sécurité)
            if existing is None or signal_engine.block_until_iso > existing:
                state_mgr.set('block_until_iso', signal_engine.block_until_iso)

        # Persister état signal_engine -> state.symbols[symbol]
        pl_new, ps_new = signal_engine.get_positions_state()
        state_mgr.ensure_symbol(symbol)
        state_mgr.state['symbols'][symbol]['positions_long'] = pl_new
        state_mgr.state['symbols'][symbol]['positions_short'] = ps_new
        state_mgr.save()

    print(f"\n{'='*70}\n✅ Run terminé. {overall_signals} signal(s) total. State sauvegardé.\n{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

