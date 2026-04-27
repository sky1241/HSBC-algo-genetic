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

        # Une instance SignalEngine par symbole (state isolé)
        signal_engine = SignalEngine(
            max_positions=max_pos_per_symbol,
            daily_loss_threshold=0.10,
            atr_trailing_mult=float(settings.get('atr_trailing_multiplier', 2.0)),
        )
        signal_engine.load_state(positions_long, positions_short, daily_loss)

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

