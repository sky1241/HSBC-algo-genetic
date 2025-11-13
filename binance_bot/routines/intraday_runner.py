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
from bot.state_manager import StateManager
from bot.trade_manager import TradeManager
from bot.risk_manager import RiskManager


def main():
    print("="*70)
    print(f"🔄 INTRADAY RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Charger settings
    settings_path = ROOT / "configs" / "bot_settings.yaml"
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    symbol = settings['symbol']
    timeframe = settings['timeframe']
    state_file = ROOT / settings['state_file']
    
    # Initialiser modules
    try:
        state_mgr = StateManager(str(state_file))
        data_fetcher = DataFetcher(symbol=symbol, timeframe=timeframe)
        signal_engine = SignalEngine(
            max_positions=settings['max_positions_per_side'],
            daily_loss_threshold=0.10
        )
        risk_mgr = RiskManager(
            initial_capital=state_mgr.get('initial_capital_usdt', 1000.0),
            stop_global_pct=settings['stop_global_equity'],
            position_size_pct=settings['position_size_pct']
        )
        # Mode: "simulation" (pas d'ordres) ou "live" (ordres réels)
        # ⚠️ Vérifier .env: BINANCE_TESTNET=false pour compte réel
        trade_mode = "live"  # MODE LIVE ACTIVÉ - Ordres réels sur Binance
        trade_mgr = TradeManager(
            exchange=data_fetcher.exchange,
            symbol=symbol,
            mode=trade_mode
        )
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
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
    
    # Récupérer bougies H2
    print(f"\n📈 Récupération bougies H2...")
    df_ohlc = data_fetcher.get_ohlcv(limit=300)  # 300 bougies pour Ichimoku
    print(f"   {len(df_ohlc)} bougies récupérées")
    
    # Calculer Ichimoku
    df_ichimoku = calculate_ichimoku(
        df_ohlc,
        tenkan=params['tenkan'],
        kijun=params['kijun'],
        senkou_b=params['senkou_b'],
        shift=params['shift']
    )
    
    current_price = float(df_ichimoku.iloc[-1]['close'])
    print(f"   Prix actuel: {current_price:.2f} USDT")
    
    # Charger positions depuis state
    positions_long = state_mgr.get('positions_long', [])
    positions_short = state_mgr.get('positions_short', [])
    daily_loss = state_mgr.get('daily_loss', 0.0)
    
    signal_engine.load_state(positions_long, positions_short, daily_loss)
    
    # Détecter signaux
    signals = signal_engine.detect_signals(df_ichimoku, params, current_price)
    
    if len(signals) == 0:
        print(f"\n✅ Aucun signal (positions: {len(positions_long)} LONG, {len(positions_short)} SHORT)")
        return 0
    
    print(f"\n🔔 {len(signals)} signal(s) détecté(s):")
    
    # Exécuter signaux
    current_equity = state_mgr.get('equity', 1.0)
    capital_usdt = state_mgr.get('initial_capital_usdt', 1000.0) * current_equity
    
    # Vérifier stop global AVANT d'exécuter
    if risk_mgr.check_global_stop(capital_usdt):
        print(f"🛑 STOP GLOBAL ATTEINT (equity={capital_usdt:.2f} USDT <= seuil={risk_mgr.stop_global_threshold:.2f})")
        print(f"   Fermeture de toutes les positions et ARRÊT du bot.")
        # TODO: fermer toutes positions + désactiver bot
        return 3
    
    for sig in signals:
        print(f"\n   Action: {sig['action']}")
        order_id = trade_mgr.execute_signal(sig, capital_usdt)
        
        # Mettre à jour state selon action
        if sig['action'] == 'open_long' and order_id:
            state_mgr.add_position('long', sig['entry'], sig['stop'], sig['tp'], sig.get('size', 0.01))
        elif sig['action'] == 'open_short' and order_id:
            state_mgr.add_position('short', sig['entry'], sig['stop'], sig['tp'], sig.get('size', 0.01))
        elif 'close' in sig['action']:
            state_mgr.remove_position(
                'long' if 'long' in sig['action'] else 'short',
                sig.get('pos_id', '')
            )
    
    # Sauvegarder positions après exécution
    positions_long_new, positions_short_new = signal_engine.get_positions_state()
    state_mgr.set('positions_long', positions_long_new)
    state_mgr.set('positions_short', positions_short_new)
    state_mgr.save()
    
    print(f"\n✅ Run terminé. State sauvegardé.")
    print("="*70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

