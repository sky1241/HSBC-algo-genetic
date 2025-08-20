import os
import sys
import json
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import ccxt

# Make parent importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ichimoku_pipeline_web_v4_8_fixed import calculate_ichimoku, log


def read_env_file(dotenv_path: str) -> dict:
    data = {}
    try:
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' in s:
                        k, v = s.split('=', 1)
                        data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return data


def load_baseline(baseline_path: str) -> dict:
    try:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️  Impossible de lire la baseline {baseline_path}: {e}")
        return {}


def fetch_df(exchange, symbol: str, timeframe: str = '2h', limit: int = 600) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv or len(ohlcv) < 10:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df


def ensure_symbol_settings(ex, symbol: str, leverage: int = 10):
    try:
        ex.setLeverage(leverage, symbol)
    except Exception as e:
        log(f"ℹ️ setLeverage({leverage}) échoué pour {symbol}: {e}")
    try:
        ex.setMarginMode('ISOLATED', symbol)
    except Exception as e:
        log(f"ℹ️ setMarginMode(ISOLATED) échoué pour {symbol}: {e}")


def place_protective_stop(ex, symbol: str, side: str, qty: float, stop_price: float):
    # side: 'buy' or 'sell' for the STOP to reduce the opposing direction
    params = {
        'reduceOnly': True,
        'stopPrice': float(stop_price),
        'workingType': 'MARK_PRICE',
        'timeInForce': 'GTC',
    }
    order_side = 'sell' if side == 'buy' else 'buy'
    try:
        ex.create_order(symbol, 'STOP_MARKET', order_side, float(qty), None, params)
        log(f"🛡️ STOP_MARKET placé {symbol} {order_side} qty={qty:.6f} stop={stop_price}")
    except Exception as e:
        log(f"⚠️ STOP_MARKET échec {symbol}: {e}")


def place_trailing_stop(ex, symbol: str, side: str, qty: float, callback_rate_pct: float, activation_price: float | None = None):
    # Binance Futures USDM trailing stop market
    # callbackRate in [0.1, 5]
    cr = max(0.1, min(float(callback_rate_pct), 5.0))
    params = {
        'reduceOnly': True,
        'workingType': 'MARK_PRICE',
        'callbackRate': cr,
        'timeInForce': 'GTC',
    }
    if activation_price is not None and activation_price > 0:
        params['activationPrice'] = float(activation_price)
    order_side = 'sell' if side == 'buy' else 'buy'
    try:
        ex.create_order(symbol, 'TRAILING_STOP_MARKET', order_side, float(qty), None, params)
        log(f"🧲 Trailing STOP placé {symbol} {order_side} qty={qty:.6f} callbackRate={cr:.3f}% act={activation_price}")
        return True
    except Exception as e:
        log(f"⚠️ TRAILING_STOP_MARKET échec {symbol}: {e}")
        return False


def main():
    # Config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'outputs')
    baseline_path = os.path.join(out_dir, 'BEST_BASELINE.json')

    # Load env from file if present
    env_file = os.path.join(root, '.env')
    env_map = read_env_file(env_file)
    # Also accept a local testnet env file in outputs
    env_file_alt = os.path.join(out_dir, 'TESTNET_ENV.txt')
    env_map_alt = read_env_file(env_file_alt)
    env_map.update({k: v for k, v in env_map_alt.items() if k not in env_map})
    for k, v in env_map.items():
        os.environ.setdefault(k, v)

    api_key = os.getenv('BINANCE_API_KEY', '').strip()
    api_secret = os.getenv('BINANCE_API_SECRET', '').strip()

    position_size = float(os.getenv('POSITION_SIZE', '0.01'))
    leverage = int(float(os.getenv('LEVERAGE', '10')))
    max_pos_per_side = int(os.getenv('MAX_POS_PER_SIDE', '3'))
    global_stop_frac = float(os.getenv('GLOBAL_STOP_FRACTION', '0.50'))
    daily_cap_frac = float(os.getenv('DAILY_LOSS_CAP_FRACTION', '0.08'))
    timeframe = os.getenv('TIMEFRAME', '2h')

    if not api_key or not api_secret:
        log("❌ Clés API manquantes. Définir BINANCE_API_KEY et BINANCE_API_SECRET (TESTNET)")
        sys.exit(1)

    # Load baseline params
    baseline = load_baseline(baseline_path)
    if not baseline:
        log("❌ Baseline introuvable ou vide. Placer outputs/BEST_BASELINE.json")
        sys.exit(1)
    symbols = list(baseline.keys())
    log(f"▶️ Lancement live TESTNET {symbols} timeframe={timeframe} pos={position_size*100:.1f}% levier={leverage} maxPos={max_pos_per_side}")

    # Exchanges: data (mainnet) + trade (testnet)
    ex_data = ccxt.binanceusdm({'enableRateLimit': True})
    ex_data.set_sandbox_mode(False)
    ex_data.load_markets()

    ex_trade = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    ex_trade.set_sandbox_mode(True)
    ex_trade.options = {**getattr(ex_trade, 'options', {}), 'defaultType': 'future'}
    markets = ex_trade.load_markets()

    # Wallet / equity (USDT)
    try:
        bal = ex_trade.fetch_balance()
        equity_usdt = float(bal.get('total', {}).get('USDT', 0.0))
    except Exception:
        equity_usdt = 0.0
    if equity_usdt <= 0:
        log("⚠️ Solde USDT nul sur TESTNET. Déposez des fonds de démo.")
    initial_equity_usdt = max(1.0, equity_usdt)
    global_stop_usdt = initial_equity_usdt * global_stop_frac

    # One-shot bar-close logic per symbol
    results = []
    for sym in symbols:
        try:
            df = fetch_df(ex_data, sym, timeframe=timeframe, limit=600)
            if df.empty:
                log(f"⚠️ Pas de données OHLCV pour {sym}")
                continue
            p = baseline.get(sym, {})
            tenkan = int(p.get('tenkan', 9))
            kijun = int(p.get('kijun', 26))
            senkou_b = int(p.get('senkou_b', 52))
            shift = int(p.get('shift', 26))
            atr_mult = float(p.get('atr_mult', 3.0))
            df_sig = calculate_ichimoku(df.copy(), tenkan, kijun, senkou_b, shift)
            last = df_sig.iloc[-1]
            close = float(last['close'])
            atr = float(last['ATR']) if 'ATR' in last and pd.notna(last['ATR']) else np.nan

            # Ensure symbol settings
            ensure_symbol_settings(ex_trade, sym, leverage)

            # Position sizing
            try:
                bal = ex_trade.fetch_balance()
                equity_usdt = float(bal.get('total', {}).get('USDT', equity_usdt))
            except Exception:
                pass
            if equity_usdt <= global_stop_usdt:
                log(f"🛑 STOP GLOBAL atteint: equity={equity_usdt:.2f}USDT <= {global_stop_usdt:.2f}USDT")
                break

            notional = equity_usdt * position_size
            qty = max(0.0, notional / max(close, 1e-12))
            # Round with market precision
            m = markets.get(sym) or {}
            amount_prec = int(m.get('precision', {}).get('amount', 6)) if m else 6
            qty = float(f"{qty:.{amount_prec}f}")
            if qty <= 0:
                log(f"⚠️ Quantité nulle pour {sym}")
                continue

            # Detect simple flat/position from positions endpoint (best-effort)
            has_long = False
            has_short = False
            try:
                poss = ex_trade.fetch_positions([sym])
                for pos in poss:
                    amt = float(pos.get('contracts') or pos.get('entryPrice', 0))
                    # ccxt unifies: pos['side'] may exist, else use amt sign
                    side = pos.get('side')
                    if side == 'long' and abs(amt) > 0:
                        has_long = True
                    if side == 'short' and abs(amt) > 0:
                        has_short = True
            except Exception:
                # best-effort; assume flat
                pass

            did = False
            if bool(last.get('signal_long', False)) and not has_long:
                try:
                    o = ex_trade.create_order(sym, 'MARKET', 'buy', qty)
                    log(f"📈 BUY {sym} qty={qty} filled")
                    if not np.isnan(atr):
                        # ATR-based trailing as percentage of price
                        cb_rate = (atr_mult * atr / max(close, 1e-12)) * 100.0
                        ok = place_trailing_stop(ex_trade, sym, 'buy', qty, cb_rate, activation_price=None)
                        if not ok:
                            stop_price = max(0.0, close - atr_mult * atr)
                            place_protective_stop(ex_trade, sym, 'buy', qty, stop_price)
                    did = True
                except Exception as e:
                    log(f"❌ BUY échoué {sym}: {e}")

            if bool(last.get('signal_short', False)) and not has_short:
                try:
                    o = ex_trade.create_order(sym, 'MARKET', 'sell', qty)
                    log(f"📉 SELL {sym} qty={qty} filled")
                    if not np.isnan(atr):
                        cb_rate = (atr_mult * atr / max(close, 1e-12)) * 100.0
                        ok = place_trailing_stop(ex_trade, sym, 'sell', qty, cb_rate, activation_price=None)
                        if not ok:
                            stop_price = float(close + atr_mult * atr)
                            place_protective_stop(ex_trade, sym, 'sell', qty, stop_price)
                    did = True
                except Exception as e:
                    log(f"❌ SELL échoué {sym}: {e}")

            results.append({"symbol": sym, "close": close, "qty": qty, "acted": did})
        except Exception as e:
            log(f"⚠️ Erreur {sym}: {e}")

    # Summary
    try:
        log("Résumé des actions:")
        for r in results:
            log(str(r))
    except Exception:
        pass


if __name__ == '__main__':
    main()


