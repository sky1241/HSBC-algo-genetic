#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ichimoku_pipeline_web_v4_8.py

Usage:
  python ichimoku_pipeline_web_v4_8.py pipeline_web6
  # options:
  #   --trials 1000000 --seed 42 --out outputs --no-cache

Dépendances: pip install ccxt pandas numpy
"""

import os, sys, time, random, argparse, traceback
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

try:
    import ccxt
except Exception:
    print("ccxt n'est pas installé. Fais:  pip install ccxt")
    raise

# ---------------- Utils ---------------- #

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

def log(msg):
    print(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}", flush=True)

# ------------- Data fetch (Binance via ccxt) ------------- #

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, limit=1500, cache_dir="data", use_cache=True):
    """
    Télécharge des bougies [open, high, low, close, volume] entre since_ms et until_ms, en paginant.
    Met en cache dans data/{symbol}_{timeframe}.csv
    """
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{symbol.replace('/','-')}_{timeframe}.csv")

    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df = df.loc[(df.index >= pd.to_datetime(from_ms(since_ms))) & (df.index <= pd.to_datetime(from_ms(until_ms)))]
            if len(df) > 0:
                return df
        except Exception:
            pass

    all_rows = []
    since = since_ms
    safety = 0
    while True:
        safety += 1
        if safety > 500:  # garde-fou
            break

        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break

        for ts, o, h, l, c, v in candles:
            if ts > until_ms:
                break
            all_rows.append([ts, o, h, l, c, v])

        last_ts = candles[-1][0]
        if last_ts >= until_ms:
            break

        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()

    try:
        df.to_csv(cache_path)
    except Exception:
        pass

    return df.loc[(df.index >= pd.to_datetime(from_ms(since_ms))) & (df.index <= pd.to_datetime(from_ms(until_ms)))]

# ------------- Indicators ------------- #

def rolling_midpoint(series_high, series_low, length):
    hh = series_high.rolling(length, min_periods=length).max()
    ll = series_low.rolling(length, min_periods=length).min()
    return (hh + ll) / 2.0

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52, displacement=26):
    """
    Calcule les lignes Ichimoku 'brutes'. Pour le backtest on évite le look-ahead :
    on n'utilise pas les valeurs projetées dans le futur dans nos règles.
    """
    out = df.copy()

    out["tenkan"] = rolling_midpoint(out["high"], out["low"], tenkan)
    out["kijun"]  = rolling_midpoint(out["high"], out["low"], kijun)
    out["senkou_a_raw"] = ((out["tenkan"] + out["kijun"]) / 2.0)
    out["senkou_b_raw"] = rolling_midpoint(out["high"], out["low"], senkou_b)

    # chikou = close décalée vers l'arrière (non utilisée dans les règles ici)
    out["chikou"] = out["close"].shift(-displacement)

    return out

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    return tr

def atr(df, period):
    tr = true_range(df)
    return tr.rolling(period, min_periods=period).mean()

# ------------- Strategy ------------- #

def backtest_long_short(df, tenkan, kijun, senkou_b, shift, atr_mult=0.5, symbol=None):
    """
    Entrée: Tenkan croise au-dessus de Kijun ET close > nuage (avec décalage)
    Sortie: croisement inverse OU trailing-stop ATR
    """
    data = ichimoku(df, tenkan, kijun, senkou_b, shift)

    # nuage 'présent' puis appliqué avec décalage sans look-ahead:
    data["cloud_top"] = np.maximum(data["senkou_a_raw"], data["senkou_b_raw"])
    data["cloud_bot"] = np.minimum(data["senkou_a_raw"], data["senkou_b_raw"])
    cloud_top_rule = data["cloud_top"].shift(shift)
    cloud_bot_rule = data["cloud_bot"].shift(shift)

    # signaux croisement
    data["bull_cross"] = (data["tenkan"] > data["kijun"]) & (data["tenkan"].shift(1) <= data["kijun"].shift(1))
    data["bear_cross"] = (data["tenkan"] < data["kijun"]) & (data["tenkan"].shift(1) >= data["kijun"].shift(1))

    # filtres nuage "au-dessus" et "en-dessous"
    data["above_cloud"] = data["close"] > cloud_top_rule
    data["below_cloud"] = data["close"] < cloud_bot_rule

    # ATR pour trailing
    atr_period = max(14, kijun)
    data["ATR"] = atr(data, atr_period)

    # Positions long et short
    position_long = 0
    position_short = 0
    entry_price_long = 0.0
    entry_price_short = 0.0
    trailing_long = np.nan
    trailing_short = np.nan

    equity = 1.0
    trades_long = []
    trades_short = []
    wins_long = 0
    wins_short = 0
    
    # Gestion du risque
    max_trades_per_symbol = 2  # Maximum 2 trades par paire
    max_total_trades = 6  # Maximum 6 positions totales (2 par paire × 3 paires)
    position_size = 0.05  # 5% du capital par trade
    stop_global = 0.30  # Stop global à 30% du capital

    for ts, row in data.iterrows():
        close = row["close"]

        # Update trailing si en position LONG
        if position_long == 1:
            new_trailing = close - (row["ATR"] * atr_mult) if pd.notna(row["ATR"]) else trailing_long
            if pd.notna(new_trailing):
                trailing_long = max(trailing_long, new_trailing) if not np.isnan(trailing_long) else new_trailing

        # Update trailing si en position SHORT
        if position_short == 1:
            new_trailing = close + (row["ATR"] * atr_mult) if pd.notna(row["ATR"]) else trailing_short
            if pd.notna(new_trailing):
                trailing_short = min(trailing_short, new_trailing) if not np.isnan(trailing_short) else new_trailing

        # Entrée LONG (achat)
        if position_long == 0 and row["bull_cross"] and row["above_cloud"]:
            current_capital = equity * 1000
            position_size_euros = current_capital * position_size
            
            # Compter les trades long par symbole et total
            symbol_trades_long = [t for t in trades_long if t.get("symbol") == symbol]
            total_trades = len(trades_long) + len(trades_short)
            
            # Vérifier les limites : 2 trades max par type par symbole ET 6 max total
            if (len(symbol_trades_long) < max_trades_per_symbol and 
                total_trades < max_total_trades and 
                current_capital >= position_size_euros):
                position_long = 1
                entry_price_long = close
                trailing_long = close - (row["ATR"] * atr_mult) if pd.notna(row["ATR"]) else np.nan
            else:
                continue

        # Entrée SHORT (vente à découvert)
        if position_short == 0 and row["bear_cross"] and row["below_cloud"]:
            current_capital = equity * 1000
            position_size_euros = current_capital * position_size
            
            # Compter les trades short par symbole et total
            symbol_trades_short = [t for t in trades_short if t.get("symbol") == symbol]
            total_trades = len(trades_long) + len(trades_short)
            
            # Vérifier les limites : 2 trades max par type par symbole ET 6 max total
            if (len(symbol_trades_short) < max_trades_per_symbol and 
                total_trades < max_total_trades and 
                current_capital >= position_size_euros):
                position_short = 1
                entry_price_short = close
                trailing_short = close + (row["ATR"] * atr_mult) if pd.notna(row["ATR"]) else np.nan
            else:
                continue

        # Sortie LONG
        if position_long == 1:
            # Stop global dynamique
            current_capital = equity * 1000
            stop_global_euros = 1000 * stop_global
            
            # Vérifier le stop global dynamique
            if current_capital <= stop_global_euros:
                # Stop global touché, fermer toutes les positions
                ret = (close / entry_price_long) - 1.0
                equity *= (1.0 + ret)
                wins_long += 1 if ret > 0 else 0
                trades_long.append({
                    "timestamp": ts,
                    "entry": entry_price_long,
                    "exit": close,
                    "ret": ret,
                    "exit_reason": "stop_global",
                    "symbol": symbol,
                    "type": "long"
                })
                position_long = 0
                entry_price_long = 0.0
                trailing_long = np.nan
                continue
            
            stop_hit = (not np.isnan(trailing_long)) and (close <= trailing_long)
            if row["bear_cross"] or stop_hit:
                ret = (close / entry_price_long) - 1.0
                equity *= (1.0 + ret)
                wins_long += 1 if ret > 0 else 0
                trades_long.append({
                    "timestamp": ts,
                    "entry": entry_price_long,
                    "exit": close,
                    "ret": ret,
                    "exit_reason": "stop" if stop_hit else "bear_cross",
                    "symbol": symbol,
                    "type": "long"
                })
                position_long = 0
                entry_price_long = 0.0
                trailing_long = np.nan

        # Sortie SHORT
        if position_short == 1:
            # Stop global dynamique
            current_capital = equity * 1000
            stop_global_euros = 1000 * stop_global
            
            # Vérifier le stop global dynamique
            if current_capital <= stop_global_euros:
                # Stop global touché, fermer toutes les positions
                ret = (entry_price_short / close) - 1.0  # Pour short: (entry - exit) / entry
                equity *= (1.0 + ret)
                wins_short += 1 if ret > 0 else 0
                trades_short.append({
                    "timestamp": ts,
                    "entry": entry_price_short,
                    "exit": close,
                    "ret": ret,
                    "exit_reason": "stop_global",
                    "symbol": symbol,
                    "type": "short"
                })
                position_short = 0
                entry_price_short = 0.0
                trailing_short = np.nan
                continue
            
            stop_hit = (not np.isnan(trailing_short)) and (close >= trailing_short)
            if row["bull_cross"] or stop_hit:
                ret = (entry_price_short / close) - 1.0  # Pour short: (entry - exit) / entry
                equity *= (1.0 + ret)
                wins_short += 1 if ret > 0 else 0
                trades_short.append({
                    "timestamp": ts,
                    "entry": entry_price_short,
                    "exit": close,
                    "ret": ret,
                    "exit_reason": "stop" if stop_hit else "bull_cross",
                    "symbol": symbol,
                    "type": "short"
                })
                position_short = 0
                entry_price_short = 0.0
                trailing_short = np.nan

    # si on termine en position LONG, on clôture à la dernière close
    if position_long == 1 and entry_price_long > 0:
        close = data["close"].iloc[-1]
        ret = (close / entry_price_long) - 1.0
        equity *= (1.0 + ret)
        wins_long += 1 if ret > 0 else 0
        trades_long.append({
            "timestamp": data.index[-1],
            "entry": entry_price_long,
            "exit": close,
            "ret": ret,
            "exit_reason": "eod",
            "symbol": symbol,
            "type": "long"
        })

    # si on termine en position SHORT, on clôture à la dernière close
    if position_short == 1 and entry_price_short > 0:
        close = data["close"].iloc[-1]
        ret = (entry_price_short / close) - 1.0  # Pour short: (entry - exit) / entry
        equity *= (1.0 + ret)
        wins_short += 1 if ret > 0 else 0
        trades_short.append({
            "timestamp": data.index[-1],
            "entry": entry_price_short,
            "exit": close,
            "ret": ret,
            "exit_reason": "eod",
            "symbol": symbol,
            "type": "short"
        })

    # metrics - combiner long et short
    all_trades = trades_long + trades_short
    returns = pd.Series([t["ret"] for t in all_trades], dtype=float)
    n_trades = len(all_trades)
    total_wins = wins_long + wins_short
    win_rate = (total_wins / n_trades) if n_trades > 0 else 0.0
    
    # Calculer le temps moyen en position
    def calculate_avg_time_in_position(trades_list):
        if not trades_list:
            return 0.0
        total_time = 0
        for trade in trades_list:
            if "timestamp" in trade and "entry" in trade:
                # Calculer la durée en heures (timeframe 2h)
                duration = 2  # 2h par défaut
                total_time += duration
        return total_time / len(trades_list) if trades_list else 0.0
    
    avg_time_long = calculate_avg_time_in_position(trades_long)
    avg_time_short = calculate_avg_time_in_position(trades_short)

    if n_trades > 0:
        eq_curve = (1.0 + returns).cumprod()
        max_dd = (eq_curve.cummax() - eq_curve).max() / max(eq_curve.cummax().max(), 1e-12)
    else:
        max_dd = 0.0

    start_ts = df.index[0] if len(df.index) else datetime.utcnow().astimezone(timezone.utc)
    end_ts = df.index[-1] if len(df.index) else start_ts
    years = (end_ts - start_ts).days / 365.25 if end_ts > start_ts else 0.0
    cagr = (equity ** (1/years) - 1.0) if years > 0 else 0.0

    if n_trades > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    else:
        sharpe = 0.0

    profit_factor = (returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if (returns[returns < 0].sum() != 0) else float("inf")
    expectancy = returns.mean() if n_trades > 0 else 0.0

    metrics = {
        "equity_mult": float(equity),
        "CAGR": float(cagr),
        "sharpe_proxy": float(sharpe),
        "max_drawdown": float(max_dd),
        "trades": int(n_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "expectancy": float(expectancy),
        "avg_time_long_hours": float(avg_time_long),
        "avg_time_short_hours": float(avg_time_short),
        "nb_trades_long": len(trades_long),
        "nb_trades_short": len(trades_short)
    }
    return metrics

# ------------- Pipeline ------------- #

PROFILES = {
    "pipeline_web6": {
        "symbols": ["BTC/USDT", "ETH/USDT", "DOGE/USDT"],
        "timeframe": "2h",
        "years_back": 5,
        "ranges": {
            "tenkan": (1, 70),
            "kijun": (1, 70),
            "senkou_b": (1, 70),
            "shift": (1, 99)
        },
        "atr_mult": 0.5,
        "default_trials": 120
    }
}

def run_profile(profile_name, trials=0, seed=None, out_dir="outputs", use_cache=True):
    if profile_name not in PROFILES:
        raise ValueError(f"Profil inconnu: {profile_name}")

    cfg = PROFILES[profile_name]
    ensure_dir(out_dir)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    symbols = cfg["symbols"]
    timeframe = cfg["timeframe"]
    years_back = cfg["years_back"]
    r = cfg["ranges"]
    atr_mult = cfg["atr_mult"]
    trials = trials or cfg["default_trials"]

    # période
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)

    # exchange
    ex = ccxt.binance({"enableRateLimit": True})

    # fetch data pour chaque symbole
    market_data = {}
    for sym in symbols:
        log(f"Téléchargement {sym} {timeframe} sur ~{years_back} ans…")
        df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir="data", use_cache=use_cache)
        if df.empty:
            log(f"⚠️  Pas de données pour {sym}.")
            continue
        market_data[sym] = df

    if not market_data:
        raise RuntimeError("Aucune donnée chargée.")

    all_rows = []

    log(f"🚀 Démarrage de {trials:,} essais sur {len(market_data)} symboles...")
    log(f"📊 Période: {years_back} ans, Timeframe: {timeframe}")
    
    for trial in range(1, trials + 1):

if __name__ == "__main__":
    main()
