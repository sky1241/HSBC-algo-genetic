import os
import time
from datetime import timedelta
import pandas as pd

def fetch_ohlcv_range(exchange, symbol, timeframe, since_ms, until_ms, cache_dir="data", use_cache=True, logger=print):
    """Fetch OHLCV data with optional caching."""
    cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            start_date = pd.to_datetime(since_ms, unit='ms')
            end_date = pd.to_datetime(until_ms, unit='ms')
            required_days = (end_date - start_date).days
            available_days = (df.index.max() - df.index.min()).days
            if len(df) > 0 and available_days >= required_days * 0.8:
                margin_days = max(30, required_days * 0.1)
                start_with_margin = start_date - timedelta(days=margin_days)
                end_with_margin = end_date + timedelta(days=margin_days)
                df_filtered = df[(df.index >= start_with_margin) & (df.index <= end_with_margin)]
                if len(df_filtered) > 0:
                    logger(f"📁 Données chargées depuis le cache: {len(df_filtered)} bougies (période demandée + marge)")
                    return df[(df.index >= start_date) & (df.index <= end_date)]
                else:
                    logger("⚠️ Cache existant mais données insuffisantes pour la période demandée")
            else:
                logger(f"⚠️ Cache existant mais période insuffisante (disponible: {available_days}j, requis: {required_days}j)")
        except Exception as e:
            logger(f"⚠️ Erreur lors du chargement du cache: {e}")
    all_data = []
    current_ms = since_ms
    while current_ms < until_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_ms, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current_ms = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            logger(f"⚠️  Erreur lors du téléchargement: {e}")
            break
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df_all = df.copy()
    df = df[(df.index >= pd.to_datetime(since_ms, unit='ms')) &
            (df.index <= pd.to_datetime(until_ms, unit='ms'))]
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df_all.to_csv(cache_file)
        logger(f"💾 Données sauvegardées en cache: {len(df_all)} bougies (total téléchargé)")
    return df

def utc_ms(dt):
    """Convert datetime to UTC milliseconds."""
    return int(dt.timestamp() * 1000)

def make_annual_folds(df: pd.DataFrame, start_year: int | None = None, end_year: int | None = None):
    """Return list of (YYYY-MM-DD, YYYY-MM-DD) per calendar year present in df.index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    years = pd.Index(sorted(df.index.year.unique()))
    if start_year is not None:
        years = years[years >= start_year]
    if end_year is not None:
        years = years[years <= end_year]
    folds = []
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y}-12-31")
        if ((df.index >= start) & (df.index <= end)).any():
            folds.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if not folds:
        raise ValueError("Aucun fold annuel trouvé pour la plage demandée")
    return folds
