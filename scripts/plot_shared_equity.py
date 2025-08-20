import json
import os
from datetime import datetime, timedelta

import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import sys

# Assurer l'import du module parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ichimoku_pipeline_web_v4_8_fixed import PROFILES, utc_ms, fetch_ohlcv_range, backtest_shared_portfolio


def main() -> int:
    profile = 'pipeline_web6'
    out_dir = 'outputs'

    # Charger meilleurs paramètres existants
    # On prend le plus récent best_params_per_symbol_* pour le profil
    best_files = sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith('best_params_per_symbol_') and profile in f and f.endswith('.json')
    ])
    if not best_files:
        print('Aucun best_params_per_symbol_*.json trouvé dans outputs/.')
        return 1
    params_path = best_files[-1]
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    cfg = PROFILES[profile]
    ex = ccxt.binance({'enableRateLimit': True})
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(365.25 * cfg['years_back']))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)

    market_data = {}
    for sym in cfg['symbols']:
        df = fetch_ohlcv_range(ex, sym, cfg['timeframe'], since_ms, until_ms, cache_dir='data', use_cache=True)
        if not df.empty:
            market_data[sym] = df

    result = backtest_shared_portfolio(market_data, params, timeframe=cfg['timeframe'], record_curve=True)

    curve = result.get('equity_curve', [])
    if not curve:
        print('Pas de courbe enregistrée.')
        return 1

    # Export CSV de la courbe
    ts_label = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'shared_equity_curve_{profile}_{ts_label}.csv')
    df_curve = pd.DataFrame(curve)
    # Convert to datetime and compute EUR equity
    df_curve['timestamp'] = pd.to_datetime(df_curve['timestamp'])
    df_curve['equity_eur'] = df_curve['equity_mult'].astype(float) * 1000.0
    df_curve.to_csv(csv_path, index=False)

    # Plot PNG
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_curve['timestamp'], df_curve['equity_eur'], color='#0D47A1', linewidth=1.6, label='Equity (€)')
    ax.set_title('Portefeuille partagé — Equity (EUR)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Solde (€)')
    # Format axes
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}€"))

    # Marqueur min equity
    min_eq = result.get('min_equity')
    min_ts = pd.to_datetime(result.get('min_equity_ts')) if result.get('min_equity_ts') else None
    if min_eq is not None and min_ts is not None:
        ax.axhline(min_eq * 1000.0, color='red', linestyle='--', alpha=0.6, label=f"Min = {min_eq*1000.0:,.0f}€")
        ax.axvline(min_ts, color='red', linestyle=':', alpha=0.6)

    # Plot trade markers if available
    trades = result.get('trades_detail', [])
    if trades:
        df_tr = pd.DataFrame(trades)
        if not df_tr.empty and 'timestamp' in df_tr.columns:
            df_tr['timestamp'] = pd.to_datetime(df_tr['timestamp'])
            if 'entry_ts' in df_tr.columns:
                df_tr['entry_ts'] = pd.to_datetime(df_tr['entry_ts'])
            df_tr['ret'] = pd.to_numeric(df_tr['ret'], errors='coerce')
            df_tr.sort_values('timestamp', inplace=True)
            df_curve_sorted = df_curve[['timestamp', 'equity_eur']].sort_values('timestamp')
            mapped = pd.merge_asof(df_tr, df_curve_sorted, on='timestamp', direction='backward')

            # Define subsets for markers
            long_win = mapped[(mapped['type'] == 'long') & (mapped['ret'] > 0)]
            long_loss = mapped[(mapped['type'] == 'long') & (mapped['ret'] <= 0)]
            short_win = mapped[(mapped['type'] == 'short') & (mapped['ret'] > 0)]
            short_loss = mapped[(mapped['type'] == 'short') & (mapped['ret'] <= 0)]

            ax.scatter(long_win['timestamp'], long_win['equity_eur'], marker='^', color='green', s=18, alpha=0.5, label='Long gain (exit)')
            ax.scatter(long_loss['timestamp'], long_loss['equity_eur'], marker='v', color='red', s=18, alpha=0.5, label='Long perte (exit)')
            ax.scatter(short_win['timestamp'], short_win['equity_eur'], marker='v', color='green', s=18, alpha=0.5, label='Short gain (exit)')
            ax.scatter(short_loss['timestamp'], short_loss['equity_eur'], marker='^', color='red', s=18, alpha=0.5, label='Short perte (exit)')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    png_path = os.path.join(out_dir, f'shared_equity_curve_{profile}_{ts_label}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(csv_path)
    print(png_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


