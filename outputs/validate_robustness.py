import os
import sys
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

# Ensure repo root on path so we can import main module from outputs/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ichimoku_pipeline_web_v4_8_fixed import (
    PROFILES,
    fetch_ohlcv_range,
    backtest_shared_portfolio,
    utc_ms,
)


def load_baseline(outputs_dir: str) -> dict:
    # Prefer top-decile minimal DD if available, else BEST_BASELINE
    cand = [
        os.path.join(outputs_dir, 'BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json'),
        os.path.join(outputs_dir, 'BEST_BASELINE.json'),
    ]
    for p in cand:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError('No baseline JSON found in outputs/')


def build_market(profile: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    cfg = PROFILES[profile]
    timeframe = cfg['timeframe']
    years_back = cfg['years_back']
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)
    ex = __import__('ccxt').binance({'enableRateLimit': True})
    out = {}
    for sym in cfg['symbols']:
        df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir='data', use_cache=use_cache)
        if not df.empty:
            out[sym] = df
    if not out:
        raise RuntimeError('No market data loaded')
    return out


def split_is_oos(df: pd.DataFrame, oos_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    n = len(df)
    cut = max(1, int(n * (1.0 - oos_ratio)))
    ts_cut = df.index[cut]
    return df.iloc[:cut], df.iloc[cut:]


def main():
    profile = os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    baseline = load_baseline(outputs_dir)
    market_full = build_market(profile, use_cache=True)
    timeframe = PROFILES[profile]['timeframe']

    market_is = {}
    market_oos = {}
    for sym, df in market_full.items():
        df_is, df_oos = split_is_oos(df, oos_ratio=0.2)
        if not df_is.empty:
            market_is[sym] = df_is
        if not df_oos.empty:
            market_oos[sym] = df_oos

    # Portfolio evaluation
    res_is = backtest_shared_portfolio(market_is, baseline, timeframe=timeframe, record_curve=False) if market_is else {}
    res_oos = backtest_shared_portfolio(market_oos, baseline, timeframe=timeframe, record_curve=False) if market_oos else {}

    # Save report
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    txt = os.path.join(outputs_dir, f'ROBUSTNESS_REPORT_{ts}.txt')
    lines = []
    lines.append(f'ROBUSTNESS REPORT — {profile} — {ts}')
    lines.append('Baseline source: ' + ('BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json' if os.path.exists(os.path.join(outputs_dir,'BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json')) else 'BEST_BASELINE.json'))
    def fmt_block(title: str, d: dict):
        if not isinstance(d, dict) or not d:
            return [f'{title}: (no data)']
        eq = d.get('equity_mult', float('nan')) * 1000.0 if 'equity_mult' in d else float('nan')
        dd = d.get('max_drawdown', float('nan'))
        if dd is not None:
            try:
                dd = float(dd) * 100.0 if float(dd) <= 5.0 else float(dd)
            except Exception:
                pass
        sh = d.get('sharpe_proxy', float('nan'))
        tr = d.get('trades', 0)
        return [
            f'{title}:',
            f'  Equity: {eq:,.0f} €',
            f'  Max DD: {dd:.2f} %',
            f'  Sharpe*: {sh:.2f}',
            f'  Trades: {tr}',
        ]
    lines += fmt_block('In-sample (80%)', res_is)
    lines += fmt_block('Out-of-sample (20%)', res_oos)
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).replace(',', ' '))
    print(txt)


if __name__ == '__main__':
    main()


