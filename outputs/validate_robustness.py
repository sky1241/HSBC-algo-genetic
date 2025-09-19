import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure repo root on path so we can import main module from outputs/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ichimoku_pipeline_web_v4_8_fixed import (
    PROFILES,
    backtest_shared_portfolio,
)

from src.io_loader import (
    align_funding_to_ohlcv,
    load_funding,
    load_ohlcv,
)

DATA_ROOT = Path(REPO_ROOT) / "data"


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


def _ohlcv_candidates(symbol: str, timeframe: str) -> list[list[Path]]:
    symbol_key = symbol.replace('/', '_').upper()
    tf_key = timeframe.lower()
    candidates: list[list[Path]] = []
    if symbol_key == "BTC_USDT" and tf_key == "2h":
        bitstamp = DATA_ROOT / "BTC_USD_2h.csv"
        binance = DATA_ROOT / "BTC_USDT_2h.csv"
        fused_clean = DATA_ROOT / "BTC_FUSED_2h_clean.csv"
        fused = DATA_ROOT / "BTC_FUSED_2h.csv"
        if bitstamp.exists() and binance.exists():
            candidates.append([bitstamp, binance])
        if fused_clean.exists():
            candidates.append([fused_clean])
        if fused.exists():
            candidates.append([fused])
    else:
        direct = DATA_ROOT / f"{symbol_key}_{tf_key}.csv"
        if direct.exists():
            candidates.append([direct])
    if not candidates:
        raise FileNotFoundError(f"aucun CSV OHLCV trouvé pour {symbol} {timeframe}")
    return candidates


def _resolve_funding_paths(symbol: str) -> list[Path]:
    base = symbol.replace('/', '').upper()
    for name in (
        f"{base}_funding_8h.csv",
        f"{base}_funding.csv",
        f"{base}_funding_rate_8h.csv",
    ):
        path = DATA_ROOT / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"aucun CSV de funding trouvé pour {symbol}")


def build_market(profile: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    del use_cache  # legacy parameter kept for backward compatibility
    cfg = PROFILES[profile]
    timeframe = cfg['timeframe']
    market: dict[str, pd.DataFrame] = {}
    for sym in cfg['symbols']:
        last_error: Exception | None = None
        df_prices: pd.DataFrame | None = None
        for bundle in _ohlcv_candidates(sym, timeframe):
            try:
                df_prices = load_ohlcv(bundle, tz="UTC")
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - fallback when a bundle is invalid
                last_error = exc
                continue
        if df_prices is None:
            raise RuntimeError(f"échec du chargement OHLCV pour {sym}: {last_error}")
        funding_paths = _resolve_funding_paths(sym)
        df_funding = load_funding(funding_paths)
        aligned = align_funding_to_ohlcv(df_prices, df_funding, freq=timeframe.upper())
        close_match = aligned['close']
        if not close_match.equals(df_prices['close']):
            raise ValueError(f"close désaligné après fusion pour {sym}")
        df_combined = df_prices.copy()
        df_combined['funding'] = aligned['funding']
        market[sym] = df_combined
    if not market:
        raise RuntimeError('No market data loaded')
    return market


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


