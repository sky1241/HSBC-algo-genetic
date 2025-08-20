import os
import sys
import math
import json
import random
from datetime import datetime, timezone, timedelta

import numpy as np

# Ensure repo root in path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ichimoku_pipeline_web_v4_8_fixed import (
    PROFILES,
    fetch_ohlcv_range,
    backtest_shared_portfolio,
    utc_ms,
)


def load_baseline(outputs_dir: str) -> dict:
    for name in [
        'BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json',
        'BEST_BASELINE.json',
        'BEST_PER_SYMBOL.json',
    ]:
        p = os.path.join(outputs_dir, name)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError('No baseline found in outputs/')


def build_market(profile: str, use_cache: bool = True) -> tuple[str, dict]:
    cfg = PROFILES[profile]
    timeframe = cfg['timeframe']
    years_back = cfg['years_back']
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    start_dt = end_dt - timedelta(days=int(365.25 * years_back))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)
    ex = __import__('ccxt').binance({'enableRateLimit': True})
    market = {}
    for sym in cfg['symbols']:
        df = fetch_ohlcv_range(ex, sym, timeframe, since_ms, until_ms, cache_dir='data', use_cache=use_cache)
        if not df.empty:
            market[sym] = df
    if not market:
        raise RuntimeError('No data loaded')
    return timeframe, market


def equity_curve_from_shared(shared: dict) -> list[float]:
    curv = shared.get('equity_curve')
    if isinstance(curv, list) and len(curv) > 1:
        # entries can be:
        #  - list/tuple like [ts, equity]
        #  - dict like {"timestamp": ..., "equity_mult": ...}
        #  - plain numeric values (fallback)
        eqs = []
        for p in curv:
            try:
                if isinstance(p, dict):
                    # Prefer explicit equity_mult, else common aliases
                    val = p.get('equity_mult')
                    if val is None:
                        val = p.get('equity') if 'equity' in p else p.get('eq')
                    if val is not None:
                        eqs.append(float(val))
                        continue
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    eqs.append(float(p[1]))
                    continue
                # Last resort: try to cast the item itself
                eqs.append(float(p))
            except Exception:
                # ignore badly formatted entries
                pass
        if len(eqs) > 1:
            return eqs
    # Fallback: no curve -> return just start/end
    em = shared.get('equity_mult', 1.0)
    try:
        em = float(em)
    except Exception:
        em = 1.0
    return [1.0, em]


def max_drawdown(eqs: list[float]) -> float:
    peak = -1e9
    mdd = 0.0
    for x in eqs:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd  # in fraction (0..1)


def block_bootstrap(returns: np.ndarray, block_size: int, length: int) -> np.ndarray:
    # Concatenate random blocks until reaching length
    out = []
    n = len(returns)
    if n == 0:
        return np.array([])
    while len(out) < length:
        i = random.randint(0, max(0, n - block_size))
        out.extend(returns[i:i + block_size])
    return np.array(out[:length])


def sharpe_proxy(log_rets: np.ndarray) -> float:
    # Simple proxy: mean/std * sqrt(252) assuming daily-ish scaling
    if log_rets.size == 0:
        return float('nan')
    mu = log_rets.mean()
    sd = log_rets.std(ddof=1) if log_rets.size > 1 else 0.0
    if sd <= 0:
        return float('nan')
    return (mu / sd) * math.sqrt(252)


def main():
    random.seed(42)
    np.random.seed(42)
    profile = os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    outputs_dir = HERE
    # Use env overrides for POS/LEV; default are taken in backtester
    baseline = load_baseline(outputs_dir)
    timeframe, market = build_market(profile, use_cache=True)
    shared = backtest_shared_portfolio(market, baseline, timeframe=timeframe, record_curve=True)
    eq = equity_curve_from_shared(shared)
    # Returns
    eq = np.array(eq, dtype=float)
    eq = eq[eq > 0]
    if eq.size < 3:
        print('Not enough points for MC')
        return
    rets = eq[1:] / eq[:-1]
    log_rets = np.log(rets)
    L = log_rets.size
    B = max(10, min(50, L // 20))  # block size heuristic

    N = int(os.environ.get('MC_TRIALS', '1000'))
    finals = []
    dds = []
    sharps = []
    for _ in range(N):
        lr = block_bootstrap(log_rets, block_size=B, length=L)
        path = np.exp(lr).cumprod()
        path = np.insert(path, 0, 1.0)
        finals.append(path[-1])
        dds.append(max_drawdown(path))
        sharps.append(sharpe_proxy(lr))

    finals = np.array(finals)
    dds = np.array(dds)
    sharps = np.array(sharps)

    def pct(a, q):
        return float(np.nanpercentile(a, q))

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(outputs_dir, f'MC_REPORT_{ts}.txt')
    lines = []
    lines.append(f'MONTE CARLO REPORT — {profile} — {ts}')
    lines.append(f'Params: POSITION_SIZE={os.environ.get("POSITION_SIZE","default")}, LEVERAGE={os.environ.get("LEVERAGE","default")}')
    lines.append(f'N={N}, block_size={B}, steps={L}')
    lines.append('Final equity (×):')
    lines.append(f'  p5={pct(finals,5):.2f}, p50={pct(finals,50):.2f}, p95={pct(finals,95):.2f}, prob_loss={(finals<1.0).mean():.2%}')
    lines.append('Max DD (classic % from peak):')
    lines.append(f'  p5={pct(dds*100,5):.1f}%, p50={pct(dds*100,50):.1f}%, p95={pct(dds*100,95):.1f}%')
    lines.append('Sharpe proxy:')
    lines.append(f'  p5={pct(sharps,5):.2f}, p50={pct(sharps,50):.2f}, p95={pct(sharps,95):.2f}')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(out)


if __name__ == '__main__':
    main()


