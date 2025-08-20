import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def build_market(profile: str, use_cache: bool = True):
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
        eqs = []
        for p in curv:
            try:
                if isinstance(p, dict):
                    val = p.get('equity_mult')
                    if val is None:
                        val = p.get('equity') if 'equity' in p else p.get('eq')
                    if val is not None:
                        eqs.append(float(val))
                        continue
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    eqs.append(float(p[1]))
                    continue
                eqs.append(float(p))
            except Exception:
                pass
        if len(eqs) > 1:
            return eqs
    em = shared.get('equity_mult', 1.0)
    try:
        em = float(em)
    except Exception:
        em = 1.0
    return [1.0, em]


def max_drawdown(eqs: np.ndarray) -> float:
    peak = -1e9
    mdd = 0.0
    for x in eqs:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def block_bootstrap(returns: np.ndarray, block_size: int, length: int) -> np.ndarray:
    out = []
    n = len(returns)
    if n == 0:
        return np.array([])
    while len(out) < length:
        i = np.random.randint(0, max(1, n - block_size + 1))
        out.extend(returns[i:i + block_size])
    return np.array(out[:length])


def sharpe_proxy(log_rets: np.ndarray) -> float:
    if log_rets.size == 0:
        return float('nan')
    mu = log_rets.mean()
    sd = log_rets.std(ddof=1) if log_rets.size > 1 else 0.0
    if sd <= 0:
        return float('nan')
    return (mu / sd) * math.sqrt(252)


def run_mc_for_baseline(profile: str, market: dict, timeframe: str, baseline_path: str, trials: int = 1000):
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    shared = backtest_shared_portfolio(market, baseline, timeframe=timeframe, record_curve=True)
    eq = np.array(equity_curve_from_shared(shared), dtype=float)
    eq = eq[eq > 0]
    if eq.size < 3:
        return None
    rets = eq[1:] / eq[:-1]
    log_rets = np.log(rets)
    L = log_rets.size
    B = max(10, min(50, L // 20))
    finals = []
    dds = []
    sharps = []
    for _ in range(trials):
        lr = block_bootstrap(log_rets, block_size=B, length=L)
        path = np.exp(lr).cumprod()
        path = np.insert(path, 0, 1.0)
        finals.append(path[-1])
        dds.append(max_drawdown(path))
        sharps.append(sharpe_proxy(lr))
    return {
        'finals': np.array(finals, dtype=float),
        'dds': np.array(dds, dtype=float),
        'sharps': np.array(sharps, dtype=float),
    }


def summarize(arr: np.ndarray) -> dict:
    def pct(a, q):
        return float(np.nanpercentile(a, q))
    return {
        'p5': pct(arr, 5),
        'p50': pct(arr, 50),
        'p95': pct(arr, 95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='append', required=True, help='Path to baseline JSON (repeat for two baselines)')
    parser.add_argument('--label', action='append', help='Label for each baseline (repeat twice)')
    parser.add_argument('--trials', type=int, default=int(os.environ.get('MC_TRIALS', '1000')))
    parser.add_argument('--profile', default=os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6'))
    args = parser.parse_args()

    if len(args.baseline) != 2:
        print('Need exactly two --baseline paths')
        sys.exit(1)
    labels = args.label or []
    while len(labels) < 2:
        labels.append(os.path.splitext(os.path.basename(args.baseline[len(labels)]))[0])

    timeframe, market = build_market(args.profile, use_cache=True)

    np.random.seed(42)

    res = []
    for bl in args.baseline:
        res.append(run_mc_for_baseline(args.profile, market, timeframe, bl, trials=args.trials))

    if res[0] is None or res[1] is None:
        print('Not enough points for MC for one of the baselines')
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_png = os.path.join(HERE, f'MC_DIST_COMPARE_{ts}.png')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Finals (log scale on x for visibility)
    ax = axes[0]
    for i, lab in enumerate(labels):
        finals = res[i]['finals']
        ax.hist(finals, bins=60, alpha=0.45, density=True, label=f'{lab} (med {np.median(finals):.2f}×)')
    ax.set_title('Distribution — Final equity × (MC)')
    ax.set_xlabel('Final equity ×')
    ax.set_ylabel('Density')
    ax.set_xscale('log')
    ax.legend()

    # Max DD %
    ax = axes[1]
    for i, lab in enumerate(labels):
        ddp = res[i]['dds'] * 100.0
        ax.hist(ddp, bins=50, alpha=0.45, density=True, label=f'{lab} (med {np.median(ddp):.1f}%)')
    ax.set_title('Distribution — Max drawdown % (MC)')
    ax.set_xlabel('Max DD %')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=140)

    # Print summaries
    f1 = summarize(res[0]['finals'])
    f2 = summarize(res[1]['finals'])
    d1 = summarize(res[0]['dds'] * 100.0)
    d2 = summarize(res[1]['dds'] * 100.0)
    s1 = summarize(res[0]['sharps'])
    s2 = summarize(res[1]['sharps'])

    print(out_png)
    print(json.dumps({'labels': labels, 'finals': [f1, f2], 'dd_pct': [d1, d2], 'sharpe_proxy': [s1, s2]}, ensure_ascii=False))


if __name__ == '__main__':
    main()


