import glob
import os
import re
import sys

import pandas as pd


def main() -> int:
    runs_files = sorted(glob.glob(os.path.join('outputs', 'runs_pipeline_web6_*.csv')))
    if not runs_files:
        print('No runs files found in outputs/.', file=sys.stderr)
        return 1

    src = runs_files[-1]
    df = pd.read_csv(src)

    metrics = [
        'equity_mult', 'CAGR', 'sharpe_proxy', 'max_drawdown', 'profit_factor', 'expectancy',
        'calmar_ratio', 'sortino_ratio', 'var_95', 'recovery_factor', 'win_rate',
    ]

    cols = [
        c for c in [
            'symbol', 'equity_mult', 'CAGR', 'sharpe_proxy', 'max_drawdown', 'trades', 'win_rate',
            'profit_factor', 'expectancy', 'calmar_ratio', 'sortino_ratio', 'var_95', 'recovery_factor',
            'recovery_days', 'trial', 'generation', 'tenkan', 'kijun', 'senkou_b', 'shift', 'atr_mult',
        ] if c in df.columns
    ]

    m = re.search(r'(\d{8}_\d{6})', src)
    ts = m.group(1) if m else 'latest'

    os.makedirs('outputs', exist_ok=True)
    out_paths: list[str] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        ascending = metric in ('max_drawdown', 'var_95')
        top = df.sort_values(metric, ascending=ascending).head(10)[cols]
        out_path = os.path.join('outputs', f'top10_by_{metric}_pipeline_web6_{ts}.csv')
        top.to_csv(out_path, index=False)
        out_paths.append(out_path)

    print('\n'.join(out_paths))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



