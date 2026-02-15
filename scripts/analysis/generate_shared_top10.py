import os
import json
from datetime import datetime
import pandas as pd


def load_shared_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize schema: support both {shared_metrics: {...}} and flat metrics
    metrics = {}
    if isinstance(data, dict):
        if 'shared_metrics' in data and isinstance(data['shared_metrics'], dict):
            metrics = data['shared_metrics']
        else:
            # Try flat keys
            keys = ['equity_mult', 'CAGR', 'sharpe_proxy', 'max_drawdown', 'trades', 'min_equity']
            metrics = {k: data.get(k, None) for k in keys}
    return metrics


def main() -> int:
    out_dir = 'outputs'
    files = [
        os.path.join(out_dir, f) for f in os.listdir(out_dir)
        if f.startswith('shared_portfolio_') and f.endswith('.json')
    ]
    if not files:
        print('Aucun shared_portfolio_*.json trouvé dans outputs/.')
        return 1

    rows = []
    for p in sorted(files):
        m = load_shared_json(p)
        if not m:
            continue
        row = {
            'file': os.path.basename(p),
            'equity_mult': float(m.get('equity_mult')) if m.get('equity_mult') is not None else None,
            'CAGR': float(m.get('CAGR')) if m.get('CAGR') is not None else None,
            'sharpe_proxy': float(m.get('sharpe_proxy')) if m.get('sharpe_proxy') is not None else None,
            'max_drawdown': float(m.get('max_drawdown')) if m.get('max_drawdown') is not None else None,
            'trades': int(m.get('trades')) if m.get('trades') is not None else None,
            'min_equity': float(m.get('min_equity')) if m.get('min_equity') is not None else None,
        }
        # equity in EUR for convenience (assuming 1000€ initial)
        if row['equity_mult'] is not None:
            row['equity_eur'] = row['equity_mult'] * 1000.0
        if row['min_equity'] is not None:
            row['min_equity_eur'] = row['min_equity'] * 1000.0
        rows.append(row)

    if not rows:
        print('Aucune métrique valide trouvée dans les JSONs.')
        return 1

    df = pd.DataFrame(rows)
    ts_label = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(out_dir, f'shared_portfolio_summary_{ts_label}.csv')
    df.to_csv(summary_path, index=False)
    print(summary_path)

    # Generate top-10 files by key metrics
    metrics_to_sort = [
        ('equity_mult', False),
        ('CAGR', False),
        ('sharpe_proxy', False),
        ('max_drawdown', True),  # lower is better
        ('min_equity', False),
    ]
    for col, ascending in metrics_to_sort:
        if col in df.columns and df[col].notna().any():
            top = df.sort_values(col, ascending=ascending).head(10)
            top_path = os.path.join(out_dir, f'shared_top10_by_{col}_{ts_label}.csv')
            top.to_csv(top_path, index=False)
            print(top_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


