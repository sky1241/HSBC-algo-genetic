import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


def find_shared_jsons() -> List[str]:
    paths = []
    # Current outputs
    paths += glob.glob(os.path.join('outputs', 'shared_portfolio_*.json'))
    # Archives, recursively
    paths += glob.glob(os.path.join('outputs', '**', 'shared_portfolio_*.json'), recursive=True)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def load_metrics(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {}
    metrics = {}
    if isinstance(data, dict):
        if 'shared_metrics' in data and isinstance(data['shared_metrics'], dict):
            metrics = data['shared_metrics']
        else:
            for k in ['equity_mult', 'CAGR', 'sharpe_proxy', 'max_drawdown', 'trades', 'min_equity']:
                metrics[k] = data.get(k)
    return metrics


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def main() -> int:
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    files = find_shared_jsons()
    if not files:
        print('Aucun shared_portfolio_*.json trouvé (outputs + archives).')
        return 1

    rows = []
    for p in files:
        m = load_metrics(p)
        if not m:
            continue
        row = {
            'file': os.path.basename(p),
            'path': p,
            'equity_mult': float(m.get('equity_mult')) if m.get('equity_mult') is not None else None,
            'CAGR': float(m.get('CAGR')) if m.get('CAGR') is not None else None,
            'sharpe_proxy': float(m.get('sharpe_proxy')) if m.get('sharpe_proxy') is not None else None,
            'max_drawdown': float(m.get('max_drawdown')) if m.get('max_drawdown') is not None else None,
            'trades': int(m.get('trades')) if m.get('trades') is not None else None,
            'min_equity': float(m.get('min_equity')) if m.get('min_equity') is not None else None,
        }
        rows.append(row)

    if not rows:
        print('Aucune métrique valide.')
        return 1

    df = pd.DataFrame(rows)
    # Filtre de base: on évite les cas dégénérés; on veut de la perf positive et DD défini
    df = df[df['max_drawdown'].notna()]
    df = df[df['equity_mult'].notna()]
    # Optionnel: seuil de perf pour éviter les très faibles perfs
    df_safe = df[df['equity_mult'] >= 3.0].copy()
    if df_safe.empty:
        df_safe = df.copy()

    # RANG 1: minimiser drawdown, tie-breaker equity puis Sharpe (5 meilleurs)
    top5_low_dd = df_safe.sort_values(['max_drawdown', 'equity_mult', 'sharpe_proxy'], ascending=[True, False, False]).head(5)

    # RANG 2: score composite axé sécurité
    # Normalisation: low_dd_score = 1 - norm(max_drawdown) ; perf_score = norm(equity_mult); sharpe_score = norm(sharpe)
    dd_norm = normalize_series(df_safe['max_drawdown'])
    eq_norm = normalize_series(df_safe['equity_mult'])
    sh_norm = normalize_series(df_safe['sharpe_proxy'].fillna(df_safe['sharpe_proxy'].median() or 0.0))
    composite = 0.6 * (1.0 - dd_norm) + 0.3 * eq_norm + 0.1 * sh_norm
    df_safe = df_safe.assign(safe_score=composite)
    top5_composite = df_safe.sort_values(['safe_score', 'equity_mult'], ascending=[False, False]).head(5)

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path_lowdd = os.path.join(out_dir, f'shared_top5_low_drawdown_{ts}.csv')
    path_comp = os.path.join(out_dir, f'shared_top5_composite_safe_{ts}.csv')
    top5_low_dd.to_csv(path_lowdd, index=False)
    top5_composite.to_csv(path_comp, index=False)
    print(path_lowdd)
    print(path_comp)

    # HTML rapide lisible
    def card(r: pd.Series) -> str:
        em = float(r.get('equity_mult') or 0.0)
        md = float(r.get('max_drawdown') or 0.0)
        me = float(r.get('min_equity') or 0.0)
        eq_eur = em * 1000.0
        min_eur = me * 1000.0
        return f"""
        <div class=card>
          <div class=title>{r.get('file','')}</div>
          <div class=row><b>💶 Final:</b> {eq_eur:,.0f} €</div>
          <div class=row><b>📉 Max DD:</b> {md*100:.1f}%</div>
          <div class=row><b>🔻 Pire solde:</b> {min_eur:,.0f} €</div>
          <div class=row><b>🔁 Trades:</b> {int(r.get('trades') or 0)}</div>
          <div class=row><b>📈 Sharpe*:</b> {float(r.get('sharpe_proxy') or 0.0):.2f}</div>
        </div>
        """

    html = f"""<!doctype html>
    <html><head><meta charset=utf-8><title>Top 5 Sécurité</title>
    <style>
      body{{font-family:Arial,sans-serif;background:#fafafa;color:#222;margin:0;padding:24px;}}
      h1{{margin:0 0 12px}} h2{{margin:16px 0 8px;color:#0D47A1}}
      .grid{{display:grid;grid-template-columns:repeat(1,1fr);gap:12px;}}
      @media(min-width:900px){{.grid{{grid-template-columns:repeat(2,1fr);}}}}
      .card{{background:#fff;border-radius:12px;padding:12px;box-shadow:0 2px 6px rgba(0,0,0,.08)}}
      .title{{font-weight:bold;margin-bottom:8px;color:#555}}
      .row{{margin:4px 0}}
      footer{{margin-top:16px;color:#777;font-size:12px}}
    </style></head><body>
    <h1>Top 5 — Drawdown minimal</h1>
    <div class=grid>
      {''.join(card(r) for _, r in top5_low_dd.iterrows())}
    </div>
    <h2>Top 5 — Score sécurité (DD bas + perf)</h2>
    <div class=grid>
      {''.join(card(r) for _, r in top5_composite.iterrows())}
    </div>
    <footer>* Sharpe approximé. Généré {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</footer>
    </body></html>"""

    html_path = os.path.join(out_dir, f'KIDS_REPORT_SAFE.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(html_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


