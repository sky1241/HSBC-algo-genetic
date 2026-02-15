import os
import io
import base64
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import ccxt

# Ensure parent path import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ichimoku_pipeline_web_v4_8_fixed import PROFILES, utc_ms, fetch_ohlcv_range, backtest_shared_portfolio


def find_shared_jsons() -> List[str]:
    paths = []
    paths += glob.glob(os.path.join('outputs', 'shared_portfolio_*.json'))
    paths += glob.glob(os.path.join('outputs', '**', 'shared_portfolio_*.json'), recursive=True)
    return sorted(set(paths))


def load_json(path: str) -> Dict[str, Any]:
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entry(path: str) -> Dict[str, Any]:
    data = load_json(path)
    metrics = data.get('shared_metrics', {}) if 'shared_metrics' in data else data
    params = data.get('best_params', {})
    return {
        'path': path,
        'file': os.path.basename(path),
        'equity_mult': float(metrics.get('equity_mult')) if metrics.get('equity_mult') is not None else None,
        'CAGR': float(metrics.get('CAGR')) if metrics.get('CAGR') is not None else None,
        'sharpe_proxy': float(metrics.get('sharpe_proxy')) if metrics.get('sharpe_proxy') is not None else None,
        'max_drawdown': float(metrics.get('max_drawdown')) if metrics.get('max_drawdown') is not None else None,
        'trades': int(metrics.get('trades')) if metrics.get('trades') is not None else None,
        'min_equity': float(metrics.get('min_equity')) if metrics.get('min_equity') is not None else None,
        'params': params,
    }


def choose_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 10 best by equity
    top10_equity = df.sort_values('equity_mult', ascending=False).head(10)

    # 5 with minimal drawdown, tie-breakers by equity and Sharpe
    candidates_dd = df.dropna(subset=['max_drawdown', 'equity_mult']).copy()
    top5_low_dd = candidates_dd.sort_values(['max_drawdown', 'equity_mult', 'sharpe_proxy'],
                                            ascending=[True, False, False]).head(5)

    # 5 personal picks: high Sharpe, DD <= 0.35, equity >= 5
    personal_pool = df.copy()
    personal_pool = personal_pool[personal_pool['equity_mult'].fillna(0) >= 5]
    personal_pool = personal_pool[personal_pool['max_drawdown'].fillna(1.0) <= 0.35]
    top5_personal = personal_pool.sort_values(['sharpe_proxy', 'equity_mult'], ascending=[False, False]).head(5)

    # Deduplicate keeping order: equity first, then add unique from dd and personal
    def dedup_concat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        seen = set()
        rows = []
        for part in parts:
            for _, r in part.iterrows():
                key = r['path']
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
        return pd.DataFrame(rows)

    # We return the individual sections (not the combined), but ensure internally they are unique
    top10_equity = dedup_concat([top10_equity])
    top5_low_dd = dedup_concat([top5_low_dd])
    top5_personal = dedup_concat([top5_personal])
    return top10_equity, top5_low_dd, top5_personal


def fetch_market_data(profile: str) -> Dict[str, pd.DataFrame]:
    cfg = PROFILES[profile]
    ex = ccxt.binance({'enableRateLimit': True})
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=int(365.25 * cfg['years_back']))
    since_ms = utc_ms(start_dt)
    until_ms = utc_ms(end_dt)
    market_data: Dict[str, pd.DataFrame] = {}
    for sym in cfg['symbols']:
        df = fetch_ohlcv_range(ex, sym, cfg['timeframe'], since_ms, until_ms, cache_dir='data', use_cache=True)
        if not df.empty:
            market_data[sym] = df
    return market_data


def small_curve_png(curve: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(curve)
    if df.empty or 'timestamp' not in df.columns:
        return ''
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['equity_eur'] = df['equity_mult'].astype(float) * 1000.0
    fig, ax = plt.subplots(figsize=(3.6, 1.1), dpi=150)
    ax.plot(df['timestamp'], df['equity_eur'], color='#0D47A1', linewidth=1.2)
    # light ticks: show years and compact euros
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1000:.0f}k"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=6, length=0)
    plt.tight_layout(pad=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def render_params(params: Dict[str, Any]) -> str:
    # Expect dict per symbol: {'BTC/USDT': {...}, 'ETH/USDT': {...}, ...}
    lines = []
    for sym in sorted(params.keys()):
        p = params[sym]
        tenkan = p.get('tenkan'); kijun = p.get('kijun'); sen = p.get('senkou_b'); sh = p.get('shift'); atr = p.get('atr_mult')
        atr_txt = f"{float(atr):.2f}" if atr is not None else "?"
        lines.append(f"<div class=param><b>{sym}</b>: Tenkan {tenkan}, Kijun {kijun}, SenkouB {sen}, Shift {sh}, ATR× {atr_txt}</div>")
    return '\n'.join(lines)


def build_section_html(title: str, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame], profile: str) -> str:
    cards = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        params = r.get('params') or {}
        # Recompute curve for visual
        try:
            bt_res = backtest_shared_portfolio(market_data, params, timeframe=PROFILES[profile]['timeframe'], record_curve=True)
            curve = bt_res.get('equity_curve', [])
        except Exception:
            curve = []
        img_uri = small_curve_png(curve) if curve else ''
        em = float(r.get('equity_mult') or 0.0)
        md = float(r.get('max_drawdown') or 0.0)
        me = float(r.get('min_equity') or 0.0)
        sh = float(r.get('sharpe_proxy') or 0.0)
        tr = int(r.get('trades') or 0)
        eq_eur = em * 1000.0
        min_eur = me * 1000.0
        params_html = render_params(params) if params else '<div class=param><i>Paramètres indisponibles</i></div>'
        cards.append(f"""
        <div class=card>
          <div class=rank>{title} #{i}</div>
          <div class=money>💶 {eq_eur:,.0f} €</div>
          <div class=metrics>
            <div>📉 Max DD: <b>{md*100:.1f}%</b></div>
            <div>🔻 Pire solde: <b>{min_eur:,.0f} €</b></div>
            <div>📈 Sharpe*: <b>{sh:.2f}</b></div>
            <div>🔁 Trades: <b>{tr}</b></div>
          </div>
          <div class=params>
            {params_html}
          </div>
          <div class=curve>{f'<img src="{img_uri}" />' if img_uri else ''}</div>
          <div class=file>{r.get('file','')}</div>
        </div>
        """)
    return '\n'.join(cards)


def main() -> int:
    profile = 'pipeline_web6'
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    files = find_shared_jsons()
    if not files:
        print('Aucun shared_portfolio_*.json trouvé.')
        return 1

    entries = [extract_entry(p) for p in files]
    df = pd.DataFrame(entries)
    df = df[df['equity_mult'].notna()]
    if df.empty:
        print('Aucune métrique exploitable.')
        return 1

    top10, top5_dd, top5_perso = choose_sets(df)

    # Market data once
    market_data = fetch_market_data(profile)

    sec_top10 = build_section_html('Top equity', top10, market_data, profile)
    sec_dd = build_section_html('Mini DD + rendement', top5_dd, market_data, profile)
    sec_perso = build_section_html('Sélection assistant', top5_perso, market_data, profile)

    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    html = f"""<!doctype html>
    <html lang=fr><head><meta charset=utf-8>
    <title>MASTER REPORT — 10 + 5 + 5</title>
    <style>
      body{{font-family:Arial,sans-serif;background:#f7f7f9;color:#222;margin:0;padding:24px;}}
      h1{{margin:0 0 8px}} .sub{{color:#666;margin:0 0 16px}}
      h2{{margin:16px 0 8px;color:#0D47A1}}
      .grid{{display:grid;grid-template-columns:repeat(1,1fr);gap:12px;}}
      @media(min-width:1100px){{.grid{{grid-template-columns:repeat(2,1fr);}}}}
      .card{{background:#fff;border-radius:12px;padding:12px;box-shadow:0 2px 6px rgba(0,0,0,.08)}}
      .rank{{font-weight:bold;color:#555;margin-bottom:6px}}
      .money{{font-size:22px;font-weight:bold;color:#0D47A1;margin-bottom:6px}}
      .metrics{{display:grid;grid-template-columns:repeat(2,1fr);gap:4px;margin:6px 0}}
      .params .param{{font-size:13px;color:#444;margin:2px 0}}
      .curve img{{width:100%;height:auto;border-radius:6px;border:1px solid #eee;margin-top:6px}}
      .file{{color:#888;font-size:12px;margin-top:6px}}
      footer{{margin-top:16px;color:#777;font-size:12px}}
    </style></head><body>
    <h1>MASTER REPORT</h1>
    <p class=sub>10 meilleurs par equity, 5 à drawdown minimal + rendement, 5 choisis par l'assistant. Montants en euros sur 1 000 € de départ.</p>
    <h2>Top 10 — Equity</h2>
    <div class=grid>{sec_top10}</div>
    <h2>Top 5 — Mini DD + rendement</h2>
    <div class=grid>{sec_dd}</div>
    <h2>Top 5 — Sélection assistant</h2>
    <div class=grid>{sec_perso}</div>
    <footer>* Sharpe approximé. Généré {ts}.</footer>
    </body></html>"""

    out_html = os.path.join(out_dir, 'MASTER_REPORT.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(out_html)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


