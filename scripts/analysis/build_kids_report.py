import os
import glob
import pandas as pd
from datetime import datetime


def find_latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def risk_band(max_dd: float) -> tuple[str, str]:
    # max_dd is fraction (e.g., 0.35 = 35%)
    if max_dd is None:
        return ("?", "#9E9E9E")
    if max_dd <= 0.20:
        return ("Faible", "#4CAF50")  # green
    if max_dd <= 0.40:
        return ("Moyen", "#FFC107")  # amber
    return ("Fort", "#F44336")      # red


def stars(equity_mult: float, max_dd: float) -> int:
    if equity_mult is None or max_dd is None:
        return 3
    score = 0
    if equity_mult >= 20:
        score += 3
    elif equity_mult >= 10:
        score += 2
    elif equity_mult >= 5:
        score += 1
    if max_dd <= 0.20:
        score += 2
    elif max_dd <= 0.40:
        score += 1
    return max(1, min(5, score))


def render_card(idx: int, row: pd.Series) -> str:
    em = float(row.get('equity_mult') or 0.0)
    me = float(row.get('min_equity') or 0.0)
    md = float(row.get('max_drawdown') or 0.0)
    trades = int(row.get('trades') or 0)
    equity_eur = em * 1000.0 if em else 0.0
    min_eur = me * 1000.0 if me else 0.0
    risk_txt, risk_color = risk_band(md)
    s = stars(em, md)
    stars_txt = '★' * s + '☆' * (5 - s)
    md_pct = md * 100.0
    return f'''
    <div class="card">
      <div class="rank">Algo {idx}</div>
      <div class="big">💶 {equity_eur:,.0f} €</div>
      <div class="row"><span>🛡️ Risque:</span> <span class="pill" style="background:{risk_color}">{risk_txt}</span></div>
      <div class="row"><span>📉 Baisse maxi:</span> <strong>{md_pct:.0f}%</strong> <small>({min_eur:,.0f} € au plus bas)</small></div>
      <div class="row"><span>🔁 Trades:</span> <strong>{trades}</strong></div>
      <div class="row"><span>⭐ Note:</span> <span class="stars">{stars_txt}</span></div>
    </div>
    '''


def build_html(df: pd.DataFrame) -> str:
    cards_html = []
    for i, (_, r) in enumerate(df.head(10).iterrows(), start=1):
        cards_html.append(render_card(i, r))
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    return f'''<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Top 10 — Rapport Kids</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#fafafa; color:#222; margin:0; padding:24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p.subtitle {{ margin:0 0 16px; color:#555; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:16px; }}
    @media (min-width: 1100px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
    .card {{ background:#fff; border-radius:12px; padding:16px; box-shadow:0 2px 6px rgba(0,0,0,0.08); }}
    .rank {{ font-weight:bold; color:#666; margin-bottom:8px; }}
    .big {{ font-size:28px; font-weight:bold; color:#0D47A1; margin-bottom:12px; }}
    .row {{ display:flex; justify-content:space-between; align-items:center; margin:6px 0; }}
    .pill {{ color:#fff; padding:2px 8px; border-radius:999px; font-weight:bold; }}
    .stars {{ color:#FFC107; letter-spacing:2px; }}
    footer {{ margin-top:16px; color:#777; font-size:12px; }}
  </style>
  </head>
  <body>
    <h1>Top 10 algos (simple)</h1>
    <p class="subtitle">Montants en euros sur 1 000 € de départ. Regarde l'argent, le risque, la baisse maxi et le nombre de trades.</p>
    <div class="grid">
      {''.join(cards_html)}
    </div>
    <footer>Généré: {ts}</footer>
  </body>
</html>'''


def main() -> int:
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    # Priorité: top-10 du portefeuille partagé
    csv_path = find_latest(os.path.join(out_dir, 'shared_top10_by_equity_mult_*.csv'))
    if csv_path is None:
        # Fallback: top-10 individuels
        csv_path = find_latest(os.path.join(out_dir, 'top10_by_equity_mult_*.csv'))
    df = None
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        # Normaliser colonnes attendues
        for col in ['equity_mult', 'max_drawdown', 'min_equity']:
            if col not in df.columns:
                df[col] = None
        if 'trades' not in df.columns:
            df['trades'] = 0
    # Si on a moins de 10 lignes, fallback sur le dernier runs complet et prendre top 10
    if df is None or len(df) < 10:
        runs_path = find_latest(os.path.join(out_dir, 'runs_*_*.csv'))
        if runs_path is None:
            print('Aucun fichier de résultats suffisant pour 10 algos dans outputs/.')
            return 1
        df_runs = pd.read_csv(runs_path)
        for col in ['equity_mult', 'max_drawdown', 'min_equity']:
            if col not in df_runs.columns:
                df_runs[col] = None
        if 'trades' not in df_runs.columns:
            df_runs['trades'] = 0
        df = df_runs.sort_values('equity_mult', ascending=False).head(10)

    html = build_html(df)
    out_html = os.path.join(out_dir, 'KIDS_REPORT.html')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(out_html)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


