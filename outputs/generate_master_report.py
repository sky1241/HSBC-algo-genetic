import os
import sys
import json
import glob
import io
import base64
from datetime import datetime, timezone


def find_live_json(profile: str) -> str | None:
    # Cherche d'abord dans outputs (repo root)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    cand = [
        os.path.join(repo_root, 'outputs', f'LIVE_BEST_{profile}.json'),
        os.path.join(here, f'LIVE_BEST_{profile}.json'),
    ]
    # Fallback éventuel: dossier live temp si défini
    live_dir = os.environ.get('ICHIMOKU_LIVE_DIR') or r"C:\\Temp\\ichimoku_live"
    cand.append(os.path.join(live_dir, f'LIVE_BEST_{profile}.json'))
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def fmt_eur(x):
    try:
        return f"{float(x):,.0f} €".replace(',', ' ').replace('\xa0', ' ')
    except Exception:
        return ''


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    here = os.path.dirname(os.path.abspath(__file__))
    out_html_tmp = os.path.join(here, 'MASTER_REPORT.html.tmp')
    out_html = os.path.join(here, 'MASTER_REPORT.html')
    run_label = os.environ.get('RUN_LABEL', '')
    # Optional filter: only include snapshots at or after this ts label (YYYYMMDD_HHMMSS)
    since_label = os.environ.get('REPORT_SINCE_TS_LABEL')
    since_dt = None
    if since_label:
        try:
            since_dt = datetime.strptime(since_label, '%Y%m%d_%H%M%S')
        except Exception:
            since_dt = None

    # 1) Charger le LIVE (fallback si pas d'archives)
    live_path = find_live_json(profile)
    live_data = None
    if live_path and os.path.exists(live_path):
        try:
            with open(live_path, 'r', encoding='utf-8') as f:
                live_data = json.load(f)
        except Exception:
            live_data = None

    # Charger le dernier snapshot si présent (pour courbe d'équity globale)
    latest_snapshot = None
    latest_path = os.path.join(here, f'shared_portfolio_{profile}_latest.json')
    if os.path.exists(latest_path):
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                latest_snapshot = json.load(f)
        except Exception:
            latest_snapshot = None

    # 2) Scanner les archives de portefeuilles dans outputs
    shared_items = []  # list of dicts: {file, equity_eur, dd_pct, min_eur, sharpe, trades, params_map}
    for fp in sorted(glob.glob(os.path.join(here, 'shared_portfolio_*.json'))):
        # Filter by since_dt if provided, using filename ts label when possible
        if since_dt:
            try:
                base_no_ext = os.path.basename(fp).replace('.json', '')
                parts = base_no_ext.split('_')
                ts_part = '_'.join(parts[-2:]) if len(parts) >= 2 else ''
                f_dt = datetime.strptime(ts_part, '%Y%m%d_%H%M%S')
                if f_dt < since_dt:
                    continue
            except Exception:
                # Fallback: filter by mtime
                try:
                    mtime_dt = datetime.fromtimestamp(os.path.getmtime(fp))
                    if mtime_dt < since_dt:
                        continue
                except Exception:
                    pass
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                d = json.load(f)
            eq_mult = float(d.get('equity_mult', 1.0))
            equity_eur = eq_mult * 1000.0
            min_eq = d.get('min_equity', None)
            # DD robuste: privilégier min_equity si disponible (dd ≈ 1 - min_equity)
            if isinstance(min_eq, (int, float)) and 0 < float(min_eq) <= 1.5:
                dd_pct = max(0.0, (1.0 - float(min_eq)) * 100.0)
            else:
                md = d.get('max_drawdown', None)
                try:
                    md = float(md)
                except Exception:
                    md = float('nan')
                # Heuristique: si md <= 5 → suppose ratio, sinon déjà en %
                if md == md:
                    dd_pct = md * 100.0 if md <= 5.0 else md
                else:
                    dd_pct = float('nan')
            min_eur = float(min_eq) * 1000.0 if isinstance(min_eq, (int, float)) else float('nan')
            sharpe = float(d.get('sharpe_proxy', float('nan')))
            trades = int(d.get('trades', 0))
            params_map = d.get('best_params', {}) or d.get('params', {}) or {}
            per_symbol = d.get('per_symbol', {}) if isinstance(d, dict) else {}
            equity_curve = d.get('equity_curve', None)
            shared_items.append({
                'file': os.path.basename(fp),
                'equity_eur': equity_eur,
                'dd_pct': dd_pct,
                'min_eur': min_eur,
                'sharpe': sharpe,
                'trades': trades,
                'params_map': params_map,
                'per_symbol': per_symbol,
                'equity_curve': equity_curve,
            })
        except Exception:
            continue

    # Si aucune archive mais LIVE dispo, créer une archive minimale immédiate
    if not shared_items and live_data:
        try:
            from datetime import datetime as _dt
            ts_label = _dt.utcnow().strftime('%Y%m%d_%H%M%S')
            d = live_data
            s = d.get('shared_metrics', {}) if isinstance(d, dict) else {}
            s = dict(s) if isinstance(s, dict) else {}
            s['best_params'] = d.get('best_params', {}) if isinstance(d, dict) else {}
            fp = os.path.join(here, f'shared_portfolio_{profile}_{ts_label}.json')
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
            # Re-scan juste ce fichier
            d = s
            eq_mult = float(d.get('equity_mult', 1.0))
            equity_eur = eq_mult * 1000.0
            min_eq = d.get('min_equity', None)
            if isinstance(min_eq, (int, float)) and 0 < float(min_eq) <= 1.5:
                dd_pct = max(0.0, (1.0 - float(min_eq)) * 100.0)
            else:
                md = d.get('max_drawdown', None)
                try:
                    md = float(md)
                except Exception:
                    md = float('nan')
                dd_pct = md * 100.0 if (md == md and md <= 5.0) else (md if md == md else float('nan'))
            min_eur = float(min_eq) * 1000.0 if isinstance(min_eq, (int, float)) else float('nan')
            sharpe = float(d.get('sharpe_proxy', float('nan')))
            trades = int(d.get('trades', 0))
            params_map = d.get('best_params', {}) or d.get('params', {}) or {}
            per_symbol = d.get('per_symbol', {}) if isinstance(d, dict) else {}
            equity_curve = d.get('equity_curve', None)
            shared_items.append({
                'file': os.path.basename(fp),
                'equity_eur': equity_eur,
                'dd_pct': dd_pct,
                'min_eur': min_eur,
                'sharpe': sharpe,
                'trades': trades,
                'params_map': params_map,
                'per_symbol': per_symbol,
                'equity_curve': equity_curve,
            })
        except Exception:
            pass

    # Helpers
    def fmt_params(pm: dict) -> str:
        return (
            f"Tenkan {pm.get('tenkan','?')}, Kijun {pm.get('kijun','?')}, "
            f"SenkouB {pm.get('senkou_b','?')}, Shift {pm.get('shift','?')}, "
            f"ATR× {pm.get('atr_mult','?')}"
        )

    def block_for_item(idx: int, item: dict, title_prefix: str) -> str:
        lines = []
        lines.append(f"<h3>{title_prefix} #{idx}</h3>")
        lines.append(f"<p>💶 {fmt_eur(item['equity_eur'])}<br/>")
        lines.append(f"📉 Max DD: {'' if item['dd_pct']!=item['dd_pct'] else f'{item['dd_pct']:.1f}%'}<br/>")
        min_txt = '' if item['min_eur']!=item['min_eur'] else fmt_eur(item['min_eur'])
        lines.append(f"🔻 Pire solde: {min_txt}<br/>")
        sharpe_txt = '' if item['sharpe']!=item['sharpe'] else f"{item['sharpe']:.2f}"
        lines.append(f"📈 Sharpe*: {sharpe_txt}<br/>")
        lines.append(f"🔁 Trades: {item['trades']}</p>")
        # Graphiques: courbe d'équity et barres P&L par symbole (si dispo)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # Equity curve + barres P&L par symbole si disponible
            curve_src = item.get('equity_curve')
            if not (isinstance(curve_src, list) and len(curve_src) > 1):
                # Fallback: utiliser la dernière courbe globale disponible
                if isinstance(latest_snapshot, dict):
                    curve_src = latest_snapshot.get('equity_curve')
                elif isinstance(live_data, dict) and isinstance(live_data.get('shared_metrics'), dict):
                    curve_src = live_data['shared_metrics'].get('equity_curve')
            if isinstance(curve_src, list) and len(curve_src) > 1:
                try:
                    from matplotlib import dates as mdates
                    # Extraire dates et valeurs
                    def extract(point):
                        if isinstance(point, dict):
                            return point.get('timestamp'), point.get('equity_mult')
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            return point[0], point[1]
                        return None, None
                    dts_raw, ys_raw = zip(*(extract(p) for p in curve_src))
                    def parse_dt(x):
                        try:
                            return datetime.fromisoformat(x.replace('Z','+00:00')).replace(tzinfo=None)
                        except Exception:
                            return None
                    dts = [parse_dt(x) if isinstance(x, str) else x for x in dts_raw]
                    dts = [dt for dt in dts if isinstance(dt, datetime)]
                    ys = []
                    for p in curve_src:
                        _, y = extract(p)
                        try:
                            ys.append(float(y))
                        except Exception:
                            ys.append(float('nan'))
                    fig, ax = plt.subplots(figsize=(7.5, 2.2), dpi=110)
                    ax.plot(dts, ys, color='#0D47A1', linewidth=1.6)
                    ax.set_title('Equity curve (×)', fontsize=10)
                    ax.grid(alpha=0.25)
                    try:
                        locator = mdates.AutoDateLocator()
                        ax.xaxis.set_major_locator(locator)
                        try:
                            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                        except Exception:
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    except Exception:
                        pass
                    ax.tick_params(axis='x', labelsize=8)
                    try:
                        start_txt = dts[0].strftime('%Y-%m-%d')
                        end_txt = dts[-1].strftime('%Y-%m-%d')
                        fig.text(0.01, 0.02, start_txt, fontsize=8, color='#444')
                        fig.text(0.99, 0.02, end_txt, fontsize=8, color='#444', ha='right')
                    except Exception:
                        pass
                    plt.subplots_adjust(bottom=0.22)
                    ax.set_ylabel('×')
                    try:
                        from matplotlib.ticker import MaxNLocator
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    except Exception:
                        pass
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    plt.close(fig)
                    data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
                    lines.append(f"<img alt='equity' src='{data_uri}' style='display:block;margin:6px 0;max-width:100%;' />")
                except Exception:
                    pass

            # Barres P&L par symbole (si per_symbol a des pnl)
            sym_pnls = []
            ps = item.get('per_symbol') if isinstance(item.get('per_symbol'), dict) else {}
            for s, sd in sorted(ps.items()):
                try:
                    pnl = float(sd.get('pnl_eur', 'nan'))
                except Exception:
                    pnl = float('nan')
                sym_pnls.append((s, pnl))
            valid_pnls = [(s, p) for s, p in sym_pnls if p == p]
            if valid_pnls:
                labels = [s for s, _ in valid_pnls]
                values = [p for _, p in valid_pnls]
                try:
                    fig, ax = plt.subplots(figsize=(7, 2.2), dpi=110)
                    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in values]
                    ax.bar(range(len(values)), values, color=colors)
                    ax.set_title('P&L par symbole (EUR)', fontsize=10)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                    ax.grid(axis='y', alpha=0.25)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    plt.close(fig)
                    data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
                    lines.append(f"<img alt='pnl_by_symbol' src='{data_uri}' style='display:block;margin:6px 0;max-width:100%;' />")
                except Exception:
                    # Fallback SVG inline horizontal bars
                    try:
                        max_abs = max(abs(v) for v in values) or 1.0
                        width = 750
                        bar_h = 16
                        gap = 6
                        pad = 6
                        total_h = pad*2 + len(values)*(bar_h+gap)
                        svg = [f"<svg width='{width}' height='{total_h}' viewBox='0 0 {width} {total_h}' xmlns='http://www.w3.org/2000/svg' style='display:block;margin:6px 0;max-width:100%'>"]
                        svg.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
                        mid_x = width/2
                        # axis
                        svg.append(f"<line x1='{mid_x}' y1='{pad}' x2='{mid_x}' y2='{total_h-pad}' stroke='#999' stroke-width='1' />")
                        for i,(lab,val) in enumerate(values):
                            y = pad + i*(bar_h+gap)
                            w = (width/2 - 40) * (abs(val)/max_abs)
                            x = mid_x - w if val < 0 else mid_x
                            color = '#C62828' if val < 0 else '#2E7D32'
                            svg.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{bar_h}' fill='{color}' />")
                            svg.append(f"<text x='{5}' y='{y+bar_h-4}' font-size='11' fill='#333'>{lab}</text>")
                            svg.append(f"<text x='{width-5}' y='{y+bar_h-4}' font-size='11' fill='#333' text-anchor='end'>{val:,.0f} €</text>")
                        svg.append("</svg>")
                        lines.append(''.join(svg))
                    except Exception:
                        pass
        except Exception:
            pass
        # params list
        for s in sorted(item['params_map'].keys()):
            lines.append(f"<div>{s}: {fmt_params(item['params_map'][s])}</div>")
        lines.append(f"<div style='color:#666'>{item['file']}</div>")
        return '\n'.join(lines)

    # Build sections
    now_txt = datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M UTC')
    secs = []

    # Construire des timelines globales à partir des snapshots (fallback pour graphes)
    timeline_points = []  # list of (dt, equity_eur)
    dd_timeline_points = []  # list of (dt, dd_pct)
    for it in shared_items:
        fname = it['file']
        # attend format *_YYYYMMDD_HHMMSS.json
        try:
            base_no_ext = fname.replace('.json', '')
            parts = base_no_ext.split('_')
            if len(parts) < 2:
                continue
            ts_part = '_'.join(parts[-2:])
            dt = datetime.strptime(ts_part, '%Y%m%d_%H%M%S')
            timeline_points.append((dt, float(it['equity_eur'])))
            dd_val = it.get('dd_pct')
            try:
                dd_val = float(dd_val)
            except Exception:
                dd_val = float('nan')
            dd_timeline_points.append((dt, dd_val))
        except Exception:
            continue
    timeline_points.sort(key=lambda x: x[0])
    dd_timeline_points.sort(key=lambda x: x[0])

    # Top 10 — Equity
    if shared_items:
        top_equity = sorted(shared_items, key=lambda x: (x['equity_eur']), reverse=True)[:10]
        blocks = [block_for_item(i+1, it, 'Top equity') for i, it in enumerate(top_equity)]
        # Si aucune courbe par item n'est dispo, ajouter une courbe globale timeline
        if blocks and timeline_points and not any(isinstance(it.get('equity_curve'), list) and len(it['equity_curve'])>1 for it in top_equity):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib import dates as mdates
                dts = [p[0] for p in timeline_points]
                ys = [p[1]/1000.0 for p in timeline_points]  # normaliser en multipliers pour lisibilité
                fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
                ax.plot(dts, ys, color='#0D47A1', linewidth=1.6)
                ax.set_title('Évolution equity (×) — snapshots', fontsize=10)
                ax.grid(alpha=0.25)
                try:
                    locator = mdates.AutoDateLocator()
                    ax.xaxis.set_major_locator(locator)
                    try:
                        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    except Exception:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                except Exception:
                    pass
                ax.tick_params(axis='x', labelsize=8)
                try:
                    start_txt = dts[0].strftime('%Y-%m-%d')
                    end_txt = dts[-1].strftime('%Y-%m-%d')
                    fig.text(0.01, 0.02, start_txt, fontsize=8, color='#444')
                    fig.text(0.99, 0.02, end_txt, fontsize=8, color='#444', ha='right')
                except Exception:
                    pass
                plt.subplots_adjust(bottom=0.22)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
                blocks.insert(0, f"<img alt='equity_timeline' src='{data_uri}' style='display:block;margin:6px 0;max-width:100%;' />")
            except Exception:
                pass
        secs.append('<h2>Top 10 — Equity</h2>' + '\n'.join(blocks))
    else:
        secs.append('<h2>Top 10 — Equity</h2><p>Aucune archive disponible pour le moment.</p>')

    # Top 5 — Mini DD + rendement (trier par DD croissant puis equity décroissant)
    if shared_items:
        top_minidd = sorted(shared_items, key=lambda x: (x['dd_pct'], -x['equity_eur']))[:5]
        blocks = [block_for_item(i+1, it, 'Mini DD + rendement') for i, it in enumerate(top_minidd)]
        secs.append('<h2>Top 5 — Mini DD + rendement</h2>' + '\n'.join(blocks))
    else:
        secs.append('<h2>Top 5 — Mini DD + rendement</h2><p>Aucune archive disponible pour le moment.</p>')

    # Top 5 — Sélection assistant (Sharpe décroissant)
    if shared_items:
        sel = sorted(shared_items, key=lambda x: (float('-inf') if x['sharpe']!=x['sharpe'] else x['sharpe']), reverse=True)[:5]
        blocks = [block_for_item(i+1, it, 'Sélection assistant') for i, it in enumerate(sel)]
        secs.append('<h2>Top 5 — Sélection assistant</h2>' + '\n'.join(blocks))
    else:
        secs.append('<h2>Top 5 — Sélection assistant</h2><p>Aucune archive disponible pour le moment.</p>')

    # Si pas d'archives, afficher au moins le LIVE courant (table légère)
    if not shared_items and live_data:
        updated_at = live_data.get('updated_at') or now_txt
        best_params = live_data.get('best_params', {}) if isinstance(live_data, dict) else {}
        shared = live_data.get('shared_metrics', {}) if isinstance(live_data, dict) else {}
        per_sym = shared.get('per_symbol', {}) if isinstance(shared, dict) else {}
        try:
            eq_mult = float(shared.get('equity_mult', 1.0)) if isinstance(shared, dict) else 1.0
        except Exception:
            eq_mult = 1.0
        equity_eur = eq_mult * 1000.0
        try:
            dd_pct = float(shared.get('max_drawdown', 0.0)) * 100.0
        except Exception:
            dd_pct = float('nan')
        trades = int(shared.get('trades', 0)) if isinstance(shared, dict) else 0

        rows = []
        for s in sorted(best_params.keys()):
            pm = best_params[s]
            ps = per_sym.get(s, {}) if isinstance(per_sym, dict) else {}
            pnl_eur = ps.get('pnl_eur', '')
            dd_idx = ps.get('max_dd_indexed', '')
            dd_idx_txt = (f"{float(dd_idx)*100:.1f}%" if isinstance(dd_idx, (int,float)) else '')
            rows.append(
                f"<tr><td>{s}</td><td>{fmt_params(pm)}</td>"
                f"<td style='text-align:right'>{fmt_eur(pnl_eur)}</td>"
                f"<td style='text-align:right'>{dd_idx_txt}</td></tr>"
            )
        # Mini graphes à partir du LIVE si dispo
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import dates as mdates
            imgs = []
            if isinstance(shared.get('equity_curve'), list) and len(shared['equity_curve']) > 1:
                def extract(point):
                    if isinstance(point, dict):
                        return point.get('timestamp'), point.get('equity_mult')
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        return point[0], point[1]
                    return None, None
                dts_raw, ys_raw = zip(*(extract(p) for p in shared['equity_curve']))
                def parse_dt(x):
                    try:
                        return datetime.fromisoformat(x.replace('Z','+00:00')).replace(tzinfo=None)
                    except Exception:
                        return None
                dts = [parse_dt(x) if isinstance(x, str) else x for x in dts_raw]
                dts = [dt for dt in dts if isinstance(dt, datetime)]
                ys = []
                for p in shared['equity_curve']:
                    _, y = extract(p)
                    try:
                        ys.append(float(y))
                    except Exception:
                        ys.append(float('nan'))
                fig, ax = plt.subplots(figsize=(7, 2.2), dpi=110)
                ax.plot(dts, ys, color='#0D47A1', linewidth=1.6)
                ax.set_title('Equity curve (×)', fontsize=10)
                ax.grid(alpha=0.25)
                try:
                    locator = mdates.AutoDateLocator()
                    ax.xaxis.set_major_locator(locator)
                    try:
                        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    except Exception:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                except Exception:
                    pass
                ax.tick_params(axis='x', labelsize=8)
                plt.subplots_adjust(bottom=0.22)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                data_uri_eq = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
                imgs.append(f"<img alt='equity' src='{data_uri_eq}' style='display:block;margin:6px 0;max-width:100%;' />")
        except Exception:
            imgs = []
        secs.append(
            '<h2>État LIVE (fallback)</h2>' +
            f"<p>Equity: <b>{fmt_eur(equity_eur)}</b> — Max DD: <b>{'' if dd_pct!=dd_pct else f'{dd_pct:.1f}%'}</b> — Trades: <b>{trades}</b></p>" +
            '<table><tr><th>Symbole</th><th>Paramètres</th><th>P&L</th><th>Max DD (indexé portefeuille)</th></tr>' +
            ''.join(rows) + '</table>' + ''.join(imgs)
        )

    # Image inline (base64) de la courbe d'équity de la run en cours (depuis LIVE), si dispo
    live_equity_img = ''
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import dates as mdates
        if isinstance(live_data, dict):
            sm = live_data.get('shared_metrics') if isinstance(live_data.get('shared_metrics'), dict) else None
            curve = sm.get('equity_curve') if sm else None
            # Fallback: si pas de courbe dans LIVE, prendre la plus récente des snapshots
            if not (isinstance(curve, list) and len(curve) > 1):
                latest = latest_snapshot if isinstance(latest_snapshot, dict) else None
                if latest and isinstance(latest.get('equity_curve'), list) and len(latest['equity_curve']) > 1:
                    curve = latest['equity_curve']
            if isinstance(curve, list) and len(curve) > 1:
                # Construire axes temporels + annotations début/fin
                def extract(point):
                    if isinstance(point, dict):
                        return point.get('timestamp'), point.get('equity_mult')
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        return point[0], point[1]
                    return None, None
                dts_raw, ys_raw = zip(*(extract(p) for p in curve))
                def parse_dt(x):
                    try:
                        # Support ISO datetime
                        return datetime.fromisoformat(x.replace('Z','+00:00')).replace(tzinfo=None)
                    except Exception:
                        return None
                dts = [parse_dt(x) if isinstance(x, str) else x for x in dts_raw]
                dts = [dt for dt in dts if isinstance(dt, datetime)]
                ys = []
                for p in curve:
                    _, y = extract(p)
                    try:
                        ys.append(float(y))
                    except Exception:
                        ys.append(float('nan'))
                if dts and ys:
                    fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
                    ax.plot(dts, ys, color='#0D47A1', linewidth=1.8)
                    ax.set_title('Équity (×) — run en cours', fontsize=11)
                    ax.grid(alpha=0.3)
                    try:
                        locator = mdates.AutoDateLocator()
                        ax.xaxis.set_major_locator(locator)
                        try:
                            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                        except Exception:
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    except Exception:
                        pass
                    ax.tick_params(axis='x', labelsize=8)
                    try:
                        start_txt = dts[0].strftime('%Y-%m-%d')
                        end_txt = dts[-1].strftime('%Y-%m-%d')
                        fig.text(0.01, 0.02, start_txt, fontsize=8, color='#444')
                        fig.text(0.99, 0.02, end_txt, fontsize=8, color='#444', ha='right')
                    except Exception:
                        pass
                    plt.subplots_adjust(bottom=0.22)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                live_equity_img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        live_equity_img = ''

    # Générer/rafraîchir les graphes PNG de timeline
    eq_png = dd_png = None
    try:
        from outputs.build_graphs_from_snapshots import save_timeline_graphs
    except Exception:
        save_timeline_graphs = None
    if save_timeline_graphs:
        try:
            eq_png, dd_png = save_timeline_graphs(here, profile)
        except Exception:
            eq_png = dd_png = None

    # Convertir en images inline (base64) pour garantir l'affichage
    eq_timeline_img = ''
    dd_timeline_img = ''
    try:
        if eq_png and os.path.exists(eq_png):
            # Préférer un lien de fichier relatif pour compatibilité navigateur
            eq_timeline_img = f"graphs/{os.path.basename(eq_png)}"
        elif timeline_points:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                xs = list(range(len(timeline_points)))
                ys = [p[1] / 1000.0 for p in timeline_points]
                fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
                ax.plot(xs, ys, color='#0D47A1', linewidth=1.8)
                ax.set_title("Évolution equity (×) — snapshots", fontsize=11)
                ax.grid(alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                eq_timeline_img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
            except Exception:
                eq_timeline_img = ''
    except Exception:
        eq_timeline_img = ''

    try:
        if dd_png and os.path.exists(dd_png):
            dd_timeline_img = f"graphs/{os.path.basename(dd_png)}"
        elif dd_timeline_points:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                xs = list(range(len(dd_timeline_points)))
                ys = [p[1] for p in dd_timeline_points]
                fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
                ax.plot(xs, ys, color='#C62828', linewidth=1.8)
                ax.set_title("Évolution max drawdown (%) — snapshots", fontsize=11)
                ax.grid(alpha=0.3)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                dd_timeline_img = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
            except Exception:
                dd_timeline_img = ''
    except Exception:
        dd_timeline_img = ''

    title_extra = f" — {run_label}" if run_label else ""
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>MASTER REPORT — {profile}{title_extra}</title>
    <style>body{{font-family:Arial,sans-serif;padding:16px;background:#fafafa}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px}}</style>
    </head><body>
    <h1>MASTER REPORT — {profile}{title_extra}</h1>
    <p style='color:#666'>* Sharpe approximé. Généré {now_txt}.</p>
    {f"<p style='color:#444'>Label: <b>{run_label}</b></p>" if run_label else ''}
    {f"<p style='color:#444'>Filtré depuis: <b>{since_label}</b></p>" if since_label else ''}
    <p style='color:#333'>Réglages backtest courants: Levier = {'' if not (isinstance(live_data, dict) and isinstance(live_data.get('shared_metrics'), dict)) else live_data['shared_metrics'].get('leverage', '')} ×, Taille position = {'' if not (isinstance(live_data, dict) and isinstance(live_data.get('shared_metrics'), dict)) else f"{float(live_data['shared_metrics'].get('position_size', 0.0))*100:.2f}%"}, Max positions/side = {'' if not (isinstance(live_data, dict) and isinstance(live_data.get('shared_metrics'), dict)) else live_data['shared_metrics'].get('max_positions_per_side', '')}</p>
    {f"<img alt='equity_live' src='{live_equity_img}' style='display:block;margin:6px 0;max-width:100%;' />" if live_equity_img else ''}
    {f"<img alt='equity_timeline' src='{eq_timeline_img}' style='display:block;margin:6px 0;max-width:100%;' />" if eq_timeline_img else ''}
    {f"<img alt='dd_timeline' src='{dd_timeline_img}' style='display:block;margin:6px 0;max-width:100%;' />" if dd_timeline_img else ''}
    {''.join(secs)}
    </body></html>"""

    with open(out_html_tmp, 'w', encoding='utf-8') as f:
        f.write(html)
    os.replace(out_html_tmp, out_html)
    print(out_html)

    # 3) Écrire un résumé CSV (Top 3 equity) pour comparatifs inter-runs
    try:
        import csv
        summary_path_tmp = os.path.join(here, 'MASTER_SUMMARY.csv.tmp')
        summary_path = os.path.join(here, 'MASTER_SUMMARY.csv')
        top3 = sorted(shared_items, key=lambda x: (x['equity_eur']), reverse=True)[:3] if shared_items else []
        with open(summary_path_tmp, 'w', encoding='utf-8', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['label','profile','generated_utc','rank','file','equity_eur','dd_pct','min_eur','sharpe','trades'])
            for idx, it in enumerate(top3, start=1):
                writer.writerow([
                    run_label,
                    profile,
                    now_txt,
                    idx,
                    it.get('file',''),
                    f"{it.get('equity_eur',float('nan')):.0f}",
                    f"{it.get('dd_pct',float('nan')):.2f}",
                    f"{it.get('min_eur',float('nan')):.0f}",
                    f"{it.get('sharpe',float('nan')):.2f}",
                    it.get('trades',0)
                ])
        os.replace(summary_path_tmp, summary_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()


