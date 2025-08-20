import os
import glob
import json
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_snapshots(outputs_dir: str, profile: str):
    items = []
    pattern = os.path.join(outputs_dir, f"shared_portfolio_{profile}_*.json")
    for fp in sorted(glob.glob(pattern)):
        base = os.path.basename(fp)
        # Expect *_YYYYMMDD_HHMMSS.json (join last two segments before extension)
        try:
            base_no_ext = os.path.splitext(base)[0]
            parts = base_no_ext.split('_')
            if len(parts) < 2:
                continue
            ts = '_'.join(parts[-2:])
            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
        except Exception:
            continue
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        try:
            eq_mult = float(d.get('equity_mult', 1.0))
            equity_eur = eq_mult * 1000.0
        except Exception:
            equity_eur = float('nan')
        # Drawdown heuristic
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
        items.append((dt, equity_eur, dd_pct))
    items.sort(key=lambda x: x[0])
    return items


def save_timeline_graphs(outputs_dir: str, profile: str):
    items = load_snapshots(outputs_dir, profile)
    graphs_dir = os.path.join(outputs_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    if not items:
        return None
    dts = [dt for dt, _, _ in items]
    eq_mults = [(v/1000.0) if (v == v) else float('nan') for _, v, _ in items]
    dds = [dd for _, _, dd in items]

    # Equity timeline (×)
    try:
        from matplotlib import dates as mdates
        fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
        ax.plot(dts, eq_mults, color='#0D47A1', linewidth=1.8)
        ax.set_title('Évolution equity (×) — snapshots', fontsize=11)
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
        # Dates début/fin en annotation bas-gauche/droite
        try:
            start_txt = dts[0].strftime('%Y-%m-%d')
            end_txt = dts[-1].strftime('%Y-%m-%d')
            fig.text(0.01, 0.02, start_txt, fontsize=8, color='#444')
            fig.text(0.99, 0.02, end_txt, fontsize=8, color='#444', ha='right')
        except Exception:
            pass
        plt.subplots_adjust(bottom=0.22)
        ax.set_ylabel('Equity (×)')
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        except Exception:
            pass
        path_eq = os.path.join(graphs_dir, f'equity_timeline_{profile}.png')
        fig.savefig(path_eq, format='png')
        plt.close(fig)
    except Exception:
        path_eq = None

    # Drawdown timeline (%)
    try:
        from matplotlib import dates as mdates
        fig, ax = plt.subplots(figsize=(7.5, 2.4), dpi=110)
        ax.plot(dts, dds, color='#C62828', linewidth=1.8)
        ax.set_title('Évolution max drawdown (%) — snapshots', fontsize=11)
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
        ax.set_ylabel('DD (%)')
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        except Exception:
            pass
        path_dd = os.path.join(graphs_dir, f'dd_timeline_{profile}.png')
        fig.savefig(path_dd, format='png')
        plt.close(fig)
    except Exception:
        path_dd = None

    return path_eq, path_dd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    profile = os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    paths = save_timeline_graphs(here, profile)
    if paths:
        print('\n'.join([p for p in paths if p]))


if __name__ == '__main__':
    main()


