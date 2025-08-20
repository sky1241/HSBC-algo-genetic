import os
import sys
import glob
import json
from datetime import datetime


def dd_percent(d: dict) -> float:
    min_eq = d.get('min_equity', None)
    if isinstance(min_eq, (int, float)) and 0 < float(min_eq) <= 1.5:
        return max(0.0, (1.0 - float(min_eq)) * 100.0)
    md = d.get('max_drawdown', None)
    try:
        md = float(md)
    except Exception:
        md = float('nan')
    return md * 100.0 if (md == md and md <= 5.0) else (md if md == md else float('nan'))


def fmt_eur(x):
    try:
        return f"{float(x):,.0f} €".replace(',', ' ').replace('\xa0', ' ')
    except Exception:
        return ''


def load_items(outputs_dir: str, profile: str):
    items = []
    for fp in glob.glob(os.path.join(outputs_dir, f"shared_portfolio_{profile}_*.json")):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                d = json.load(f)
            eq_mult = float(d.get('equity_mult', 1.0))
            equity_eur = eq_mult * 1000.0
            items.append((fp, equity_eur, dd_percent(d), int(d.get('trades', 0)), d))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def write_top3(outputs_dir: str, profile: str) -> str:
    items = load_items(outputs_dir, profile)
    out_path = os.path.join(outputs_dir, f"TOP3_{profile}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Top 3 — {profile}\n\n")
        for rank, it in enumerate(items[:3], start=1):
            fp, equity_eur, dd_pct, trades, d = it
            base = os.path.basename(fp)
            f.write(f"#{rank}  {base}\n")
            f.write(f"  Equity: {fmt_eur(equity_eur)}\n")
            f.write(f"  Max DD: {'' if dd_pct!=dd_pct else f'{dd_pct:.1f}%'}\n")
            f.write(f"  Trades: {trades}\n")
            params = d.get('best_params', {}) or d.get('params', {}) or {}
            for s in sorted(params.keys()):
                pm = params[s]
                f.write(
                    f"  - {s}: Tenkan {pm.get('tenkan')}, Kijun {pm.get('kijun')}, SenkouB {pm.get('senkou_b')}, Shift {pm.get('shift')}, ATR× {pm.get('atr_mult')}\n"
                )
            f.write("\n")
    return out_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    profile = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    p = write_top3(here, profile)
    print(p)


if __name__ == '__main__':
    main()


