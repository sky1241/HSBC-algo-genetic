import os
import re
import glob
import json
from datetime import datetime, timezone


def fmt_eur(x):
    try:
        return f"{float(x):,.0f} €".replace(',', ' ').replace('\xa0', ' ')
    except Exception:
        return ''


def latest_snapshot_path(outputs_dir: str, profile: str) -> str | None:
    paths = glob.glob(os.path.join(outputs_dir, f"shared_portfolio_{profile}_*.json"))
    if not paths:
        return None
    def key_fn(p: str) -> float:
        # Prefer timestamp embedded in filename; fallback to mtime
        m = re.search(r"_(\d{8}_\d{6})\.json$", os.path.basename(p))
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
        try:
            return float(os.path.getmtime(p))
        except Exception:
            return 0.0
    return max(paths, key=key_fn)


def read_json(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def make_summary(outputs_dir: str, profile: str) -> str:
    snap_path = latest_snapshot_path(outputs_dir, profile)
    live_path = os.path.join(outputs_dir, f"LIVE_BEST_{profile}.json")
    snap = read_json(snap_path) if snap_path else None
    live = read_json(live_path)

    lines = []
    lines.append(f"Résumé — {profile}")
    lines.append("")

    # Snapshot block
    if snap is not None and snap_path:
        try:
            bn = os.path.basename(snap_path)
            m = re.search(r"_(\d{8}_\d{6})\.json$", bn)
            ts = (m.group(1) if m else 'latest')
        except Exception:
            ts = 'latest'
        eq_mult = float(snap.get('equity_mult', 1.0))
        equity_eur = eq_mult * 1000.0
        # Prefer explicit max_drawdown (fraction) if present; fallback to min_equity
        md_val = snap.get('max_drawdown', None)
        dd_pct: float
        try:
            if isinstance(md_val, (int, float)):
                dd_pct = float(md_val) * 100.0
            else:
                raise ValueError
        except Exception:
            min_eq = snap.get('min_equity')
            if isinstance(min_eq, (int, float)) and 0 < float(min_eq) <= 1.5:
                dd_pct = max(0.0, (1.0 - float(min_eq)) * 100.0)
            else:
                dd_pct = float('nan')
        trades = int(snap.get('trades', 0))
        lines.append(f"Snapshot: {ts}")
        lines.append(f"Equity: {fmt_eur(equity_eur)}  |  Max DD: {'' if dd_pct!=dd_pct else f'{dd_pct:.1f}%'}  |  Trades: {trades}")
        per_sym = snap.get('per_symbol', {}) if isinstance(snap, dict) else {}
        if isinstance(per_sym, dict) and per_sym:
            for s in sorted(per_sym.keys()):
                ps = per_sym[s]
                pnl = ps.get('pnl_eur', '')
                ddx = ps.get('max_dd_indexed', '')
                ddx_txt = (f"{float(ddx)*100:.1f}%" if isinstance(ddx, (int,float)) else '')
                lines.append(f" - {s}: P&L {fmt_eur(pnl)} | DD idx {ddx_txt}")
    else:
        lines.append("Pas encore de snapshot.")

    # Best params (live)
    if isinstance(live, dict):
        best = live.get('best_params', {}) or {}
        if best:
            lines.append("")
            lines.append("Paramètres actuels (meilleurs trouvés):")
            for s in sorted(best.keys()):
                pm = best[s]
                lines.append(
                    f" - {s}: Tenkan {pm.get('tenkan')}, Kijun {pm.get('kijun')}, SenkouB {pm.get('senkou_b')}, Shift {pm.get('shift')}, ATR× {pm.get('atr_mult')}"
                )

    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    profile = os.environ.get('ICHIMOKU_PROFILE', 'pipeline_web6')
    txt_path = os.path.join(here, f"SUMMARY_{profile}.txt")
    content = make_summary(here, profile)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content + "\n")
    print(txt_path)


if __name__ == '__main__':
    main()


