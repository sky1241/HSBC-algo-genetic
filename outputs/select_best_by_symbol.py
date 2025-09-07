import os
import re
import json
import glob
from datetime import datetime, timezone


def list_shared_files(profile: str) -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(here, f"shared_portfolio_{profile}_*.json")
    files = sorted(glob.glob(pattern))
    return files


def parse_ts_from_name(fname: str) -> str | None:
    # expected: shared_portfolio_<profile>_YYYYMMDD_HHMMSS.json
    m = re.search(r"_(\d{8}_\d{6})\.json$", os.path.basename(fname))
    return m.group(1) if m else None


def load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def main():
    profile = os.environ.get("ICHIMOKU_PROFILE", "pipeline_web6")
    here = os.path.dirname(os.path.abspath(__file__))

    files = list_shared_files(profile)
    if not files:
        print("No shared_portfolio files found.")
        return

    best_by_symbol: dict[str, dict] = {}
    all_by_symbol: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for fp in files:
        data = load_json(fp)
        if not isinstance(data, dict):
            continue
        # Gating sécurité global
        md = data.get("max_drawdown")
        try:
            md_val = float(md)
        except Exception:
            md_val = None
        min_eq = data.get("min_equity")
        liquid = int(data.get("liquidations", 0) or 0)
        margin = int(data.get("margin_calls", 0) or 0)
        if (liquid > 0) or (margin > 0) or (isinstance(min_eq, (int, float)) and float(min_eq) < 0.6) or (isinstance(md_val, float) and md_val > 0.6):
            continue
        # Use timezone-aware UTC timestamp for portability
        try:
            ts_dt = datetime.fromtimestamp(os.path.getmtime(fp), timezone.utc)
        except Exception:
            ts_dt = datetime.now(timezone.utc)
        ts_label = parse_ts_from_name(fp) or ts_dt.strftime("%Y%m%d_%H%M%S")
        eq_mult = fmt_float(data.get("equity_mult"))
        equity_eur = eq_mult * 1000.0 if eq_mult == eq_mult else float("nan")
        dd = data.get("max_drawdown")
        try:
            dd_pct = float(dd) * 100.0 if float(dd) <= 5.0 else float(dd)
        except Exception:
            dd_pct = float("nan")
        trades = int(data.get("trades", 0))
        params_map = data.get("best_params") or data.get("params") or {}
        per_symbol = data.get("per_symbol") or {}
        # Exclure snapshots multi-symbole (on ne retient que les runs 1 symbole pour éviter les mélanges)
        try:
            nb_syms = len(params_map.keys()) if isinstance(params_map, dict) else 0
        except Exception:
            nb_syms = 0
        if nb_syms != 1:
            continue

        for sym, pm in params_map.items():
            ps = per_symbol.get(sym, {}) if isinstance(per_symbol, dict) else {}
            pnl_eur = fmt_float(ps.get("pnl_eur")) if isinstance(ps, dict) else float("nan")
            # Gating par symbole (si métriques détaillées existent)
            dd_sym = ps.get("max_dd_indexed")
            try:
                dd_sym_val = float(dd_sym)
            except Exception:
                dd_sym_val = None
            liq_sym = int(ps.get("liquidations", 0) or 0) if isinstance(ps, dict) else 0
            mar_sym = int(ps.get("margin_calls", 0) or 0) if isinstance(ps, dict) else 0
            if (liq_sym > 0) or (mar_sym > 0) or (isinstance(dd_sym_val, float) and dd_sym_val > 0.6):
                continue
            # Build row (skip NaN equity/pnl)
            row = {
                "symbol": sym,
                "pnl_eur": pnl_eur,
                "equity_eur": equity_eur,
                "dd_pct": dd_pct,
                "trades": trades,
                "tenkan": pm.get("tenkan"),
                "kijun": pm.get("kijun"),
                "senkou_b": pm.get("senkou_b"),
                "shift": pm.get("shift"),
                "atr_mult": pm.get("atr_mult"),
                "file": os.path.basename(fp),
                "ts": ts_label,
            }
            if (row["pnl_eur"] == row["pnl_eur"]) and (row["equity_eur"] == row["equity_eur"]):
                rows.append(row)
            else:
                continue
            all_by_symbol.setdefault(sym, []).append(row)

            # Update best per symbol using pnl_eur as primary criterion, tie-breaker equity_eur
            prev = best_by_symbol.get(sym)
            if prev is None:
                best_by_symbol[sym] = row
            else:
                prev_pnl = prev.get("pnl_eur", float("nan"))
                cur_pnl = pnl_eur
                if cur_pnl == cur_pnl and (prev_pnl != prev_pnl or cur_pnl > prev_pnl):
                    best_by_symbol[sym] = row
                elif (cur_pnl == prev_pnl) and (row.get("equity_eur", float("nan")) > prev.get("equity_eur", float("nan"))):
                    best_by_symbol[sym] = row

    # Save CSV
    import csv
    csv_tmp = os.path.join(here, "BEST_PER_SYMBOL.csv.tmp")
    csv_out = os.path.join(here, "BEST_PER_SYMBOL.csv")
    with open(csv_tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","pnl_eur","equity_eur","dd_pct","trades","tenkan","kijun","senkou_b","shift","atr_mult","ts","file"])
        for sym in sorted(best_by_symbol.keys()):
            r = best_by_symbol[sym]
            w.writerow([
                sym,
                f"{fmt_float(r.get('pnl_eur')):.0f}",
                f"{fmt_float(r.get('equity_eur')):.0f}",
                f"{fmt_float(r.get('dd_pct')):.2f}",
                int(r.get("trades", 0)),
                r.get("tenkan"), r.get("kijun"), r.get("senkou_b"), r.get("shift"), r.get("atr_mult"),
                r.get("ts"), r.get("file"),
            ])
    os.replace(csv_tmp, csv_out)

    # Save TXT
    txt_out = os.path.join(here, "BEST_PER_SYMBOL.txt")
    lines = ["Meilleurs réglages par paire (P&L EUR max)"]
    for sym in sorted(best_by_symbol.keys()):
        r = best_by_symbol[sym]
        params = f"Tenkan {r.get('tenkan')}, Kijun {r.get('kijun')}, SenkouB {r.get('senkou_b')}, Shift {r.get('shift')}, ATR× {r.get('atr_mult')}"
        lines.append(
            f"{sym}: P&L {fmt_float(r.get('pnl_eur')):,.0f} € — Equity {fmt_float(r.get('equity_eur')):,.0f} € — DD {fmt_float(r.get('dd_pct')):.2f}% — {params} — {r.get('file')}"
            .replace(",", " ")
        )
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save JSON mapping { symbol: {params..., meta...} }
    best_json_tmp = os.path.join(here, "BEST_PER_SYMBOL.json.tmp")
    best_json_out = os.path.join(here, "BEST_PER_SYMBOL.json")
    out_map = {}
    for sym, r in best_by_symbol.items():
        out_map[sym] = {
            "params": {
                "tenkan": r.get("tenkan"),
                "kijun": r.get("kijun"),
                "senkou_b": r.get("senkou_b"),
                "shift": r.get("shift"),
                "atr_mult": r.get("atr_mult"),
            },
            "pnl_eur": fmt_float(r.get("pnl_eur")),
            "equity_eur": fmt_float(r.get("equity_eur")),
            "dd_pct": fmt_float(r.get("dd_pct")),
            "trades": int(r.get("trades", 0)),
            "file": r.get("file"),
            "ts": r.get("ts"),
        }
    with open(best_json_tmp, "w", encoding="utf-8") as f:
        json.dump(out_map, f, ensure_ascii=False, indent=2)
    os.replace(best_json_tmp, best_json_out)

    # Top-decile (by pnl_eur) then minimal DD per symbol
    def percentile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return float("nan")
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    topdec_map: dict[str, dict] = {}
    for sym, lst in all_by_symbol.items():
        # Collect pnl list
        pnl_vals = [fmt_float(r.get("pnl_eur")) for r in lst if fmt_float(r.get("pnl_eur")) == fmt_float(r.get("pnl_eur"))]
        pnl_vals.sort()
        thr = percentile(pnl_vals, 0.9) if pnl_vals else float("nan")
        # Filter top-decile
        candidates = [r for r in lst if fmt_float(r.get("pnl_eur")) == fmt_float(r.get("pnl_eur")) and fmt_float(r.get("pnl_eur")) >= thr]
        # Pick minimal DD, tie-break by higher equity
        def key_fn(r):
            dd = fmt_float(r.get("dd_pct"))
            eq = fmt_float(r.get("equity_eur"))
            return (dd if dd == dd else float("inf"), -(eq if eq == eq else float("-inf")))
        top = min(candidates, key=key_fn) if candidates else (best_by_symbol.get(sym) or (lst[0] if lst else None))
        if top:
            topdec_map[sym] = top

    # Save decile selection
    csv2_tmp = os.path.join(here, "BEST_PER_SYMBOL_TOP_DECILE_DDMIN.csv.tmp")
    csv2_out = os.path.join(here, "BEST_PER_SYMBOL_TOP_DECILE_DDMIN.csv")
    with open(csv2_tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","pnl_eur","equity_eur","dd_pct","trades","tenkan","kijun","senkou_b","shift","atr_mult","ts","file"])
        for sym in sorted(topdec_map.keys()):
            r = topdec_map[sym]
            w.writerow([
                sym,
                f"{fmt_float(r.get('pnl_eur')):.0f}",
                f"{fmt_float(r.get('equity_eur')):.0f}",
                f"{fmt_float(r.get('dd_pct')):.2f}",
                int(r.get("trades", 0)),
                r.get("tenkan"), r.get("kijun"), r.get("senkou_b"), r.get("shift"), r.get("atr_mult"),
                r.get("ts"), r.get("file"),
            ])
    os.replace(csv2_tmp, csv2_out)

    txt2_out = os.path.join(here, "BEST_PER_SYMBOL_TOP_DECILE_DDMIN.txt")
    lines2 = ["Meilleurs par paire (Top décile P&L → DD minimal)"]
    for sym in sorted(topdec_map.keys()):
        r = topdec_map[sym]
        params = f"Tenkan {r.get('tenkan')}, Kijun {r.get('kijun')}, SenkouB {r.get('senkou_b')}, Shift {r.get('shift')}, ATR× {r.get('atr_mult')}"
        lines2.append(
            f"{sym}: P&L {fmt_float(r.get('pnl_eur')):,.0f} € — Equity {fmt_float(r.get('equity_eur')):,.0f} € — DD {fmt_float(r.get('dd_pct')):.2f}% — {params} — {r.get('file')}"
            .replace(",", " ")
        )
    with open(txt2_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))

    best2_tmp = os.path.join(here, "BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json.tmp")
    best2_out = os.path.join(here, "BEST_PER_SYMBOL_TOP_DECILE_DDMIN.json")
    out_map2 = {}
    for sym, r in topdec_map.items():
        out_map2[sym] = {
            "params": {
                "tenkan": r.get("tenkan"),
                "kijun": r.get("kijun"),
                "senkou_b": r.get("senkou_b"),
                "shift": r.get("shift"),
                "atr_mult": r.get("atr_mult"),
            },
            "pnl_eur": fmt_float(r.get("pnl_eur")),
            "equity_eur": fmt_float(r.get("equity_eur")),
            "dd_pct": fmt_float(r.get("dd_pct")),
            "trades": int(r.get("trades", 0)),
            "file": r.get("file"),
            "ts": r.get("ts"),
        }
    with open(best2_tmp, "w", encoding="utf-8") as f:
        json.dump(out_map2, f, ensure_ascii=False, indent=2)
    os.replace(best2_tmp, best2_out)

    # Save flattened baseline mapping: { symbol: {tenkan,kijun,senkou_b,shift,atr_mult} }
    baseline_tmp = os.path.join(here, "BEST_BASELINE.json.tmp")
    baseline_out = os.path.join(here, "BEST_BASELINE.json")
    flat = {}
    for sym, r in best_by_symbol.items():
        flat[sym] = {
            "tenkan": r.get("tenkan"),
            "kijun": r.get("kijun"),
            "senkou_b": r.get("senkou_b"),
            "shift": r.get("shift"),
            "atr_mult": r.get("atr_mult"),
        }
    with open(baseline_tmp, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    os.replace(baseline_tmp, baseline_out)

    print(csv_out)
    print(txt_out)
    print(best_json_out)
    print(baseline_out)


if __name__ == "__main__":
    main()


