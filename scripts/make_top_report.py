from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from datetime import datetime, timezone


def load_top_results(path: Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def load_best_per_symbol(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    except Exception:
        return []
    return rows


def fmt_num(x, nd=2) -> str:
    try:
        v = float(x)
        if abs(v) >= 1000 and nd >= 0:
            return f"{v:,.0f}".replace(",", " ")
        return f"{v:.{nd}f}"
    except Exception:
        return "nan"


def build_top5_table(items: list[dict]) -> str:
    # Filter outliers (e.g., equity from multi-symbol or different scope)
    items = [it for it in items if float(it.get("equity_eur", 0)) < 5000]
    head = "| # | file | equity (€) | DD (%) | trades | tenkan | kijun | senkou_b | shift | ATR× |\n|---:|:-----|----------:|------:|------:|------:|-----:|---------:|-----:|-----:|"
    lines = [head]
    for idx, it in enumerate(items[:5], start=1):
        params_map = it.get("params") or {}
        # Prefer BTC/USDT if present, else first symbol
        sym = "BTC/USDT" if "BTC/USDT" in params_map else (next(iter(params_map.keys()), None))
        p = params_map.get(sym, {}) if isinstance(params_map, dict) else {}
        row = "| {idx} | {file} | {eq} | {dd} | {tr} | {tenkan} | {kijun} | {sb} | {shift} | {atr} |".format(
            idx=idx,
            file=os.path.basename(str(it.get("file", ""))),
            eq=fmt_num(it.get("equity_eur"), 0),
            dd=fmt_num(it.get("dd_pct"), 2),
            tr=int(it.get("trades", 0)),
            tenkan=p.get("tenkan", ""),
            kijun=p.get("kijun", ""),
            sb=p.get("senkou_b", ""),
            shift=p.get("shift", ""),
            atr=fmt_num(p.get("atr_mult", ""), 2),
        )
        lines.append(row)
    return "\n".join(lines)


def build_best_table(rows: list[dict]) -> str:
    head = "| symbol | equity (€) | DD (%) | trades | tenkan | kijun | senkou_b | shift | ATR× | ts | file |\n|:------|----------:|------:|------:|------:|-----:|---------:|-----:|-----:|:-------|:-----|"
    out = [head]
    for r in rows:
        out.append(
            "| {sym} | {eq} | {dd} | {tr} | {tnk} | {kj} | {sb} | {sh} | {atr} | {ts} | {file} |".format(
                sym=r.get("symbol", ""),
                eq=fmt_num(r.get("equity_eur"), 0),
                dd=fmt_num(r.get("dd_pct"), 2),
                tr=r.get("trades", ""),
                tnk=r.get("tenkan", ""),
                kj=r.get("kijun", ""),
                sb=r.get("senkou_b", ""),
                sh=r.get("shift", ""),
                atr=fmt_num(r.get("atr_mult", ""), 2),
                ts=r.get("ts", ""),
                file=r.get("file", ""),
            )
        )
    return "\n".join(out)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    top_json = root / "outputs" / "top_results.json"
    best_csv = root / "outputs" / "BEST_PER_SYMBOL.csv"
    out_md = root / "docs" / "TOP_BEST_RESULTS.md"

    top_items = load_top_results(top_json)
    best_rows = load_best_per_symbol(best_csv)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"### Top‑5 global — pipeline_web6\n\nGénéré: {ts}\n")
    if top_items:
        lines.append(build_top5_table(top_items))
    else:
        lines.append("Aucun résultat trouvé dans outputs/top_results.json.")

    lines.append("\n\n### Best par symbole\n\nSource: `outputs/BEST_PER_SYMBOL.csv`\n")
    if best_rows:
        lines.append(build_best_table(best_rows))
    else:
        lines.append("Aucune ligne dans BEST_PER_SYMBOL.csv.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


