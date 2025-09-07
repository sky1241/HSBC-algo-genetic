import os
import json
import glob
from pathlib import Path


def dd_percent(d: dict) -> float:
    min_eq = d.get("min_equity")
    if isinstance(min_eq, (int, float)) and 0 < float(min_eq) <= 1.5:
        return max(0.0, (1.0 - float(min_eq)) * 100.0)
    md = d.get("max_drawdown")
    try:
        md = float(md)
    except Exception:
        return float("nan")
    return md * 100.0 if (md == md and md <= 5.0) else (md if md == md else float("nan"))


def load_reports(outputs_dir: Path, profile: str):
    pattern = outputs_dir / f"shared_portfolio_{profile}_*.json"
    items = []
    for fp in glob.glob(str(pattern)):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Gating sécurité
            md = data.get("max_drawdown")
            try:
                md = float(md)
            except Exception:
                md = None
            min_eq = data.get("min_equity")
            liquid = int(data.get("liquidations", 0) or 0)
            margin = int(data.get("margin_calls", 0) or 0)
            if (liquid > 0) or (margin > 0) or (isinstance(min_eq, (int, float)) and float(min_eq) < 0.6) or (isinstance(md, float) and md > 0.6):
                continue
            eq_mult = float(data.get("equity_mult", 1.0))
            items.append({
                "file": os.path.basename(fp),
                "equity_eur": eq_mult * 1000.0,
                "dd_pct": dd_percent(data),
                "trades": int(data.get("trades", 0)),
                "params": data.get("best_params") or data.get("params") or {},
            })
        except Exception:
            continue
    items.sort(key=lambda x: x["equity_eur"], reverse=True)
    return items


def main(limit: int = 10) -> Path:
    profile = os.environ.get("ICHIMOKU_PROFILE", "pipeline_web6")
    here = Path(__file__).resolve().parent
    items = load_reports(here, profile)[:limit]
    out_path = here / "top_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    p = main()
    print(p)
