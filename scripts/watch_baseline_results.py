#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class BaselineRecord:
    file_path: str
    run_ts: str
    symbol: str
    tenkan: int
    kijun: int
    senkou_b: int
    shift: int
    atr_mult: float
    equity_mult: float | None
    max_drawdown: float | None
    trades: int | None
    sharpe_proxy: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch baseline fixed backtest outputs and build live summaries")
    p.add_argument("--root", default=".", help="Project root")
    p.add_argument("--interval", type=int, default=60, help="Seconds between scans")
    p.add_argument("--out-csv", default="docs/LIVE_BASELINES.csv")
    p.add_argument("--out-json", default="$live/LIVE_BASELINES.json")
    p.add_argument("--out-html", default="$live/LIVE_REPORT.html")
    p.add_argument("--upload-url", default=None, help="Optional URL to upload best snapshot (PUT/POST)")
    p.add_argument("--upload-every", type=int, default=1800, help="Seconds between uploads (default 30 min)")
    return p.parse_args()


def expand_live_path(path: str, root: Path) -> Path:
    # Allow "$live" as alias to repo's $live dir
    if path.startswith("$live/"):
        return (root / "$live" / path.split("$live/", 1)[1]).resolve()
    return (root / path).resolve()


def discover_baseline_jsons(outputs_dir: Path) -> List[Path]:
    result: List[Path] = []
    if not outputs_dir.exists():
        return result
    for sub in outputs_dir.iterdir():
        if not sub.is_dir():
            continue
        # Heuristic: baseline folders often start with "baseline_"
        if not sub.name.lower().startswith("baseline_"):
            continue
        try:
            for fp in sub.glob("BTC_BASELINE_*.json"):
                result.append(fp)
        except Exception:
            continue
    return sorted(result)


def parse_run_ts_from_name(name: str) -> Optional[str]:
    # Expect suffix like _YYYYmmdd_HHMMSS.json
    try:
        stem = name.rsplit(".", 1)[0]
        ts_part = stem.rsplit("_", 1)[-1]
        if len(ts_part) == 15 and ts_part[8] == "_":
            dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        # Fallback: current UTC
        return datetime.now(timezone.utc).isoformat()
    except Exception:
        return None


def load_record(fp: Path) -> Optional[BaselineRecord]:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None
    bp: Dict[str, Dict] = data.get("best_params") or {}
    shared: Dict = data.get("shared_metrics") or {}
    # Assume single symbol, prefer BTC/USDT if present
    sym = "BTC/USDT"
    params = bp.get(sym)
    if params is None and bp:
        # Take first available
        sym, params = next(iter(bp.items()))
    if not params:
        return None
    rec = BaselineRecord(
        file_path=str(fp.resolve()),
        run_ts=parse_run_ts_from_name(fp.name) or datetime.now(timezone.utc).isoformat(),
        symbol=str(sym),
        tenkan=int(params.get("tenkan", 0) or 0),
        kijun=int(params.get("kijun", 0) or 0),
        senkou_b=int(params.get("senkou_b", 0) or 0),
        shift=int(params.get("shift", 0) or 0),
        atr_mult=float(params.get("atr_mult", 0.0) or 0.0),
        equity_mult=(None if shared.get("equity_mult") is None else float(shared.get("equity_mult"))),
        max_drawdown=(None if shared.get("max_drawdown") is None else float(shared.get("max_drawdown"))),
        trades=(None if shared.get("trades") is None else int(shared.get("trades"))),
        sharpe_proxy=(None if shared.get("sharpe_proxy") is None else float(shared.get("sharpe_proxy"))),
    )
    return rec


def build_df(files: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for fp in files:
        rec = load_record(fp)
        if rec is None:
            continue
        rows.append(asdict(rec))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort by run time then by equity_mult desc
    try:
        df["run_ts_dt"] = pd.to_datetime(df["run_ts"], utc=True)
        df.sort_values(["run_ts_dt", "equity_mult"], ascending=[True, False], inplace=True)
        df.drop(columns=["run_ts_dt"], inplace=True)
    except Exception:
        pass
    return df


def write_outputs(root: Path, df: pd.DataFrame, out_csv: Path, out_json: Path, out_html: Path) -> Dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        # Save CSV
        df.to_csv(out_csv, index=False)
        # Build best snapshot (latest per symbol with best equity)
        best: Dict[str, Dict] = {}
        for sym in sorted(df["symbol"].dropna().unique()):
            sub = df[df["symbol"] == sym].copy()
            # Prefer latest timestamp, then highest equity
            try:
                sub["run_ts_dt"] = pd.to_datetime(sub["run_ts"], utc=True)
                sub.sort_values(["run_ts_dt", "equity_mult"], ascending=[False, False], inplace=True)
            except Exception:
                sub.sort_values(["equity_mult"], ascending=[False], inplace=True)
            row = sub.iloc[0].to_dict()
            best[sym] = {
                "params": {
                    "tenkan": int(row.get("tenkan", 0)),
                    "kijun": int(row.get("kijun", 0)),
                    "senkou_b": int(row.get("senkou_b", 0)),
                    "shift": int(row.get("shift", 0)),
                    "atr_mult": float(row.get("atr_mult", 0.0)),
                },
                "metrics": {
                    "equity_mult": (None if pd.isna(row.get("equity_mult")) else float(row.get("equity_mult"))),
                    "max_drawdown": (None if pd.isna(row.get("max_drawdown")) else float(row.get("max_drawdown"))),
                    "trades": (None if pd.isna(row.get("trades")) else int(row.get("trades"))),
                    "sharpe_proxy": (None if pd.isna(row.get("sharpe_proxy")) else float(row.get("sharpe_proxy"))),
                },
                "source_file": row.get("file_path"),
                "run_ts": row.get("run_ts"),
            }
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": int(len(df)),
            "best": best,
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Minimal HTML report (ASCII-safe)
        try:
            cols = [
                "run_ts", "symbol", "tenkan", "kijun", "senkou_b", "shift", "atr_mult",
                "equity_mult", "max_drawdown", "trades", "sharpe_proxy",
            ]
            html_table = df[cols].tail(50).to_html(index=False, border=1)
            html = (
                "<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='60'>"
                "<title>Live Baselines</title></head><body>"
                "<h2>Live Baseline Backtests (last 50)</h2>" + html_table + "</body></html>"
            )
            out_html.write_text(html, encoding="utf-8")
        except Exception:
            pass
    else:
        # Clear files when no data
        out_json.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), "count": 0, "best": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
        out_csv.write_text("", encoding="utf-8")
        out_html.write_text("<html><body><p>No data yet.</p></body></html>", encoding="utf-8")

    return {"csv": str(out_csv), "json": str(out_json), "html": str(out_html)}


def maybe_upload(url: Optional[str], payload_path: Path) -> None:
    if not url:
        return
    try:
        import requests  # type: ignore
    except Exception:
        return
    try:
        data = json.loads(payload_path.read_text(encoding="utf-8"))
        # Prefer PUT, fallback to POST
        try:
            r = requests.put(url, json=data, timeout=15)
        except Exception:
            r = requests.post(url, json=data, timeout=15)
        _ = r.status_code  # ignore
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    outputs_dir = root / "outputs"

    out_csv = expand_live_path(args.out_csv, root)
    out_json = expand_live_path(args.out_json, root)
    out_html = expand_live_path(args.out_html, root)

    last_upload_ts: float = 0.0
    upload_every = max(60, int(args.upload_every))

    last_sig: Optional[str] = None
    while True:
        try:
            files = discover_baseline_jsons(outputs_dir)
            # Build a signature (paths + mtimes + sizes)
            sig_parts: List[str] = []
            for fp in files:
                try:
                    st = fp.stat()
                    sig_parts.append(f"{fp}:{st.st_size}:{int(st.st_mtime)}")
                except FileNotFoundError:
                    sig_parts.append(f"{fp}:0:0")
            sig = "|".join(sig_parts)

            if sig and sig != last_sig:
                df = build_df(files)
                outputs = write_outputs(root, df, out_csv, out_json, out_html)
                last_sig = sig

            # Periodic upload
            now = time.time()
            if args.upload_url and (now - last_upload_ts >= upload_every):
                maybe_upload(args.upload_url, out_json)
                last_upload_ts = now
        except Exception:
            # Never crash the watcher
            pass
        time.sleep(max(5, int(args.interval)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



