#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List
import argparse
from datetime import datetime


def safe_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None


def derive_params(p: Dict[str, Any]):
    t = safe_float(p.get("tenkan"))
    kj = safe_float(p.get("kijun"))
    sb = safe_float(p.get("senkou_b"))
    rk = safe_float(p.get("r_kijun"))
    rs = safe_float(p.get("r_senkou"))
    if kj is None and (t is not None and rk is not None):
        kj = float(rk) * float(t)
    if sb is None and t is not None:
        base = kj if kj is not None else (float(rk) * float(t) if rk is not None else None)
        if (base is not None) and (rs is not None):
            sb = max(float(base), float(rs) * float(t))
    sh = safe_float(p.get("shift"))
    atr = safe_float(p.get("atr_mult"))
    return t, kj, sb, sh, atr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract top-N trials by eq_ret with >=100 trades")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out-csv", default="")
    p.add_argument("--sort", choices=["eq_ret","score"], default="eq_ret")
    p.add_argument("--mdd-max", type=float, default=None)
    p.add_argument("--min-sharpe", type=float, default=None)
    p.add_argument("--min-trades", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for fp in glob.glob("outputs/trial_logs/**/trials_from_wfa.jsonl", recursive=True):
        K = os.path.basename(os.path.dirname(fp))
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    p = rec.get("params") or {}
                    mt = rec.get("metrics_train") or {}
                    eq_ret = mt.get("eq_ret")
                    if eq_ret is None:
                        eqm = mt.get("equity_mult")
                        if isinstance(eqm, (int, float)) and float(eqm) > 0:
                            eq_ret = float(eqm) - 1.0
                    if eq_ret is None:
                        continue
                    trades = mt.get("trades") or 0
                    try:
                        trades = int(trades)
                    except Exception:
                        trades = 0
                    if trades < int(args.min_trades):
                        continue
                    # Metrics for score
                    sharpe = mt.get("sharpe_proxy")
                    try:
                        sharpe = float(sharpe) if sharpe is not None else None
                    except Exception:
                        sharpe = None
                    cagr = mt.get("CAGR")
                    try:
                        cagr = float(cagr) if cagr is not None else None
                    except Exception:
                        cagr = None
                    mdd = mt.get("max_drawdown")
                    try:
                        mdd = float(mdd) if mdd is not None else None
                    except Exception:
                        mdd = None
                    t, kj, sb, sh, atr = derive_params(p)
                    if None in (t, kj, sb, sh, atr):
                        continue
                    # Score = 0.6*Sharpe + 0.3*CAGR - 0.3*MDD (penalty trades<30 not applicable here)
                    score = None
                    if sharpe is not None and cagr is not None and mdd is not None:
                        score = 0.6 * sharpe + 0.3 * cagr - 0.3 * mdd
                    rows.append(
                        dict(
                            K=K,
                            tenkan=int(t),
                            kijun=int(kj),
                            senkou_b=int(sb),
                            shift=int(sh),
                            atr_mult=float(atr),
                            eq_ret=float(eq_ret),
                            trades=trades,
                            sharpe=sharpe,
                            cagr=cagr,
                            mdd=mdd,
                            score=score,
                        )
                    )
        except Exception:
            continue

    if not rows:
        print("No trials with >=100 trades")
        return 0

    # Optional quality filtering
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if (args.mdd_max is not None) and (r.get("mdd") is not None) and (float(r["mdd"]) > float(args.mdd_max)):
            continue
        if (args.min_sharpe is not None) and (r.get("sharpe") is not None) and (float(r["sharpe"]) < float(args.min_sharpe)):
            continue
        filtered.append(r)
    rows = filtered if filtered else rows

    # Sorting
    if args.sort == "score":
        rows.sort(key=lambda r: (float('-inf') if r.get("score") is None else float(r["score"])), reverse=True)
    else:
        rows.sort(key=lambda r: r["eq_ret"], reverse=True)
    topn = max(1, int(args.top))
    top = rows[:topn]
    # optional CSV export
    if args.out_csv is not None:
        out_csv = str(args.out_csv).strip()
        if not out_csv:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_csv = os.path.join("docs", f"TOP_TRIALS_{topn}_{ts}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["K","tenkan","kijun","senkou_b","shift","atr_mult","eq_ret","trades","sharpe","cagr","mdd","score"]) 
            w.writeheader()
            for r in top:
                w.writerow(r)
        print(out_csv)
    for r in top:
        print(
            f"eq_ret={r['eq_ret']*100:.2f}% trades={int(r['trades'])} score={(r['score'] if r['score'] is not None else float('nan')):.3f} | "
            f"K={r['K']} tenkan={r['tenkan']} kijun={r['kijun']} senkou_b={r['senkou_b']} shift={r['shift']} ATR={r['atr_mult']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


