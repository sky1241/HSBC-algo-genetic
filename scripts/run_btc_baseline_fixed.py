#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a clean BTC-only baseline backtest (fixed Ichimoku params) on full fused 2h history.

Examples (PowerShell):
  .\.venv\Scripts\python.exe scripts\run_btc_baseline_fixed.py --use-fused --atr 3 \
      --tenkan 9 --kijun 26 --senkou-b 52 --shift 26 --out-dir outputs\baseline_btc_only
  .\.venv\Scripts\python.exe scripts\run_btc_baseline_fixed.py --use-fused --atr 5 \
      --tenkan 9 --kijun 26 --senkou-b 52 --shift 26 --out-dir outputs\baseline_btc_only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd  # type: ignore

# Import pipeline module
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore


def _load_btc_fused(timeframe: str = "2h") -> pd.DataFrame:
    df = pipe._load_local_csv_if_configured("BTC/USDT", timeframe)
    if df is None:
        raise RuntimeError("Fused CSV not configured. Pass --use-fused or set USE_FUSED_H2=1.")
    return pipe.ensure_utc_index(df)


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC-only baseline backtest on fused 2h data")
    ap.add_argument("--tenkan", type=int, default=9)
    ap.add_argument("--kijun", type=int, default=26)
    ap.add_argument("--senkou-b", dest="senkou_b", type=int, default=52)
    ap.add_argument("--shift", type=int, default=26)
    ap.add_argument("--atr", dest="atr_mult", type=float, default=3.0)
    ap.add_argument("--loss-mult", type=float, default=3.0)
    ap.add_argument("--use-fused", action="store_true", help="Set USE_FUSED_H2=1 to load BTC_FUSED_2h.csv")
    ap.add_argument("--out-dir", default="outputs/baseline_btc_only")
    args = ap.parse_args()

    if args.use_fused:
        os.environ["USE_FUSED_H2"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BTC-only market data
    timeframe = "2h"
    df = _load_btc_fused(timeframe)
    market_data = {"BTC/USDT": df}

    # Fixed params
    params = {
        "tenkan": int(args.tenkan),
        "kijun": int(args.kijun),
        "senkou_b": int(args.senkou_b),
        "shift": int(args.shift),
        "atr_mult": float(args.atr_mult),
    }
    best_by_symbol = {"BTC/USDT": params}

    # Backtest portfolio (single symbol)
    shared = pipe.backtest_shared_portfolio(
        market_data,
        best_by_symbol,
        timeframe=timeframe,
        record_curve=True,
        loss_mult=float(args.loss_mult),
    )

    # Persist
    ts = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    payload = {"best_params": best_by_symbol, "shared_metrics": shared}
    out_path = out_dir / f"BTC_BASELINE_T{params['tenkan']}_K{params['kijun']}_S{params['senkou_b']}_SH{params['shift']}_ATR{params['atr_mult']}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Console summary
    eq = float(shared.get("equity_mult", float("nan")))
    dd = float(shared.get("max_drawdown", float("nan")))
    tr = int(shared.get("trades", 0))
    sh = float(shared.get("sharpe_proxy", float("nan")))
    print(f"BTC baseline {params} => equity× {eq:.3f}, MDD {dd:.2%}, trades {tr}, Sharpe≈{sh:.2f}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



