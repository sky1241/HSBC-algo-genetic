#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head-to-head benchmark: ACOR vs CMA-ES + PBO analysis.

Runs both optimizers with same budget and seed, then computes
Probability of Backtest Overfitting on the combined solution set.

Usage:
  py -3 scripts/production/run_benchmark_aco_vs_cmaes.py --budget 500 --seed 42

  # Quick smoke test
  py -3 scripts/production/run_benchmark_aco_vs_cmaes.py --budget 65 --seed 42 --start-year 2020
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe
from optimizers.aco_optimizer import ACOROptimizer, ACORConfig
from optimizers.cmaes_baseline import run_cmaes, CMAESConfig
from optimizers.fitness import FitnessSimple
from optimizers.cscv_pbo import compute_pbo, build_trials_matrix_from_wfa


def main() -> int:
    ap = argparse.ArgumentParser(description="ACOR vs CMA-ES benchmark + PBO")
    ap.add_argument("--budget", type=int, default=500, help="Total evals per optimizer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / f"benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "benchmark.log")])
    logger = logging.getLogger("benchmark")

    # Load data
    os.environ["USE_FUSED_H2"] = "1"
    df = pipe._load_local_csv_if_configured("BTC/USDT", "2h")
    if df is None:
        raise RuntimeError("Data not found")
    df = pipe.ensure_utc_index(df)
    if args.start_year:
        df = df[df.index.year >= args.start_year]
    if args.end_year:
        df = df[df.index.year <= args.end_year]
    logger.info("Data: %d rows, %s to %s", len(df), df.index[0], df.index[-1])

    fitness = FitnessSimple(min_trades=5)
    kw = {"symbol": "BTC/USDT", "timeframe": "2h"}

    # Compute ACOR params from budget
    archive = min(50, args.budget // 3)
    n_ants = max(5, archive // 3)
    max_iter = max(1, (args.budget - archive) // n_ants)

    # ---- ACOR ----
    logger.info("=" * 50)
    logger.info("Running ACOR (budget=%d)...", args.budget)
    acor_cfg = ACORConfig(n_ants=n_ants, archive_size=archive, max_iter=max_iter, seed=args.seed)
    acor_opt = ACOROptimizer(acor_cfg)
    t0 = time.time()
    acor_best = acor_opt.optimize(fitness, df, pipe.backtest_long_short,
                                   log_dir=out_dir / "acor", **kw)
    acor_time = time.time() - t0
    acor_evals = archive + max_iter * n_ants

    # ---- CMA-ES ----
    logger.info("=" * 50)
    logger.info("Running CMA-ES (budget=%d)...", args.budget)
    cmaes_cfg = CMAESConfig(max_evals=args.budget, seed=args.seed)
    # Fresh fitness (separate cache)
    fitness2 = FitnessSimple(min_trades=5)
    t0 = time.time()
    cmaes_result = run_cmaes(cmaes_cfg, fitness2, df, pipe.backtest_long_short,
                              log_dir=out_dir / "cmaes", **kw)
    cmaes_time = time.time() - t0

    # ---- PBO on combined solutions ----
    logger.info("=" * 50)
    logger.info("Computing PBO on top solutions...")

    # Collect top-20 from each optimizer
    acor_top = acor_opt.get_top_k(20)
    all_solutions = [s["params"] for s in acor_top]
    all_solutions.append(cmaes_result.params)

    if len(all_solutions) >= 3:
        try:
            n_periods = min(52, len(df) // 100)
            if n_periods >= 8:
                trials_matrix = build_trials_matrix_from_wfa(
                    all_solutions, df, pipe.backtest_long_short, n_periods=n_periods, **kw)
                n_splits = min(8, n_periods - (n_periods % 2))
                if n_splits >= 4:
                    pbo, logits, pbo_details = compute_pbo(trials_matrix, n_splits=n_splits)
                else:
                    pbo, pbo_details = None, {"error": "too few splits"}
            else:
                pbo, pbo_details = None, {"error": "too few periods"}
        except Exception as e:
            logger.warning("PBO failed: %s", e)
            pbo, pbo_details = None, {"error": str(e)}
    else:
        pbo, pbo_details = None, {"error": "too few solutions"}

    # ---- Summary ----
    summary = {
        "budget": args.budget,
        "seed": args.seed,
        "data_rows": len(df),
        "acor": {
            "score": float(acor_best.score),
            "params": acor_best.decoded,
            "evals": acor_evals,
            "time_s": round(acor_time, 1),
        },
        "cmaes": {
            "score": float(cmaes_result.score),
            "params": cmaes_result.params,
            "evals": cmaes_result.n_evals,
            "time_s": round(cmaes_time, 1),
        },
        "pbo": pbo_details,
        "winner": "ACOR" if acor_best.score > cmaes_result.score else "CMA-ES",
        "score_diff": round(acor_best.score - cmaes_result.score, 4),
    }

    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Budget:   {args.budget} evals per optimizer")
    print(f"")
    print(f"ACOR:     score={acor_best.score:.4f}  time={acor_time:.0f}s  params={json.dumps(acor_best.decoded)}")
    print(f"CMA-ES:   score={cmaes_result.score:.4f}  time={cmaes_time:.0f}s  params={json.dumps(cmaes_result.params)}")
    print(f"")
    print(f"Winner:   {summary['winner']}  (diff={summary['score_diff']:+.4f})")
    if pbo is not None:
        print(f"PBO:      {pbo:.1%}  ({pbo_details.get('interpretation', '?')})")
    print(f"Results:  {out_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
