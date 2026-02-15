#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACO (ACOR) optimizer for Ichimoku parameter tuning.

Uses Ant Colony Optimization for Continuous Domains (Socha & Dorigo 2008)
to find robust Ichimoku parameters via walk-forward or single-window backtest.

Examples:
  # Quick run (simple fitness, ~10 min)
  py -3 scripts/production/run_aco_optimize.py --mode simple --n-ants 10 --max-iter 20 --seed 42

  # Robust run (walk-forward fitness, ~1h)
  py -3 scripts/production/run_aco_optimize.py --mode robust --n-ants 20 --max-iter 50 --seed 42

  # With YAML config
  py -3 scripts/production/run_aco_optimize.py --config optimizers/aco_config.yaml

  # Aggressive exploration (noisy landscape)
  py -3 scripts/production/run_aco_optimize.py --mode robust --q 1.0 --archive-size 80 --n-ants 30
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

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe
from optimizers.aco_optimizer import ACOROptimizer, ACORConfig, ICHIMOKU_PARAMS
from optimizers.fitness import FitnessSimple, FitnessRobust


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "aco_run.log", encoding="utf-8"),
        ],
    )


def load_data(use_fused: bool = True) -> pd.DataFrame:
    """Load BTC OHLCV data."""
    if use_fused:
        os.environ["USE_FUSED_H2"] = "1"
    df = pipe._load_local_csv_if_configured("BTC/USDT", "2h")
    if df is None:
        raise RuntimeError("Data not found. Ensure data/BTC_FUSED_2h.csv exists.")
    df = pipe.ensure_utc_index(df)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ACOR optimizer for Ichimoku parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Mode
    ap.add_argument("--mode", choices=["simple", "robust"], default="robust",
                    help="Fitness mode: 'simple' (fast) or 'robust' (walk-forward)")
    # ACOR hyperparameters
    ap.add_argument("--n-ants", type=int, default=20, help="Ants per iteration")
    ap.add_argument("--archive-size", type=int, default=50, help="Archive size (k)")
    ap.add_argument("--q", type=float, default=0.5, help="Exploration parameter (0.1-2.0)")
    ap.add_argument("--xi", type=float, default=1.0, help="Deviation ratio (1.0-2.0, higher=more exploration)")
    ap.add_argument("--max-iter", type=int, default=50, help="Max iterations")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--stagnation-limit", type=int, default=10,
                    help="Iterations without improvement before exploration boost")
    # Fitness tweaks
    ap.add_argument("--lam-dd", type=float, default=0.3, help="Drawdown penalty weight")
    ap.add_argument("--mu-trade", type=float, default=0.5, help="Low-trade penalty")
    ap.add_argument("--min-trades", type=int, default=30, help="Min trades threshold")
    ap.add_argument("--holdout", action="store_true", default=True,
                    help="Hold out last year for final OOS (robust mode)")
    ap.add_argument("--no-holdout", dest="holdout", action="store_false")
    # Data
    ap.add_argument("--use-fused", action="store_true", default=True)
    ap.add_argument("--start-year", type=int, default=None, help="Restrict data start year")
    ap.add_argument("--end-year", type=int, default=None, help="Restrict data end year")
    # Output
    ap.add_argument("--out-dir", default=None, help="Output directory (auto-generated if not set)")
    # Config file
    ap.add_argument("--config", default=None, help="YAML config file (overrides CLI args)")
    # Backtest
    ap.add_argument("--loss-mult", type=float, default=3.0)

    args = ap.parse_args()

    # Build config
    if args.config:
        cfg = ACORConfig.from_yaml(args.config)
    else:
        cfg = ACORConfig(
            n_ants=args.n_ants,
            archive_size=args.archive_size,
            q=args.q,
            xi=args.xi,
            max_iter=args.max_iter,
            seed=args.seed,
            stagnation_limit=args.stagnation_limit,
        )

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ROOT / "outputs" / f"aco_{args.mode}_seed_{cfg.seed}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    logger = logging.getLogger("aco_runner")
    logger.info("=" * 60)
    logger.info("ACOR Optimizer — mode=%s, seed=%d", args.mode, cfg.seed)
    logger.info("Config: %s", json.dumps(cfg.to_dict(), indent=2))
    logger.info("Output: %s", out_dir)
    logger.info("=" * 60)

    # Load data
    logger.info("Loading BTC data...")
    df = load_data(use_fused=args.use_fused)
    if args.start_year:
        df = df[df.index.year >= args.start_year]
    if args.end_year:
        df = df[df.index.year <= args.end_year]
    logger.info("Data loaded: %d rows, %s to %s", len(df), df.index[0], df.index[-1])

    # Build fitness function
    if args.mode == "simple":
        fitness = FitnessSimple(
            lam_dd=args.lam_dd,
            mu_trade=args.mu_trade,
            min_trades=args.min_trades,
        )
    else:
        fitness = FitnessRobust(
            lam_dd=args.lam_dd,
            mu_trade=args.mu_trade,
            min_trades_per_year=args.min_trades,
            holdout_last_year=args.holdout,
        )

    # Build optimizer
    optimizer = ACOROptimizer(cfg)

    # Progress callback
    def on_iter(it: int, best: Any, history: list) -> None:
        # Save incremental progress
        progress = {
            "iteration": it,
            "max_iter": cfg.max_iter,
            "percent": round(100.0 * it / cfg.max_iter, 1),
            "best_score": float(best.score),
            "best_params": best.decoded,
        }
        try:
            tmp = str(out_dir / "PROGRESS.json") + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False)
            os.replace(tmp, out_dir / "PROGRESS.json")
        except Exception:
            pass

    # Run optimization
    t0 = time.time()
    best = optimizer.optimize(
        fitness_fn=fitness,
        df=df,
        backtest_fn=pipe.backtest_long_short,
        log_dir=out_dir,
        callback=on_iter,
        symbol="BTC/USDT",
        timeframe="2h",
        loss_mult=args.loss_mult,
    )
    elapsed = time.time() - t0

    # Final holdout evaluation (if robust mode with holdout)
    holdout_result = None
    if args.mode == "robust" and args.holdout:
        years = sorted(df.index.year.unique())
        if len(years) >= 3:
            last_year = years[-1]
            holdout_df = df[df.index.year == last_year]
            if not holdout_df.empty:
                logger.info("Evaluating on holdout year %d (%d rows)...", last_year, len(holdout_df))
                from optimizers.fitness import run_backtest
                holdout_metrics = run_backtest(
                    holdout_df, best.decoded, pipe.backtest_long_short,
                    symbol="BTC/USDT", timeframe="2h", loss_mult=args.loss_mult,
                )
                holdout_result = {
                    "year": int(last_year),
                    "sharpe": float(holdout_metrics.get("sharpe_proxy", 0.0)),
                    "cagr": float(holdout_metrics.get("CAGR", 0.0)),
                    "max_dd": float(holdout_metrics.get("max_drawdown", 0.0)),
                    "equity_mult": float(holdout_metrics.get("equity_mult", 0.0)),
                    "trades": int(holdout_metrics.get("trades", 0)),
                    "calmar": float(holdout_metrics.get("calmar_ratio", 0.0)),
                }
                logger.info("Holdout %d: Sharpe=%.2f  CAGR=%.2f%%  MDD=%.2f%%  Equity=%.3fx  Trades=%d",
                            last_year, holdout_result["sharpe"],
                            holdout_result["cagr"] * 100, holdout_result["max_dd"] * 100,
                            holdout_result["equity_mult"], holdout_result["trades"])

    # Save final summary
    summary = {
        "mode": args.mode,
        "config": cfg.to_dict(),
        "best": {
            "score": float(best.score),
            "params": best.decoded,
        },
        "top_10": optimizer.get_top_k(10),
        "holdout": holdout_result,
        "cache_stats": {
            "size": len(fitness.cache),
            "hits": fitness.cache.hits,
            "misses": fitness.cache.misses,
        },
        "elapsed_seconds": round(elapsed, 1),
        "total_evals": cfg.archive_size + cfg.max_iter * cfg.n_ants,
    }
    summary_path = out_dir / "aco_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final results
    print("\n" + "=" * 60)
    print("ACOR OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Mode:     {args.mode}")
    print(f"Evals:    {summary['total_evals']}")
    print(f"Time:     {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Score:    {best.score:.4f}")
    print(f"Params:   {json.dumps(best.decoded)}")
    if holdout_result:
        print(f"\nHoldout {holdout_result['year']}:")
        print(f"  Sharpe:  {holdout_result['sharpe']:.2f}")
        print(f"  CAGR:    {holdout_result['cagr']*100:.1f}%")
        print(f"  MDD:     {holdout_result['max_dd']*100:.1f}%")
        print(f"  Equity:  {holdout_result['equity_mult']:.3f}x")
        print(f"  Trades:  {holdout_result['trades']}")
    print(f"\nResults:  {out_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
