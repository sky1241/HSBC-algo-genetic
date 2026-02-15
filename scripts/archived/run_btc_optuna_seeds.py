#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Optuna optimization for Ichimoku pipeline on BTC/USDT only,
looping over multiple seeds to assess stability.

Usage (defaults: 30 seeds, 300 trials, no time limit):
  .\.venv\Scripts\python.exe scripts\run_btc_optuna_seeds.py

Segmented usage examples (6h slices and seed batching):
  # Segment 1: first 10 seeds, hard cap at 6 hours, custom out dir
  .\.venv\Scripts\python.exe scripts\run_btc_optuna_seeds.py \
      --seed-offset 0 --seed-limit 10 --max-hours 6 \
      --out-dir outputs\optuna_b1_segment1 --trials 300

  # Segment 2: next 10 seeds
  .\.venv\Scripts\python.exe scripts\run_btc_optuna_seeds.py \
      --seed-offset 10 --seed-limit 10 --max-hours 6 \
      --out-dir outputs\optuna_b1_segment2 --trials 300

  # Segment 3: last 10 seeds
  .\.venv\Scripts\python.exe scripts\run_btc_optuna_seeds.py \
      --seed-offset 20 --seed-limit 10 --max-hours 6 \
      --out-dir outputs\optuna_b1_segment3 --trials 300
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import os
import time
from typing import List

# Import pipeline module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna on BTC/USDT only with seed batching and optional time cap")
    parser.add_argument("--profile", default="pipeline_web6", help="Profile name in pipeline (default: pipeline_web6)")
    parser.add_argument("--trials", type=int, default=300, help="Trials per seed (default: 300)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory root (default: outputs)")
    parser.add_argument("--seed-offset", type=int, default=0, help="Start index in the seed list (default: 0)")
    parser.add_argument("--seed-limit", type=int, default=None, help="Number of seeds to run from offset (default: all)")
    parser.add_argument("--seed-list", default="", help="Comma-separated seed list to override defaults (e.g., '1001,2002,3003')")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop after approx N hours (checked between seeds)")
    parser.add_argument("--loss-mult", type=float, default=3.0, help="Loss multiplier for risk (default: 3.0)")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs within a seed (default: 1)")
    parser.add_argument("--use-fused", action="store_true", help="Force using fused BTC 2h dataset (sets USE_FUSED_H2=1)")
    return parser.parse_args()


def _default_seeds() -> List[int]:
    return [
        123, 137, 149, 173, 181, 199, 223, 241, 263, 271,
        293, 311, 331, 349, 367, 383, 401, 419, 439, 457,
        479, 491, 509, 521, 541, 557, 577, 593, 613, 631,
    ]


def main() -> int:
    args = _parse_args()

    profile = args.profile
    # Limit to BTC only
    try:
        pipe.PROFILES[profile]["symbols"] = ["BTC/USDT"]
    except Exception:
        pass

    # Seed list
    if args.seed_list.strip():
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip()]
    else:
        seeds = _default_seeds()

    # Slice seeds by offset/limit
    start = max(0, int(args.seed_offset))
    if args.seed_limit is None:
        end = len(seeds)
    else:
        end = min(len(seeds), start + int(args.seed_limit))
    seeds = seeds[start:end]

    # Time budget
    deadline = None
    if args.max_hours is not None and args.max_hours > 0:
        deadline = time.monotonic() + float(args.max_hours) * 3600.0

    out_dir = str(args.out_dir)

    # Make live dir explicit so each segment can be isolated
    os.environ["ICHIMOKU_LIVE_DIR"] = out_dir

    # Force fused dataset if requested
    if bool(getattr(args, "use_fused", False)):
        os.environ["USE_FUSED_H2"] = "1"

    print(
        f"Running BTC/USDT only: {len(seeds)} seeds x {args.trials} trials — profile={profile} out_dir={out_dir}",
        flush=True,
    )
    print(
        f"USE_FUSED_H2={os.environ.get('USE_FUSED_H2','<unset>')}  ICHIMOKU_LIVE_DIR={os.environ.get('ICHIMOKU_LIVE_DIR')}",
        flush=True,
    )

    for sd in seeds:
        if deadline is not None and time.monotonic() >= deadline:
            print("Time budget reached — stopping before starting next seed.", flush=True)
            break

        print(f"\n=== Seed {sd} ===", flush=True)
        try:
            pipe.optuna_optimize_profile_per_symbol(
                profile_name=profile,
                n_trials=int(args.trials),
                seed=int(sd),
                out_dir=out_dir,
                use_cache=True,
                jobs=int(args.jobs),
                fast_ratio=1.0,
                baseline_map=None,
                loss_mult=float(args.loss_mult),
            )
        except KeyboardInterrupt:
            print("Interrupted by user — exiting gracefully.", flush=True)
            return 130
        except Exception as exc:  # Avoid killing the whole batch on one failure
            print(f"Seed {sd} failed with error: {exc}", flush=True)

    print("\nBatch completed (either all seeds ran or time budget hit).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


