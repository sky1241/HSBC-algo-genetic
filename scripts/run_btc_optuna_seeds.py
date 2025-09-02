#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Optuna optimization for Ichimoku pipeline on BTC/USDT only,
looping over multiple seeds to assess stability.

Usage:
  .\.venv\Scripts\python.exe scripts\run_btc_optuna_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import pipeline module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore


def main() -> int:
    profile = "pipeline_web6"
    # Limit to BTC only
    try:
        pipe.PROFILES[profile]["symbols"] = ["BTC/USDT"]
    except Exception:
        pass

    # 30 seeds between 123 and 999 (distinct, reproducible)
    seeds = [123, 137, 149, 173, 181, 199, 223, 241, 263, 271,
             293, 311, 331, 349, 367, 383, 401, 419, 439, 457,
             479, 491, 509, 521, 541, 557, 577, 593, 613, 631]

    out_dir = "outputs"
    trials = 300

    print(f"Running BTC/USDT only: {len(seeds)} seeds x {trials} trials …", flush=True)
    for sd in seeds:
        print(f"\n=== Seed {sd} ===", flush=True)
        # Per-symbol optimization (Optuna TPE seeded) for BTC/USDT only
        pipe.optuna_optimize_profile_per_symbol(
            profile_name=profile,
            n_trials=trials,
            seed=int(sd),
            out_dir=out_dir,
            use_cache=True,
            jobs=1,
            fast_ratio=1.0,
            baseline_map=None,
            loss_mult=3.0,
        )
    print("\nAll seeds completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


