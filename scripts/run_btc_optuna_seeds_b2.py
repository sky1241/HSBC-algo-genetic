#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore


def main() -> int:
    profile = "pipeline_web6"
    try:
        pipe.PROFILES[profile]["symbols"] = ["BTC/USDT"]
    except Exception:
        pass

    seeds = [647, 653, 677, 683, 701, 719, 733, 751, 761, 773,
             787, 809, 823, 839, 857, 863, 881, 887, 907, 919,
             941, 953, 967, 983, 997, 1013, 1031, 1051, 1069, 1091]

    out_dir = "outputs"
    trials = 300
    print(f"Running BTC/USDT only (batch 2): {len(seeds)} seeds x {trials} trials …", flush=True)
    for sd in seeds:
        print(f"\n=== Seed {sd} ===", flush=True)
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
    print("\nAll seeds completed (batch 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


