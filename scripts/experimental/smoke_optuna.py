from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore


def main() -> int:
    try:
        pipe.optuna_optimize_profile_per_symbol(
            profile_name="pipeline_web6",
            n_trials=3,
            seed=123,
            out_dir="outputs",
            use_cache=True,
            jobs=1,
            fast_ratio=0.1,
            baseline_map=None,
            loss_mult=3.0,
        )
        print("Smoke OK")
        return 0
    except Exception as e:
        print(f"Smoke ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


