#!/usr/bin/env python3
"""Mesure l'impact des coûts (fees + funding + slippage) sur un WFA partiel.

Lance ``run_walk_forward`` sur BTC/USDT 1D, deux fois :
    1. coûts désactivés (legacy gross),
    2. coûts activés (B5 realistic).

Sauvegarde les métriques dans ``outputs/wfa_with_costs_2026-04-26/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import io_loader, risk_sizing, stats_eval, wfa  # noqa: E402

OUT_DIR = ROOT / "outputs" / "wfa_with_costs_2026-04-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    data_path = ROOT / "data" / "BTC_USDT_1d.csv"
    df = io_loader.load_ohlcv_csv(data_path)
    df = df.loc["2020-01-01":"2024-12-31"].copy()
    print(f"Data: {len(df)} bars from {df.index.min()} to {df.index.max()}")

    config = wfa.WalkForwardConfig(
        n_states=2,
        n_trials=10,
        welch_nperseg_grid=(64,),
        welch_noverlap=0.5,
        lfp_horizon_days=10.0,
        volatility_window=32,
        min_train_size=200,
        min_train_years=2,
        periods_per_year=365,
        fs_per_day=1.0,
    )

    results: dict[str, dict] = {}

    for label, realistic in [("gross", False), ("net", True)]:
        # Monkey-patch simulate_strategy to flip the realistic flag for the
        # duration of this run.
        original_sim = risk_sizing.simulate_strategy
        original_phase = risk_sizing.run_phase_strategy

        def patched_sim(df_, params_, **kw):
            kw["realistic_costs"] = realistic
            return original_sim(df_, params_, **kw)

        def patched_phase(df_, phases_, params_by_phase_, **kw):
            kw["realistic_costs"] = realistic
            return original_phase(df_, phases_, params_by_phase_, **kw)

        risk_sizing.simulate_strategy = patched_sim
        risk_sizing.run_phase_strategy = patched_phase
        try:
            print(f"\n=== Running WFA [{label}] (realistic_costs={realistic}) ===")
            result = wfa.run_walk_forward(df, seeds=[0, 1, 2], config=config)
            metrics = result.metrics
            global_pa = metrics[
                (metrics["strategy"] == "phaseaware") & (metrics["phase"] == "global")
            ]
            global_bl = metrics[
                (metrics["strategy"] == "baseline") & (metrics["phase"] == "global")
            ]
            sharpe_pa = float(global_pa["sharpe"].median())
            sharpe_bl = float(global_bl["sharpe"].median())
            cagr_pa = float(global_pa["cagr"].median())
            cagr_bl = float(global_bl["cagr"].median())
            mdd_pa = float(global_pa["mdd"].median())
            mdd_bl = float(global_bl["mdd"].median())
            results[label] = {
                "sharpe_phaseaware": sharpe_pa,
                "sharpe_baseline": sharpe_bl,
                "cagr_phaseaware": cagr_pa,
                "cagr_baseline": cagr_bl,
                "mdd_phaseaware": mdd_pa,
                "mdd_baseline": mdd_bl,
                "n_folds": int(len(global_pa)),
            }
            metrics.to_csv(OUT_DIR / f"metrics_{label}.csv", index=False)
            result.returns.to_csv(OUT_DIR / f"returns_{label}.csv", index=False)
        finally:
            risk_sizing.simulate_strategy = original_sim
            risk_sizing.run_phase_strategy = original_phase

    # Δ summary
    delta = {
        "delta_sharpe_phaseaware": results["net"]["sharpe_phaseaware"]
        - results["gross"]["sharpe_phaseaware"],
        "delta_sharpe_baseline": results["net"]["sharpe_baseline"]
        - results["gross"]["sharpe_baseline"],
        "delta_cagr_phaseaware": results["net"]["cagr_phaseaware"]
        - results["gross"]["cagr_phaseaware"],
        "delta_cagr_baseline": results["net"]["cagr_baseline"]
        - results["gross"]["cagr_baseline"],
        "delta_mdd_phaseaware": results["net"]["mdd_phaseaware"]
        - results["gross"]["mdd_phaseaware"],
    }
    summary = {
        "gross": results["gross"],
        "net": results["net"],
        "delta": delta,
        "config": {
            "n_seeds": 3,
            "n_states": 2,
            "n_trials": 10,
            "period": "2020-01 to 2024-12",
            "data": "BTC/USDT 1D",
        },
    }
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("\n=== Δ Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
