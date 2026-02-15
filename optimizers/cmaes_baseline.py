"""CMA-ES baseline optimizer for comparison with ACOR.

Uses the `cmaes` library (lightweight, pip install cmaes).
CMA-ES is the gold standard for noisy continuous optimization in low-D (Hansen 2005).
Recommended as primary benchmark by the deep research analysis (2026-02-13).

Reference:
  Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .aco_optimizer import ICHIMOKU_PARAMS, ParamDef, decode_params

logger = logging.getLogger(__name__)

try:
    from cmaes import CMA
except ImportError:
    CMA = None


@dataclass
class CMAESConfig:
    """CMA-ES configuration for Ichimoku parameter tuning.

    population_size: lambda = 4 + floor(3*ln(D)) ≈ 9 for D=5. We use 20 for robustness.
    sigma0:          initial step size (fraction of domain range). 0.3 = moderate.
    max_evals:       total function evaluations budget.
    seed:            random seed.
    param_defs:      parameter space (same as ACOR).
    """
    population_size: int = 20
    sigma0: float = 0.3
    max_evals: int = 1050
    seed: int = 42
    param_defs: List[ParamDef] = field(default_factory=lambda: list(ICHIMOKU_PARAMS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "sigma0": self.sigma0,
            "max_evals": self.max_evals,
            "seed": self.seed,
            "param_defs": [
                {"name": p.name, "low": p.low, "high": p.high, "dtype": p.dtype, "step": p.step}
                for p in self.param_defs
            ],
        }


@dataclass
class CMAESResult:
    params: Dict[str, Any]   # decoded Ichimoku params
    score: float
    metrics: Dict[str, Any]
    n_evals: int
    history: List[Dict[str, Any]]


def run_cmaes(
    config: CMAESConfig,
    fitness_fn: Callable[..., Tuple[float, Dict[str, Any]]],
    df: pd.DataFrame,
    backtest_fn: Any,
    log_dir: Optional[Path] = None,
    **fitness_kw: Any,
) -> CMAESResult:
    """Run CMA-ES optimization as baseline benchmark.

    Parameters match ACOROptimizer.optimize() for fair comparison.
    """
    if CMA is None:
        raise RuntimeError("cmaes not installed. pip install cmaes")

    t0 = time.time()
    n_dims = len(config.param_defs)

    # Initial mean: center of each param range
    mean0 = np.array([
        (p.low + p.high) / 2.0 for p in config.param_defs
    ])

    # Bounds for CMA-ES
    bounds = np.array([
        [p.low, p.high] for p in config.param_defs
    ])

    rng = np.random.default_rng(config.seed)

    optimizer = CMA(
        mean=mean0,
        sigma=config.sigma0 * np.mean(bounds[:, 1] - bounds[:, 0]),
        bounds=bounds,
        seed=int(rng.integers(0, 2**31)),
        population_size=config.population_size,
    )

    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    n_evals = 0
    generation = 0

    while n_evals < config.max_evals:
        solutions = []
        for _ in range(optimizer.population_size):
            if n_evals >= config.max_evals:
                break
            x = optimizer.ask()

            # Clip and cast
            raw = {}
            for d, pdef in enumerate(config.param_defs):
                x[d] = pdef.clip_and_cast(x[d])
                raw[pdef.name] = x[d]

            decoded = decode_params(raw)
            score, metrics = fitness_fn(decoded, df, backtest_fn, **fitness_kw)
            n_evals += 1

            # CMA-ES minimizes, so negate score
            solutions.append((x, -score))

            if score > best_score:
                best_score = score
                best_params = decoded
                best_metrics = metrics

        if solutions:
            optimizer.tell(solutions)

        generation += 1
        history.append({
            "generation": generation,
            "best_score": float(best_score),
            "n_evals": n_evals,
            "elapsed_s": round(time.time() - t0, 1),
        })

        logger.info(
            "CMA-ES gen %d: best=%.4f, evals=%d/%d, params=%s",
            generation, best_score, n_evals, config.max_evals,
            json.dumps(best_params),
        )

    elapsed = time.time() - t0
    logger.info(
        "CMA-ES complete: %d gens, %d evals, %.1fs. Best=%.4f",
        generation, n_evals, elapsed, best_score,
    )

    # Save results
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "cmaes_best.json", "w", encoding="utf-8") as f:
            json.dump({
                "score": float(best_score),
                "params": best_params,
                "config": config.to_dict(),
                "n_evals": n_evals,
                "elapsed_s": round(elapsed, 1),
            }, f, indent=2, ensure_ascii=False)

        keys = ["generation", "best_score", "n_evals", "elapsed_s"]
        with open(log_dir / "cmaes_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

    return CMAESResult(
        params=best_params,
        score=best_score,
        metrics=best_metrics,
        n_evals=n_evals,
        history=history,
    )
