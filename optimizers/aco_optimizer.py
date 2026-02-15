"""ACOR — Ant Colony Optimization for Continuous Domains.

Implementation of Socha & Dorigo (2008) adapted for Ichimoku parameter tuning.

Why ACOR over other ACO variants:
- 5D quasi-continuous space (integers with large ranges + floats) → ACOR native
- No combinatorial graph structure → ACS/MMAS inappropriate
- Gaussian mixture provides natural smoothing on noisy backtest landscape
- Low dimension → archive-based approach is efficient

Reference:
  Socha, K. & Dorigo, M. (2008). Ant colony optimization for continuous domains.
  European Journal of Operational Research, 185(3), 1155-1173.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------
@dataclass
class ParamDef:
    """Single parameter definition."""
    name: str
    low: float
    high: float
    dtype: str = "float"   # "int" or "float"
    step: Optional[float] = None  # discretization step (None = continuous)

    def clip_and_cast(self, value: float) -> Any:
        v = np.clip(value, self.low, self.high)
        if self.step is not None:
            v = self.low + round((v - self.low) / self.step) * self.step
            v = np.clip(v, self.low, self.high)
        if self.dtype == "int":
            return int(round(v))
        return float(round(v, 6))


# Default Ichimoku search space (matches Optuna encoding)
ICHIMOKU_PARAMS: List[ParamDef] = [
    ParamDef("tenkan",   5,  30, "int"),
    ParamDef("r_kijun",  1,   5, "int"),
    ParamDef("r_senkou", 1,   9, "int"),
    ParamDef("shift",   20,  30, "int"),
    ParamDef("atr_mult", 0.5, 6.0, "float", step=0.1),
]


def decode_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ratio-encoded params to actual Ichimoku values.

    Handles constraint: kijun >= tenkan, senkou_b >= kijun.
    """
    tenkan = int(raw["tenkan"])
    r_kijun = int(raw["r_kijun"])
    r_senkou = int(raw["r_senkou"])
    kijun = max(tenkan, r_kijun * tenkan)
    senkou_b = max(kijun, r_senkou * tenkan)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_b": senkou_b,
        "shift": int(raw["shift"]),
        "atr_mult": float(raw["atr_mult"]),
    }


# ---------------------------------------------------------------------------
# ACOR Configuration
# ---------------------------------------------------------------------------
@dataclass
class ACORConfig:
    """ACOR hyperparameters.

    n_ants:       number of new solutions per iteration
    archive_size: k — number of solutions kept in the archive
    q:            locality of search (small → exploit, large → explore)
                  Default 0.5 = balanced. Increase to 1.0+ for very noisy landscapes.
    xi:           deviation ratio — scales Gaussian kernel width.
                  Larger xi → more exploration. Literature recommends 1.0-2.0 for
                  noisy landscapes (Socha & Dorigo 2008, confirmed by 2024 benchmarks).
                  Default 1.0. Use 0.85 for faster convergence, 2.0 for high noise.
    max_iter:     maximum iterations (total evals = n_ants × max_iter)
    seed:         random seed for reproducibility
    stagnation_limit: iterations without improvement before exploration boost
    param_defs:   parameter space definition
    n_workers:    parallel backtest workers (1 = sequential)
    """
    n_ants: int = 20
    archive_size: int = 50
    q: float = 0.5
    xi: float = 1.0
    max_iter: int = 50
    seed: int = 42
    stagnation_limit: int = 10
    param_defs: List[ParamDef] = field(default_factory=lambda: list(ICHIMOKU_PARAMS))
    n_workers: int = 1

    @property
    def total_evals(self) -> int:
        return self.n_ants * self.max_iter

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ACORConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        param_defs = d.pop("param_defs", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if param_defs:
            cfg.param_defs = [ParamDef(**p) for p in param_defs]
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ants": self.n_ants,
            "archive_size": self.archive_size,
            "q": self.q,
            "xi": self.xi,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "stagnation_limit": self.stagnation_limit,
            "n_workers": self.n_workers,
            "total_evals": self.total_evals,
            "param_defs": [
                {"name": p.name, "low": p.low, "high": p.high,
                 "dtype": p.dtype, "step": p.step}
                for p in self.param_defs
            ],
        }


# ---------------------------------------------------------------------------
# Archive entry
# ---------------------------------------------------------------------------
@dataclass
class ArchiveEntry:
    params: np.ndarray        # raw parameter vector (len = n_dims)
    score: float
    metrics: Dict[str, Any]
    decoded: Dict[str, Any]   # decoded Ichimoku params


# ---------------------------------------------------------------------------
# ACOR Optimizer
# ---------------------------------------------------------------------------
class ACOROptimizer:
    """ACOR optimizer for Ichimoku parameters.

    Algorithm outline (per iteration):
      1. Compute weights w_l = (1 / (q*k*sqrt(2*pi))) * exp(-(l-1)^2 / (2*q^2*k^2))
         where l = rank of solution (1=best), k = archive_size
      2. Normalize weights to probabilities p_l = w_l / sum(w)
      3. For each ant:
         a. Select guide solution l ~ Categorical(p)
         b. For each dimension i:
            sigma_i = xi * sum_{r=1}^{k} |S_r[i] - S_l[i]| / (k - 1)
            sample x_i ~ N(S_l[i], sigma_i)
         c. Clip to bounds, cast types
      4. Evaluate all ants
      5. Merge new solutions into archive, keep top-k
      6. Check stagnation → boost exploration if needed
    """

    def __init__(self, config: ACORConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.n_dims = len(config.param_defs)
        self.archive: List[ArchiveEntry] = []
        self.best: Optional[ArchiveEntry] = None
        self.history: List[Dict[str, Any]] = []  # per-iteration stats
        self._stagnation_count = 0
        self._iter = 0

    # ----- archive management -----

    def _sort_archive(self) -> None:
        self.archive.sort(key=lambda e: e.score, reverse=True)

    def _compute_weights(self) -> np.ndarray:
        """Gaussian kernel weights based on rank (Socha & Dorigo eq. 4)."""
        k = len(self.archive)
        if k == 0:
            return np.array([])
        ranks = np.arange(1, k + 1, dtype=float)
        q = self.cfg.q
        w = np.exp(-((ranks - 1) ** 2) / (2.0 * (q * k) ** 2))
        return w / w.sum()

    def _compute_sigma(self, guide_idx: int, dim: int) -> float:
        """Standard deviation for sampling dimension `dim` around guide solution."""
        k = len(self.archive)
        if k <= 1:
            pdef = self.cfg.param_defs[dim]
            return (pdef.high - pdef.low) / 3.0
        guide_val = self.archive[guide_idx].params[dim]
        diffs = sum(abs(self.archive[r].params[dim] - guide_val) for r in range(k))
        sigma = self.cfg.xi * diffs / (k - 1)
        # Floor: don't let sigma collapse to zero
        pdef = self.cfg.param_defs[dim]
        min_sigma = (pdef.high - pdef.low) * 0.01
        return max(sigma, min_sigma)

    # ----- solution generation -----

    def _generate_ant(self, weights: np.ndarray) -> np.ndarray:
        """Generate one new solution vector."""
        guide_idx = self.rng.choice(len(self.archive), p=weights)
        vec = np.empty(self.n_dims)
        for d in range(self.n_dims):
            mu = self.archive[guide_idx].params[d]
            sigma = self._compute_sigma(guide_idx, d)
            vec[d] = self.rng.normal(mu, sigma)
        return vec

    def _clip_and_cast(self, vec: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clip vector to bounds and build raw param dict."""
        raw = {}
        for d, pdef in enumerate(self.cfg.param_defs):
            vec[d] = pdef.clip_and_cast(vec[d])
            raw[pdef.name] = vec[d]
        return vec, raw

    def _random_solution(self) -> np.ndarray:
        """Uniform random solution for archive initialization."""
        vec = np.empty(self.n_dims)
        for d, pdef in enumerate(self.cfg.param_defs):
            if pdef.dtype == "int":
                vec[d] = float(self.rng.integers(int(pdef.low), int(pdef.high) + 1))
            else:
                vec[d] = self.rng.uniform(pdef.low, pdef.high)
            vec[d] = pdef.clip_and_cast(vec[d])
        return vec

    # ----- main optimization loop -----

    def optimize(
        self,
        fitness_fn: Callable[..., Tuple[float, Dict[str, Any]]],
        df: pd.DataFrame,
        backtest_fn: Any,
        log_dir: Optional[Path] = None,
        callback: Optional[Callable[[int, "ArchiveEntry", List[Dict[str, Any]]], None]] = None,
        **fitness_kw: Any,
    ) -> ArchiveEntry:
        """Run ACOR optimization.

        Parameters
        ----------
        fitness_fn : callable
            Fitness function: (params, df, backtest_fn, **kw) -> (score, metrics)
        df : DataFrame
            OHLCV data
        backtest_fn : callable
            backtest_long_short function from the pipeline
        log_dir : Path, optional
            Directory for CSV/JSON logs
        callback : callable, optional
            Called after each iteration: callback(iter, best_entry, history)
        **fitness_kw :
            Extra kwargs passed to fitness_fn (symbol, timeframe, loss_mult, etc.)

        Returns
        -------
        ArchiveEntry
            Best solution found
        """
        t0 = time.time()

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Initialize archive with random solutions
        logger.info("ACOR: Initializing archive with %d random solutions...", self.cfg.archive_size)
        init_solutions = [self._random_solution() for _ in range(self.cfg.archive_size)]
        for vec in init_solutions:
            vec, raw = self._clip_and_cast(vec)
            decoded = decode_params(raw)
            score, metrics = fitness_fn(decoded, df, backtest_fn, **fitness_kw)
            self.archive.append(ArchiveEntry(
                params=vec.copy(), score=score, metrics=metrics, decoded=decoded,
            ))
        self._sort_archive()
        self.best = self.archive[0]
        logger.info("ACOR: Archive initialized. Best score: %.4f", self.best.score)

        # Phase 2: Iterative improvement
        for it in range(self.cfg.max_iter):
            self._iter = it + 1
            weights = self._compute_weights()

            new_entries: List[ArchiveEntry] = []
            for _ in range(self.cfg.n_ants):
                vec = self._generate_ant(weights)
                vec, raw = self._clip_and_cast(vec)
                decoded = decode_params(raw)
                score, metrics = fitness_fn(decoded, df, backtest_fn, **fitness_kw)
                new_entries.append(ArchiveEntry(
                    params=vec.copy(), score=score, metrics=metrics, decoded=decoded,
                ))

            # Merge new solutions into archive
            self.archive.extend(new_entries)
            self._sort_archive()
            self.archive = self.archive[:self.cfg.archive_size]

            # Track improvement
            prev_best = self.best.score if self.best else -np.inf
            self.best = self.archive[0]
            improved = self.best.score > prev_best + 1e-8

            if improved:
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1

            # Stagnation detection → boost exploration
            if self._stagnation_count >= self.cfg.stagnation_limit:
                self._boost_exploration()

            # Logging
            scores = [e.score for e in self.archive]
            iter_stats = {
                "iteration": self._iter,
                "best_score": float(self.best.score),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "best_params": self.best.decoded,
                "stagnation": self._stagnation_count,
                "evals_total": self.cfg.archive_size + self._iter * self.cfg.n_ants,
                "elapsed_s": round(time.time() - t0, 1),
            }
            self.history.append(iter_stats)
            logger.info(
                "ACOR iter %d/%d: best=%.4f  mean=%.4f  stag=%d  params=%s",
                self._iter, self.cfg.max_iter, self.best.score,
                np.mean(scores), self._stagnation_count,
                json.dumps(self.best.decoded),
            )

            if callback:
                callback(self._iter, self.best, self.history)

        elapsed = time.time() - t0
        logger.info(
            "ACOR complete: %d iterations, %d evals, %.1fs. Best=%.4f params=%s",
            self.cfg.max_iter,
            self.cfg.archive_size + self.cfg.max_iter * self.cfg.n_ants,
            elapsed,
            self.best.score,
            json.dumps(self.best.decoded),
        )

        # Save results
        if log_dir:
            self._save_results(log_dir)

        return self.best

    def _boost_exploration(self) -> None:
        """Inject random solutions to escape local optima."""
        n_replace = max(1, self.cfg.archive_size // 5)
        logger.info("ACOR: Stagnation detected (%d iters). Replacing %d archive entries.",
                     self._stagnation_count, n_replace)
        for i in range(n_replace):
            idx = len(self.archive) - 1 - i
            if idx < 0:
                break
            self.archive[idx].params = self._random_solution()
            # Score will be updated next iteration when merged
        self._stagnation_count = 0

    # ----- results export -----

    def _save_results(self, log_dir: Path) -> None:
        """Save optimization results to disk."""
        # Best solution
        best_path = log_dir / "aco_best.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({
                "score": float(self.best.score),
                "params": self.best.decoded,
                "config": self.cfg.to_dict(),
            }, f, indent=2, ensure_ascii=False)

        # Top-K solutions
        topk_path = log_dir / "aco_top_k.json"
        topk = []
        for i, entry in enumerate(self.archive[:min(10, len(self.archive))]):
            topk.append({
                "rank": i + 1,
                "score": float(entry.score),
                "params": entry.decoded,
            })
        with open(topk_path, "w", encoding="utf-8") as f:
            json.dump(topk, f, indent=2, ensure_ascii=False)

        # Iteration history CSV
        hist_path = log_dir / "aco_history.csv"
        if self.history:
            keys = ["iteration", "best_score", "mean_score", "std_score",
                     "stagnation", "evals_total", "elapsed_s"]
            with open(hist_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.history)

        # Full archive dump
        archive_path = log_dir / "aco_archive.json"
        archive_data = [
            {"rank": i + 1, "score": float(e.score), "params": e.decoded}
            for i, e in enumerate(self.archive)
        ]
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(archive_data, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", log_dir)

    def get_top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k solutions as list of dicts."""
        return [
            {"rank": i + 1, "score": float(e.score), "params": e.decoded}
            for i, e in enumerate(self.archive[:k])
        ]
