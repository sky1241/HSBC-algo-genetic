"""Fitness functions for ACO optimization.

FIT_SIMPLE:  perf_net - lambda*maxDD - mu*turnover  (fast, single-window)
FIT_ROBUSTE: walk-forward aggregated score + penalties  (recommended)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation cache (hash(params) -> metrics dict)
# ---------------------------------------------------------------------------
class EvalCache:
    """In-memory cache keyed on parameter hash to avoid duplicate backtests."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, params: Dict[str, Any]) -> str:
        # Deterministic hash: round floats to avoid floating-point noise
        canonical = {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in sorted(params.items())}
        return hashlib.md5(json.dumps(canonical).encode()).hexdigest()

    def get(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        k = self._key(params)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def put(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self._store[self._key(params)] = metrics

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Backtest wrapper (calls pipe.backtest_long_short)
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, Any],
    backtest_fn: Any,
    symbol: str = "BTC/USDT",
    timeframe: str = "2h",
    loss_mult: float = 3.0,
    confidence_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Thin wrapper around the pipeline backtest function."""
    m = backtest_fn(
        df,
        int(params["tenkan"]),
        int(params["kijun"]),
        int(params["senkou_b"]),
        int(params["shift"]),
        float(params["atr_mult"]),
        loss_mult=loss_mult,
        symbol=symbol,
        timeframe=timeframe,
        tp_mult=float(params["tp_mult"]) if params.get("tp_mult") is not None else None,
        confidence_series=confidence_series,
    )
    return m


# ---------------------------------------------------------------------------
# FIT_SIMPLE
# ---------------------------------------------------------------------------
@dataclass
class FitnessSimple:
    """Fast single-window fitness: sharpe - lambda*maxDD - mu*(1/trades).

    Good for quick iterations.  Includes fees/slippage via the backtest itself.
    """

    lam_dd: float = 0.3        # penalty weight for max drawdown
    mu_trade: float = 0.5      # penalty if < min_trades
    min_trades: int = 30       # minimum trades threshold
    cache: EvalCache = field(default_factory=EvalCache)

    def __call__(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        backtest_fn: Any,
        **kw: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        cached = self.cache.get(params)
        if cached is not None:
            return self._score(cached), cached

        metrics = run_backtest(df, params, backtest_fn, **kw)
        self.cache.put(params, metrics)
        return self._score(metrics), metrics

    def _score(self, m: Dict[str, Any]) -> float:
        sharpe = float(m.get("sharpe_proxy", 0.0))
        cagr = float(m.get("CAGR", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        trades = int(m.get("trades", 0))
        trade_pen = self.mu_trade if trades < self.min_trades else 0.0
        return 0.6 * sharpe + 0.3 * cagr - self.lam_dd * dd - trade_pen


# ---------------------------------------------------------------------------
# FIT_ROBUSTE  (walk-forward)
# ---------------------------------------------------------------------------
@dataclass
class FitnessRobust:
    """Walk-forward aggregated fitness with complexity/trade penalties.

    Splits data into annual expanding folds: train on years[0..y-1], test on year y.
    Score = mean(OOS_sharpe) - penalty(std_sharpe) - penalty(low_trades) - penalty(high_dd)

    Anti-overfitting: OOS evaluation only.  The params are evaluated on the
    *test* fold — no re-optimization inside (that is done by the outer ACO loop).
    """

    lam_dd: float = 0.3
    mu_trade: float = 0.5
    sigma_penalty: float = 0.5   # penalty for cross-fold Sharpe instability
    min_trades_per_year: int = 20
    holdout_last_year: bool = True   # keep last year untouched for final OOS
    cache: EvalCache = field(default_factory=EvalCache)

    def __call__(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        backtest_fn: Any,
        **kw: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        cached = self.cache.get(params)
        if cached is not None:
            return self._aggregate(cached), cached

        years = sorted(df.index.year.unique())
        if len(years) < 3:
            # Fallback to simple evaluation
            m = run_backtest(df, params, backtest_fn, **kw)
            self.cache.put(params, m)
            return self._score_simple(m), m

        # Walk-forward: expanding train, annual test
        fold_metrics: List[Dict[str, Any]] = []
        end_idx = len(years) - 1 if self.holdout_last_year else len(years)

        for i in range(1, end_idx):
            test_year = years[i]
            train_end = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year}-12-31")

            test_df = df.loc[test_start:test_end]
            if test_df.empty or len(test_df) < 50:
                continue

            m = run_backtest(test_df, params, backtest_fn, **kw)
            m["_fold_year"] = test_year
            fold_metrics.append(m)

        if not fold_metrics:
            dummy = {"sharpe_proxy": -1.0, "CAGR": -1.0, "max_drawdown": 1.0, "trades": 0}
            self.cache.put(params, {"_folds": [], "_aggregate": dummy})
            return -10.0, dummy

        result = {"_folds": fold_metrics, "_aggregate": self._compute_aggregate(fold_metrics)}
        self.cache.put(params, result)
        return self._aggregate(result), result

    def _score_simple(self, m: Dict[str, Any]) -> float:
        sharpe = float(m.get("sharpe_proxy", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        trades = int(m.get("trades", 0))
        trade_pen = self.mu_trade if trades < self.min_trades_per_year else 0.0
        return sharpe - self.lam_dd * dd - trade_pen

    def _compute_aggregate(self, folds: List[Dict[str, Any]]) -> Dict[str, Any]:
        sharpes = [float(f.get("sharpe_proxy", 0.0)) for f in folds]
        cagrs = [float(f.get("CAGR", 0.0)) for f in folds]
        dds = [float(f.get("max_drawdown", 0.0)) for f in folds]
        trades_list = [int(f.get("trades", 0)) for f in folds]
        return {
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "mean_cagr": float(np.mean(cagrs)),
            "mean_dd": float(np.mean(dds)),
            "max_dd": float(np.max(dds)),
            "mean_trades": float(np.mean(trades_list)),
            "n_folds": len(folds),
        }

    def _aggregate(self, result: Dict[str, Any]) -> float:
        agg = result.get("_aggregate")
        if agg is None:
            return self._score_simple(result)
        mean_sh = float(agg.get("mean_sharpe", 0.0))
        std_sh = float(agg.get("std_sharpe", 0.0))
        mean_dd = float(agg.get("mean_dd", 0.0))
        mean_trades = float(agg.get("mean_trades", 0.0))
        trade_pen = self.mu_trade if mean_trades < self.min_trades_per_year else 0.0
        return mean_sh - self.sigma_penalty * std_sh - self.lam_dd * mean_dd - trade_pen
