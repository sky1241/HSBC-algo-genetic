#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.6 - Rolling Walk-Forward Analysis

Provides rolling window walk-forward validation:
- 6-month rolling windows (configurable)
- Anchored vs expanding windows
- Regime-aware re-optimization
- Automatic parameter adaptation

Usage:
    from src.spectral.walk_forward import RollingWalkForward

    wf = RollingWalkForward(
        train_months=12,
        test_months=6,
        step_months=3,
    )

    results = wf.run(df, optimize_fn, backtest_fn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
import pandas as pd
import numpy as np


@dataclass
class WFWindow:
    """A single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    @property
    def train_range(self) -> Tuple[datetime, datetime]:
        return (self.train_start, self.train_end)

    @property
    def test_range(self) -> Tuple[datetime, datetime]:
        return (self.test_start, self.test_end)


@dataclass
class WFResult:
    """Results from a single walk-forward window."""
    window: WFWindow
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    regime: Optional[str] = None


@dataclass
class WFSummary:
    """Summary of walk-forward analysis."""
    windows: List[WFResult]
    overall_metrics: Dict[str, float]
    params_stability: Dict[str, Dict[str, float]]
    n_windows: int
    total_test_period: Tuple[datetime, datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "total_test_period": [
                self.total_test_period[0].isoformat(),
                self.total_test_period[1].isoformat(),
            ],
            "overall_metrics": self.overall_metrics,
            "params_stability": self.params_stability,
            "windows": [
                {
                    "window_id": w.window.window_id,
                    "train": [w.window.train_start.isoformat(), w.window.train_end.isoformat()],
                    "test": [w.window.test_start.isoformat(), w.window.test_end.isoformat()],
                    "params": w.params,
                    "train_metrics": w.train_metrics,
                    "test_metrics": w.test_metrics,
                    "regime": w.regime,
                }
                for w in self.windows
            ],
        }


class RollingWalkForward:
    """Rolling Walk-Forward Analysis.

    Provides flexible walk-forward validation with:
    - Configurable window sizes
    - Rolling or anchored training windows
    - Regime-aware optimization
    """

    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 6,
        step_months: int = 3,
        anchored: bool = False,
        min_train_bars: int = 1000,
    ):
        """
        Args:
            train_months: Training window size in months
            test_months: Test window size in months
            step_months: Step size between windows
            anchored: If True, training always starts from beginning
            min_train_bars: Minimum bars required for training
        """
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.anchored = anchored
        self.min_train_bars = min_train_bars

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WFWindow]:
        """Generate walk-forward windows.

        Args:
            start_date: Data start date
            end_date: Data end date

        Returns:
            List of WFWindow objects
        """
        windows = []
        window_id = 0

        # Calculate in days (approximate months)
        train_days = self.train_months * 30
        test_days = self.test_months * 30
        step_days = self.step_months * 30

        if self.anchored:
            # Anchored: training always starts from beginning
            train_start = start_date
            test_start = start_date + timedelta(days=train_days)

            while test_start + timedelta(days=test_days) <= end_date:
                train_end = test_start
                test_end = test_start + timedelta(days=test_days)

                windows.append(WFWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                ))

                window_id += 1
                test_start = test_end  # Next window

        else:
            # Rolling: training window moves with test
            current_start = start_date

            while current_start + timedelta(days=train_days + test_days) <= end_date:
                train_start = current_start
                train_end = current_start + timedelta(days=train_days)
                test_start = train_end
                test_end = test_start + timedelta(days=test_days)

                windows.append(WFWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                ))

                window_id += 1
                current_start += timedelta(days=step_days)

        return windows

    def run(
        self,
        df: pd.DataFrame,
        optimize_fn: Callable[[pd.DataFrame], Dict[str, Any]],
        backtest_fn: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        regime_fn: Optional[Callable[[pd.DataFrame], str]] = None,
    ) -> WFSummary:
        """Run walk-forward analysis.

        Args:
            df: DataFrame with datetime index and OHLCV data
            optimize_fn: Function(train_df) -> params dict
            backtest_fn: Function(df, params) -> metrics dict
            regime_fn: Optional function(df) -> regime string

        Returns:
            WFSummary with all results
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        start_date = df.index.min().to_pydatetime()
        end_date = df.index.max().to_pydatetime()

        windows = self.generate_windows(start_date, end_date)

        if not windows:
            raise ValueError("No valid windows generated. Check date range and window sizes.")

        results: List[WFResult] = []

        for window in windows:
            # Slice data
            train_df = df[
                (df.index >= pd.Timestamp(window.train_start)) &
                (df.index < pd.Timestamp(window.train_end))
            ]
            test_df = df[
                (df.index >= pd.Timestamp(window.test_start)) &
                (df.index < pd.Timestamp(window.test_end))
            ]

            if len(train_df) < self.min_train_bars:
                continue

            # Detect regime if function provided
            regime = None
            if regime_fn is not None:
                try:
                    regime = regime_fn(train_df)
                except Exception:
                    pass

            # Optimize on training data
            try:
                params = optimize_fn(train_df)
            except Exception as e:
                print(f"Window {window.window_id} optimization failed: {e}")
                continue

            # Backtest on train
            try:
                train_metrics = backtest_fn(train_df, params)
            except Exception:
                train_metrics = {}

            # Backtest on test
            try:
                test_metrics = backtest_fn(test_df, params)
            except Exception:
                test_metrics = {}

            results.append(WFResult(
                window=window,
                params=params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                regime=regime,
            ))

        return self._summarize(results)

    def _summarize(self, results: List[WFResult]) -> WFSummary:
        """Summarize walk-forward results."""
        if not results:
            return WFSummary(
                windows=[],
                overall_metrics={},
                params_stability={},
                n_windows=0,
                total_test_period=(datetime.now(), datetime.now()),
            )

        # Aggregate test metrics
        all_metrics: Dict[str, List[float]] = {}
        for r in results:
            for k, v in r.test_metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(float(v))

        overall = {}
        for k, vals in all_metrics.items():
            overall[f"{k}_mean"] = float(np.mean(vals))
            overall[f"{k}_std"] = float(np.std(vals))
            overall[f"{k}_p50"] = float(np.percentile(vals, 50))

        # Parameter stability
        all_params: Dict[str, List[float]] = {}
        for r in results:
            for k, v in r.params.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    if k not in all_params:
                        all_params[k] = []
                    all_params[k].append(float(v))

        params_stability = {}
        for k, vals in all_params.items():
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                params_stability[k] = {
                    "mean": mean,
                    "std": std,
                    "cv": std / mean if mean != 0 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

        # Total test period
        test_starts = [r.window.test_start for r in results]
        test_ends = [r.window.test_end for r in results]

        return WFSummary(
            windows=results,
            overall_metrics=overall,
            params_stability=params_stability,
            n_windows=len(results),
            total_test_period=(min(test_starts), max(test_ends)),
        )


def make_monthly_folds(
    df: pd.DataFrame,
    test_months: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate monthly train/test folds.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Get unique months
    months = df.index.to_period("M").unique().sort_values()

    folds = []
    for i in range(len(months) - test_months):
        train_end_month = months[i]
        test_start_month = months[i + 1]
        test_end_month = months[min(i + test_months, len(months) - 1)]

        train_start = df.index.min()
        train_end = pd.Timestamp(train_end_month.end_time)
        test_start = pd.Timestamp(test_start_month.start_time)
        test_end = pd.Timestamp(test_end_month.end_time)

        folds.append((train_start, train_end, test_start, test_end))

    return folds


def make_annual_folds(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Generate annual folds (compatible with existing code).

    Returns list of (start_date, end_date) string tuples.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    years = df.index.year.unique()

    if start_year is not None:
        years = years[years >= start_year]
    if end_year is not None:
        years = years[years <= end_year]

    folds = []
    for year in sorted(years):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        folds.append((start, end))

    return folds
