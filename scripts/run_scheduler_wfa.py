#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward (scheduler) backtest for BTC-only on fused 2h data.

- Granularity: annual (expanding train) or monthly (rolling 12m train)
- Optimizes Ichimoku params on train with Optuna, applies on next period
- Outputs a JSON summary with per-fold params and OOS metrics + overall stats

Example (annual):
  .\.venv\Scripts\python.exe scripts\run_scheduler_wfa.py --use-fused \
      --granularity annual --trials 300 --out-dir outputs\scheduler_annual_btc
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Import pipeline
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore

try:
    import optuna  # type: ignore
except Exception as _e:
    optuna = None


@dataclass
class FoldResult:
    period_label: str
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    params: Dict[str, float]
    metrics: Dict[str, float]


def _load_btc_fused(timeframe: str = "2h") -> pd.DataFrame:
    df = pipe._load_local_csv_if_configured("BTC/USDT", timeframe)
    if df is None:
        raise RuntimeError("Fused CSV not configured. Pass --use-fused or set USE_FUSED_H2=1.")
    return pipe.ensure_utc_index(df)


def _optimize_on_train(train_df: pd.DataFrame, timeframe: str, n_trials: int, seed: int | None, jobs: int, loss_mult: float) -> Dict[str, float]:
    if optuna is None:
        raise RuntimeError("Optuna non disponible. pip install optuna")

    def _objective(trial):
        p = pipe.sample_params_optuna(trial)
        m = pipe.backtest_long_short(
            train_df,
            int(p["tenkan"]), int(p["kijun"]), int(p["senkou_b"]), int(p["shift"]), float(p["atr_mult"]),
            loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
        )
        cagr = float(m.get("CAGR", 0.0))
        sharpe = float(m.get("sharpe_proxy", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        trades = int(m.get("trades", 0))
        score = 0.6 * sharpe + 0.3 * cagr - 0.3 * dd - (0.5 if trades < 30 else 0.0)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
    )
    study.optimize(_objective, n_trials=int(n_trials), n_jobs=int(jobs))
    bp = study.best_trial.params
    # Normalize keys
    params = {
        "tenkan": int(bp.get("tenkan")),
        "kijun": int(bp.get("kijun", max(int(bp.get("tenkan")), int(bp.get("r_kijun", 1)) * int(bp.get("tenkan"))))),
        "senkou_b": int(bp.get("senkou_b",  max(int(bp.get("kijun", 26)), int(bp.get("r_senkou", 1)) * int(bp.get("tenkan"))))),
        "shift": int(bp.get("shift")),
        "atr_mult": float(bp.get("atr_mult")),
    }
    return params


def _apply_on_test(test_df: pd.DataFrame, timeframe: str, params: Dict[str, float], loss_mult: float) -> Dict[str, float]:
    m = pipe.backtest_long_short(
        test_df,
        int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]), int(params["shift"]), float(params["atr_mult"]),
        loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
    )
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}


def _year_bounds(df: pd.DataFrame) -> Tuple[int, int]:
    years = pd.Index(sorted(df.index.year.unique()))
    return int(years.min()), int(years.max())


def _month_list(df: pd.DataFrame) -> List[pd.Timestamp]:
    return list(pd.Index(sorted(df.index.to_period('M').to_timestamp().unique())))


def run_wfa(df: pd.DataFrame, granularity: str, n_trials: int, seed: int | None, jobs: int, loss_mult: float) -> Tuple[List[FoldResult], Dict[str, float]]:
    timeframe = "2h"
    folds: List[FoldResult] = []
    if granularity == "annual":
        y0, y1 = _year_bounds(df)
        for y in range(y0 + 1, y1 + 1):
            train_start = pd.Timestamp(f"{y0}-01-01"); train_end = pd.Timestamp(f"{y-1}-12-31")
            test_start = pd.Timestamp(f"{y}-01-01");  test_end = pd.Timestamp(f"{y}-12-31")
            train_df = df.loc[train_start:train_end]
            test_df = df.loc[test_start:test_end]
            if train_df.empty or test_df.empty:
                continue
            params = _optimize_on_train(train_df, timeframe, n_trials, seed, jobs, loss_mult)
            metrics = _apply_on_test(test_df, timeframe, params, loss_mult)
            folds.append(FoldResult(
                period_label=f"{y}",
                train_range=(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                test_range=(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')),
                params=params,
                metrics=metrics,
            ))
    else:  # monthly rolling 12m train
        months = _month_list(df)
        for i in range(12, len(months)):
            test_month = months[i]
            test_start = test_month.replace(day=1)
            next_month = (test_start + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)
            test_end = (next_month - pd.Timedelta(days=1))
            train_end = test_start - pd.Timedelta(days=1)
            train_start = test_start - pd.Timedelta(days=365)
            train_df = df.loc[train_start:train_end]
            test_df = df.loc[test_start:test_end]
            if len(train_df) < 200 or len(test_df) < 10:
                continue
            params = _optimize_on_train(train_df, timeframe, n_trials, seed, jobs, loss_mult)
            metrics = _apply_on_test(test_df, timeframe, params, loss_mult)
            folds.append(FoldResult(
                period_label=test_start.strftime('%Y-%m'),
                train_range=(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                test_range=(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')),
                params=params,
                metrics=metrics,
            ))

    # Aggregate OOS by compounding equity
    eq_mult = 1.0
    dd_max = 0.0
    trades_total = 0
    sharpes: List[float] = []
    cagrs: List[float] = []
    for fr in folds:
        eq_mult *= float(fr.metrics.get("equity_mult", 1.0))
        dd_max = max(dd_max, float(fr.metrics.get("max_drawdown", 0.0)))
        trades_total += int(fr.metrics.get("trades", 0))
        sharpes.append(float(fr.metrics.get("sharpe_proxy", 0.0)))
        cagrs.append(float(fr.metrics.get("CAGR", 0.0)))
    overall = {
        "equity_mult": float(eq_mult),
        "max_drawdown": float(dd_max),
        "trades": int(trades_total),
        "sharpe_proxy_mean": float(np.mean(sharpes) if sharpes else 0.0),
        "CAGR_mean": float(np.mean(cagrs) if cagrs else 0.0),
        "folds": len(folds),
    }
    return folds, overall


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC-only WFA scheduler (annual/monthly) on fused 2h")
    ap.add_argument("--granularity", choices=["annual", "monthly"], default="annual")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--loss-mult", type=float, default=3.0)
    ap.add_argument("--use-fused", action="store_true")
    ap.add_argument("--out-dir", default="outputs/scheduler_wfa")
    args = ap.parse_args()

    if args.use_fused:
        os.environ["USE_FUSED_H2"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_btc_fused("2h")
    df = pipe.ensure_utc_index(df)

    folds, overall = run_wfa(df, args.granularity, int(args.trials), int(args.seed), int(args.jobs), float(args.loss_mult))

    ts = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"WFA_{args.granularity}_BTC_fused_{ts}.json"
    payload = {
        "granularity": args.granularity,
        "trials": int(args.trials),
        "jobs": int(args.jobs),
        "loss_mult": float(args.loss_mult),
        "overall": overall,
        "folds": [
            {
                "period": fr.period_label,
                "train": fr.train_range,
                "test": fr.test_range,
                "params": fr.params,
                "metrics": fr.metrics,
            }
            for fr in folds
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    print(f"Overall: equity× {overall['equity_mult']:.3f}, MDD {overall['max_drawdown']:.2%}, trades {overall['trades']}, Sharpe≈{overall['sharpe_proxy_mean']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



