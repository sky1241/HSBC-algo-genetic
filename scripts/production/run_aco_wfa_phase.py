#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-aware Walk-Forward with ACO (ACOR) optimizer — drop-in replacement for Optuna.

Identical logic to run_scheduler_wfa_phase.py but uses Ant Colony Optimization
instead of Optuna for per-fold, per-state parameter search.

This script does NOT modify the original WFA phase runner.
Same inputs, same outputs, same JSON format — just a different optimizer engine.

CLI example:
  py -3 scripts/production/run_aco_wfa_phase.py \
    --labels-csv data/COMBINED_labels.csv \
    --trials 300 --seed 42 --use-fused \
    --out-dir outputs/wfa_aco_phase_seed_42

Compare with Optuna version:
  py -3 scripts/production/run_scheduler_wfa_phase.py \
    --labels-csv data/COMBINED_labels.csv \
    --trials 300 --seed 42 --use-fused \
    --out-dir outputs/wfa_optuna_phase_seed_42
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ichimoku_pipeline_web_v4_8_fixed as pipe

from optimizers.aco_optimizer import ACOROptimizer, ACORConfig, decode_params
from optimizers.fitness import FitnessSimple, EvalCache

try:
    from src.checkpoint_manager import RobustRunner
except ImportError:
    RobustRunner = None

_global_runner: Optional[Any] = None


# ---------------------------------------------------------------------------
# Structures (identical to run_scheduler_wfa_phase.py)
# ---------------------------------------------------------------------------
@dataclass
class SegmentMetrics:
    start: str
    end: str
    state: str
    metrics: Dict[str, float]


@dataclass
class FoldResult:
    period_label: str
    train_range: Tuple[str, str]
    test_range: Tuple[str, str]
    params_by_state: Dict[str, Dict[str, float]]
    segments: List[SegmentMetrics]
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# ACO optimizer on train data (replaces _optimize_on_train from Optuna version)
# ---------------------------------------------------------------------------
def _optimize_on_train_aco(
    train_df: pd.DataFrame,
    timeframe: str,
    n_trials: int,
    seed: int | None,
    loss_mult: float,
    q: float = 0.5,
    xi: float = 1.0,
    progress_path: Optional[Path] = None,
    folds_done: int = 0,
    folds_total: int = 1,
    trial_log_path: Optional[Path] = None,
    trial_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Find best Ichimoku params on train_df using ACOR.

    Budget is mapped to ACO hyperparams:
      archive = min(30, n_trials // 3)
      n_ants  = max(5, archive // 3)
      max_iter = remaining budget / n_ants
    This gives roughly the same total evaluations as Optuna n_trials.
    """
    # Map n_trials budget to ACO hyperparams
    archive_size = min(30, max(10, n_trials // 3))
    n_ants = max(3, archive_size // 3)
    remaining = max(1, n_trials - archive_size)
    max_iter = max(1, remaining // n_ants)

    cfg = ACORConfig(
        n_ants=n_ants,
        archive_size=archive_size,
        q=q,
        xi=xi,
        max_iter=max_iter,
        seed=seed or 42,
        stagnation_limit=max(3, max_iter // 5),
    )

    # Fitness: simple single-window on train data (same logic as Optuna objective)
    fitness = FitnessSimple(lam_dd=0.3, mu_trade=0.5, min_trades=10)

    optimizer = ACOROptimizer(cfg)

    # Progress callback
    def _on_iter(it: int, best: Any, history: list) -> None:
        if progress_path is not None:
            try:
                total_evals = cfg.archive_size + it * cfg.n_ants
                frac = float(total_evals) / float(max(1, n_trials))
                percent = max(0.0, min(100.0, 100.0 * (float(folds_done) + frac) / float(max(1, folds_total))))
                payload = {
                    "folds_done": int(folds_done),
                    "folds_total": int(folds_total),
                    "trial": int(total_evals),
                    "trials_total": int(n_trials),
                    "percent": float(round(percent, 2)),
                    "phase": "aco",
                }
                tmp = str(progress_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, progress_path)
            except Exception:
                pass

        # Per-trial JSONL logging
        if trial_log_path is not None:
            try:
                trial_log_path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "kind": "aco_phase",
                    "iteration": it,
                    "best_score": float(best.score),
                    "best_params": best.decoded,
                    "run_context": (trial_context or {}),
                }
                with open(trial_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    best = optimizer.optimize(
        fitness_fn=fitness,
        df=train_df,
        backtest_fn=pipe.backtest_long_short,
        callback=_on_iter,
        symbol="BTC/USDT",
        timeframe=timeframe,
        loss_mult=loss_mult,
    )

    params = best.decoded
    params["tp_mult"] = None  # Not optimized in current ACOR space
    return params


# ---------------------------------------------------------------------------
# Apply on test segment (identical to Optuna version)
# ---------------------------------------------------------------------------
def _apply_on_segment(
    test_df: pd.DataFrame,
    timeframe: str,
    params: Dict[str, float],
    loss_mult: float,
    confidence_series: Optional[pd.Series] = None,
) -> Dict[str, float]:
    m = pipe.backtest_long_short(
        test_df,
        int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]),
        int(params["shift"]), float(params["atr_mult"]),
        loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
        tp_mult=float(params.get("tp_mult")) if params.get("tp_mult") is not None else None,
        confidence_series=confidence_series,
    )
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}


# ---------------------------------------------------------------------------
# ATR sweep (identical to Optuna version)
# ---------------------------------------------------------------------------
def _sweep_atr_local(
    train_df: pd.DataFrame,
    timeframe: str,
    base_params: Dict[str, float],
    loss_mult: float,
    span: float,
    step: float,
    mdd_max: float | None,
) -> Dict[str, float]:
    params = dict(base_params)
    center = float(params.get("atr_mult", 3.0))
    start = max(0.05, center - span)
    end = center + span + 1e-12
    n_steps = int(round((end - start) / step)) + 1
    candidates = sorted(set(
        [float(round(start + i * step, 6)) for i in range(max(1, n_steps)) if start + i * step > 0]
        + [center]
    ))

    best_any = None
    best_allowed = None
    for atr in candidates:
        m = pipe.backtest_long_short(
            train_df,
            int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]),
            int(params["shift"]), float(atr),
            loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
            tp_mult=float(params.get("tp_mult")) if params.get("tp_mult") is not None else None,
        )
        rec = {
            "atr": float(atr),
            "sharpe": float(m.get("sharpe_proxy", 0.0)),
            "dd": float(m.get("max_drawdown", 0.0)),
            "eq": float(m.get("equity_mult", 0.0)),
        }
        if best_any is None or rec["sharpe"] > best_any["sharpe"]:
            best_any = rec
        if (mdd_max is None or rec["dd"] <= float(mdd_max)):
            if best_allowed is None or rec["sharpe"] > best_allowed["sharpe"]:
                best_allowed = rec
    chosen = best_allowed if best_allowed is not None else best_any
    if chosen is not None:
        params["atr_mult"] = float(chosen["atr"])
    return params


# ---------------------------------------------------------------------------
# Segment extraction (identical)
# ---------------------------------------------------------------------------
def _segments_from_labels(index, labels):
    segments = []
    if index.empty:
        return segments
    s = labels.reindex(index, method='ffill').astype(str)
    start = s.index[0]
    cur = s.iloc[0]
    for ts, v in zip(s.index[1:], s.iloc[1:]):
        if str(v) != str(cur):
            segments.append((start, ts, str(cur)))
            start = ts
            cur = v
    segments.append((start, s.index[-1], str(cur)))
    return segments


# ---------------------------------------------------------------------------
# Main WFA loop (mirrors run_phase_wfa but with ACO)
# ---------------------------------------------------------------------------
def run_phase_wfa_aco(
    df: pd.DataFrame,
    labels: pd.Series,
    granularity: str,
    n_trials: int,
    seed: int | None,
    loss_mult: float,
    atr_sweep: bool,
    atr_sweep_span: float,
    atr_sweep_step: float,
    mdd_max: float | None,
    q: float = 0.5,
    xi: float = 1.0,
    progress_path: Optional[Path] = None,
    trial_log_path: Optional[Path] = None,
    k_hint: Optional[str] = None,
    checkpoint_callback: Optional[Any] = None,
    confidence_series: Optional[pd.Series] = None,
    resume_fold: Optional[int] = None,
) -> Tuple[List[FoldResult], Dict[str, float]]:
    """Phase-aware WFA using ACOR instead of Optuna."""
    timeframe = "2h"
    folds: List[FoldResult] = []

    if granularity != "annual":
        raise ValueError("Only 'annual' granularity supported.")

    years = sorted(df.index.year.unique())
    y0, y1 = int(min(years)), int(max(years))
    folds_total = max(0, y1 - (y0 + 1) + 1)
    folds_done = 0

    for y in range(y0 + 1, y1 + 1):
        # Skip folds already completed (resume support)
        if resume_fold is not None and y < resume_fold:
            print(f"  [RESUME] Skipping fold {y} (already completed)")
            folds_done += 1
            continue
        train_start = pd.Timestamp(f"{y0}-01-01")
        train_end = pd.Timestamp(f"{y-1}-12-31")
        test_start = pd.Timestamp(f"{y}-01-01")
        test_end = pd.Timestamp(f"{y}-12-31")
        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]
        if train_df.empty or test_df.empty:
            continue

        lbl_train = labels.reindex(train_df.index, method='ffill').astype(str)
        lbl_test = labels.reindex(test_df.index, method='ffill').astype(str)
        states = sorted(set(lbl_train.dropna().astype(str).unique().tolist()))
        params_by_state: Dict[str, Dict[str, float]] = {}
        completed_phases: List[str] = []

        # 1) GLOBAL optimization on full train (ACO)
        print(f"  [ACO] Fold {y}: GLOBAL optimization ({n_trials} evals)...")
        global_params = _optimize_on_train_aco(
            train_df, timeframe, n_trials, seed, loss_mult, q=q, xi=xi,
            progress_path=progress_path, folds_done=folds_done, folds_total=folds_total or 1,
            trial_log_path=trial_log_path,
            trial_context={"k": (k_hint or "K"), "fold": f"{y}", "phase_label": "GLOBAL", "seed": seed},
        )
        if atr_sweep:
            global_params = _sweep_atr_local(train_df, timeframe, global_params, loss_mult,
                                              atr_sweep_span, atr_sweep_step, mdd_max)
        completed_phases.append("GLOBAL")
        print(f"  [ACO] Fold {y}: GLOBAL -> {json.dumps({k: v for k, v in global_params.items() if k != 'tp_mult'})}")

        if checkpoint_callback is not None:
            try:
                checkpoint_callback(fold=y, phase="GLOBAL", trial=n_trials, total_trials=n_trials,
                                    best_params=global_params, best_score=0.0, completed_phases=completed_phases.copy())
            except Exception:
                pass

        # 2) Per-state optimization (ACO on each state subset)
        for st in states:
            sub = train_df[lbl_train == st]
            if len(sub) < 200:
                params_by_state[st] = dict(global_params)
                print(f"  [ACO] Fold {y}: state '{st}' too small ({len(sub)} bars) -> use GLOBAL params")
            else:
                print(f"  [ACO] Fold {y}: state '{st}' optimization ({len(sub)} bars, {n_trials} evals)...")
                p = _optimize_on_train_aco(
                    sub, timeframe, n_trials, seed, loss_mult, q=q, xi=xi,
                    trial_log_path=trial_log_path,
                    trial_context={"k": (k_hint or "K"), "fold": f"{y}", "phase_label": str(st), "seed": seed},
                )
                if atr_sweep:
                    p = _sweep_atr_local(sub, timeframe, p, loss_mult, atr_sweep_span, atr_sweep_step, mdd_max)
                params_by_state[st] = p
                print(f"  [ACO] Fold {y}: state '{st}' -> {json.dumps({k: v for k, v in p.items() if k != 'tp_mult'})}")

            completed_phases.append(str(st))
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(fold=y, phase=str(st), trial=n_trials, total_trials=n_trials,
                                        best_params=params_by_state.get(st, global_params), best_score=0.0,
                                        completed_phases=completed_phases.copy())
                except Exception:
                    pass

        # 3) Evaluate on test by contiguous segments (identical to Optuna version)
        segs = _segments_from_labels(test_df.index, lbl_test)
        seg_metrics: List[SegmentMetrics] = []
        for s0, s1, st in segs:
            seg_slice = test_df.loc[(test_df.index >= s0) & (test_df.index <= s1)]
            if seg_slice.empty:
                continue
            p = params_by_state.get(st, global_params)
            _seg_conf = None
            if confidence_series is not None:
                try:
                    _seg_conf = confidence_series.reindex(seg_slice.index, method='ffill')
                except Exception:
                    pass
            m = _apply_on_segment(seg_slice, timeframe, p, loss_mult, confidence_series=_seg_conf)
            seg_metrics.append(SegmentMetrics(start=s0.strftime('%Y-%m-%d'), end=s1.strftime('%Y-%m-%d'),
                                               state=str(st), metrics=m))

        # 4) Compose fold-level metrics
        eq_mult = 1.0
        trades_total = 0
        sharpe_vals = []
        min_equity_global = 1.0
        cur_equity = 1.0
        for sm in seg_metrics:
            m = sm.metrics
            eq = float(m.get("equity_mult", 1.0))
            me = float(m.get("min_equity", 1.0))
            tr = int(m.get("trades", 0))
            sh = float(m.get("sharpe_proxy", 0.0))
            min_equity_global = min(min_equity_global, cur_equity * me)
            cur_equity *= eq
            eq_mult *= eq
            trades_total += tr
            sharpe_vals.append(sh)

        overall_fold = {
            "equity_mult": float(eq_mult),
            "max_drawdown": float(1.0 - min_equity_global),
            "trades": int(trades_total),
            "sharpe_proxy_mean": float(np.mean(sharpe_vals) if sharpe_vals else 0.0),
        }
        print(f"  [ACO] Fold {y} TEST: eq={eq_mult:.3f}x, MDD={1-min_equity_global:.2%}, "
              f"trades={trades_total}, Sharpe~{np.mean(sharpe_vals) if sharpe_vals else 0:.2f}")

        folds.append(FoldResult(
            period_label=f"{y}",
            train_range=(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
            test_range=(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')),
            params_by_state=params_by_state,
            segments=seg_metrics,
            metrics=overall_fold,
        ))
        folds_done += 1

    # Aggregate across folds
    eq_mult_all = 1.0
    min_equity_global = 1.0
    cur_equity = 1.0
    trades_total = 0
    sharpes_all = []
    for fr in folds:
        m = fr.metrics
        eq = float(m.get("equity_mult", 1.0))
        me = float(1.0 - float(m.get("max_drawdown", 0.0)))
        min_equity_global = min(min_equity_global, cur_equity * me)
        cur_equity *= eq
        eq_mult_all *= eq
        trades_total += int(m.get("trades", 0))
        sharpes_all.append(float(m.get("sharpe_proxy_mean", 0.0)))
    overall = {
        "equity_mult": float(eq_mult_all),
        "max_drawdown": float(1.0 - min_equity_global),
        "trades": int(trades_total),
        "sharpe_proxy_mean": float(np.mean(sharpes_all) if sharpes_all else 0.0),
        "folds": len(folds),
        "optimizer": "ACOR",
    }
    return folds, overall


# ---------------------------------------------------------------------------
# Signal handlers (same as Optuna version)
# ---------------------------------------------------------------------------
def _shutdown_handler(signum, frame):
    global _global_runner
    print(f"\n[CHECKPOINT] Signal {signum} received, saving checkpoint...")
    if _global_runner is not None:
        try:
            _global_runner.shutdown()
        except Exception:
            pass
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global _global_runner

    ap = argparse.ArgumentParser(
        description="BTC-only phase-aware WFA (annual) with ACOR optimizer",
        epilog="Drop-in replacement for run_scheduler_wfa_phase.py — same inputs, same JSON output format.",
    )
    ap.add_argument("--labels-csv", required=True, help="Path to labels CSV (timestamp,label)")
    ap.add_argument("--granularity", choices=["annual"], default="annual")
    ap.add_argument("--trials", type=int, default=300, help="Eval budget per optimization (mapped to ACO params)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--loss-mult", type=float, default=3.0)
    ap.add_argument("--atr-sweep", action="store_true")
    ap.add_argument("--atr-sweep-span", type=float, default=1.0)
    ap.add_argument("--atr-sweep-step", type=float, default=0.2)
    ap.add_argument("--mdd-max", type=float, default=None)
    ap.add_argument("--use-fused", action="store_true")
    ap.add_argument("--out-dir", default="outputs/wfa_aco_phase")
    # ACO-specific
    ap.add_argument("--q", type=float, default=0.5, help="ACOR exploration param")
    ap.add_argument("--xi", type=float, default=1.0, help="ACOR deviation ratio")
    # Checkpoint
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--no-checkpoint", action="store_true")
    # Confidence sizing
    ap.add_argument("--confidence-sizing", action="store_true")
    ap.add_argument("--confidence-csv", default=None)

    args, _unknown = ap.parse_known_args()

    if args.use_fused:
        os.environ["USE_FUSED_H2"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint setup
    use_checkpoint = RobustRunner is not None and not args.no_checkpoint
    runner = None
    if use_checkpoint:
        runner = RobustRunner(out_dir, checkpoint_interval_minutes=args.checkpoint_interval, min_free_gb=10.0)
        _global_runner = runner
        try:
            signal.signal(signal.SIGINT, _shutdown_handler)
            signal.signal(signal.SIGTERM, _shutdown_handler)
        except Exception:
            pass
        def _atexit_save():
            if _global_runner is not None:
                try:
                    _global_runner.shutdown()
                except Exception:
                    pass
        atexit.register(_atexit_save)

    # Load data
    print(f"[ACO-WFA] Loading BTC fused H2 data...")
    df = pipe._load_local_csv_if_configured("BTC/USDT", "2h")
    if df is None:
        raise RuntimeError("Fused CSV not found. Use --use-fused or set USE_FUSED_H2=1.")
    df = pipe.ensure_utc_index(df)
    print(f"[ACO-WFA] Data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

    # Load labels
    labels_df = pd.read_csv(args.labels_csv, parse_dates=["timestamp"])
    if "label" not in labels_df.columns and "state" in labels_df.columns:
        labels_df = labels_df.rename(columns={"state": "label"})
    if "label" not in labels_df.columns:
        raise RuntimeError("Labels CSV must contain 'label' column")
    labels = labels_df.set_index("timestamp").sort_index()["label"].astype(str)

    # K hint
    k_hint = None
    try:
        name = Path(args.labels_csv).stem
        if name.upper().startswith("K"):
            k_hint = name.upper()
    except Exception:
        pass

    progress_path = out_dir / "PROGRESS.json"
    jsonl_path = None
    try:
        jsonl_dir = ROOT / "outputs" / "trial_logs" / "aco_phase" / (k_hint or "K")
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / "trials_from_aco_wfa.jsonl"
    except Exception:
        pass

    # Confidence series
    _conf_series = None
    if args.confidence_sizing:
        try:
            from src.confidence_sizing import load_confidence_from_labels
            _conf_csv = args.confidence_csv or args.labels_csv
            _conf_series = load_confidence_from_labels(_conf_csv)
            if _conf_series is not None:
                print(f"[SIZING] Confidence sizing ACTIVE: {len(_conf_series)} values")
        except Exception as e:
            print(f"[SIZING] Error: {e}. Using fixed 1% sizing.")

    # Checkpoint callback
    def checkpoint_callback(**kwargs):
        if runner is not None:
            runner.save_progress(seed=args.seed, **kwargs)

    # Check for resume state
    resume_fold = None
    if use_checkpoint and runner is not None:
        resume_state = runner.get_resume_state()
        if resume_state:
            resume_fold = resume_state.get('fold')
            print(f"[CHECKPOINT] Resuming from: fold={resume_fold}, "
                  f"phase={resume_state.get('phase')}, trial={resume_state.get('trial')}")
        else:
            print(f"[CHECKPOINT] Starting fresh run with checkpoints every {args.checkpoint_interval} min")

    # RUN
    print(f"[ACO-WFA] Starting ACOR phase-aware WFA: {args.trials} evals/optimization, seed={args.seed}")
    print(f"[ACO-WFA] ACOR params: q={args.q}, xi={args.xi}")
    print(f"[ACO-WFA] Labels: {args.labels_csv}")
    print()

    folds, overall = run_phase_wfa_aco(
        df, labels, args.granularity, args.trials, args.seed, args.loss_mult,
        args.atr_sweep, args.atr_sweep_span, args.atr_sweep_step,
        None if args.mdd_max is None else float(args.mdd_max),
        q=args.q, xi=args.xi,
        progress_path=progress_path, trial_log_path=jsonl_path, k_hint=k_hint,
        checkpoint_callback=checkpoint_callback if use_checkpoint else None,
        confidence_series=_conf_series,
        resume_fold=resume_fold,
    )

    # Save (same JSON format as Optuna version)
    ts = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    k_suffix = (k_hint or "K").upper()
    out_json = out_dir / f"WFA_aco_phase_{k_suffix}_BTC_fused_{ts}.json"

    payload = {
        "optimizer": "ACOR",
        "granularity": args.granularity,
        "trials": int(args.trials),
        "acor_q": args.q,
        "acor_xi": args.xi,
        "loss_mult": float(args.loss_mult),
        "labels_csv": str(Path(args.labels_csv).as_posix()),
        "confidence_sizing": _conf_series is not None,
        "overall": overall,
        "folds": [
            {
                "period": fr.period_label,
                "train": fr.train_range,
                "test": fr.test_range,
                "params_by_state": fr.params_by_state,
                "segments": [{"start": sm.start, "end": sm.end, "state": sm.state, "metrics": sm.metrics}
                             for sm in fr.segments],
                "metrics": fr.metrics,
            }
            for fr in folds
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"[ACO-WFA] COMPLETE — Saved: {out_json}")
    print(f"  Equity:  {overall['equity_mult']:.3f}x")
    print(f"  MDD:     {overall['max_drawdown']:.2%}")
    print(f"  Trades:  {overall['trades']}")
    print(f"  Sharpe~: {overall['sharpe_proxy_mean']:.2f}")
    print(f"  Folds:   {overall['folds']}")
    print(f"  Engine:  ACOR (q={args.q}, xi={args.xi})")
    print("=" * 60)

    if runner is not None:
        runner.complete_seed(args.seed, overall)
        runner.shutdown()
        _global_runner = None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
