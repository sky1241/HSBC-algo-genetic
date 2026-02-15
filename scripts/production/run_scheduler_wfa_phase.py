#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-aware Walk-Forward (annual) backtest for BTC-only on fused 2h data.

This runner uses precomputed label CSVs (e.g., outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv)
to split train/test by HMM state and optimize Ichimoku+ATR per-state on the train fold.
For the test fold, it applies the per-state parameters on contiguous state segments and
stitches segment metrics into a fold-level result.

Outputs one JSON per run compatible with summarize_wfa.py (has top-level 'overall' and 'folds').

CLI example (K=3 labels, annual WFA):
  py -3 scripts/run_scheduler_wfa_phase.py --labels-csv outputs/fourier/labels_frozen/BTC_FUSED_2h/K3.csv \
      --granularity annual --trials 300 --jobs 1 --seed 123 --atr-sweep --atr-sweep-span 1.0 --atr-sweep-step 0.2 \
      --mdd-max 0.20 --use-fused --out-dir outputs/wfa_phase_k3/seed_123
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

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ichimoku_pipeline_web_v4_8_fixed as pipe  # type: ignore

# Checkpoint system for crash recovery
try:
    from src.checkpoint_manager import RobustRunner, CheckpointManager
except ImportError:
    RobustRunner = None
    CheckpointManager = None

try:
    import optuna  # type: ignore
except Exception:
    optuna = None

# Global runner for signal handlers
_global_runner: Optional["RobustRunner"] = None


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


def _load_btc_fused(timeframe: str = "2h") -> pd.DataFrame:
    df = pipe._load_local_csv_if_configured("BTC/USDT", timeframe)
    if df is None:
        raise RuntimeError("Fused CSV not configured. Pass --use-fused or set USE_FUSED_H2=1.")
    return pipe.ensure_utc_index(df)


def _optimize_on_train(train_df: pd.DataFrame, timeframe: str, n_trials: int, seed: int | None, jobs: int, loss_mult: float,
                       progress_path: Optional[Path] = None, folds_done: int = 0, folds_total: int = 1,
                       trial_log_path: Optional[Path] = None, trial_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    if optuna is None:
        raise RuntimeError("Optuna non disponible. pip install optuna")

    def _objective(trial):
        p = pipe.sample_params_optuna(trial)
        m = pipe.backtest_long_short(
            train_df,
            int(p["tenkan"]), int(p["kijun"]), int(p["senkou_b"]), int(p["shift"]), float(p["atr_mult"]),
            loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
            tp_mult=float(p.get("tp_mult")) if "tp_mult" in p else None,
        )
        cagr = float(m.get("CAGR", 0.0))
        sharpe = float(m.get("sharpe_proxy", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        trades = int(m.get("trades", 0))
        score = 0.6 * sharpe + 0.3 * cagr - 0.3 * dd - (0.5 if trades < 30 else 0.0)
        try:
            # Attach train metrics to the trial for callback logging
            trial.set_user_attr("train_metrics", {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()})
        except Exception:
            pass
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
    )
    def _cb(st: "optuna.study.Study", tr: "optuna.trial.FrozenTrial") -> None:  # type: ignore[name-defined]
        # Update PROGRESS.json if available
        if progress_path is not None:
            try:
                tnum = int(getattr(tr, "number", 0)) + 1
                frac_trials = float(tnum) / float(max(1, int(n_trials)))
                percent = max(0.0, min(100.0, 100.0 * (float(folds_done) + frac_trials) / float(max(1, folds_total))))
                payload = {
                    "folds_done": int(folds_done),
                    "folds_total": int(folds_total),
                    "trial": int(tnum),
                    "trials_total": int(n_trials),
                    "percent": float(round(percent, 2)),
                    "phase": "optuna"
                }
                tmp = str(progress_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, progress_path)
            except Exception:
                pass

        # Per-trial JSONL logging (for live heatmaps)
        try:
            if trial_log_path is not None:
                trial_log_path.parent.mkdir(parents=True, exist_ok=True)
                rec: Dict[str, Any] = {
                    "kind": "phase",
                    "trial_number": int(getattr(tr, "number", 0)),
                    "params": dict(tr.params or {}),
                    "score": (float(tr.value) if isinstance(tr.value, (int, float, np.floating)) else tr.value),
                    "run_context": (trial_context or {}),
                }
                tm = getattr(tr, "user_attrs", {}).get("train_metrics") if hasattr(tr, "user_attrs") else None
                if isinstance(tm, dict):
                    rec["metrics_train"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in tm.items()}
                with open(trial_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    study.optimize(_objective, n_trials=int(n_trials), n_jobs=int(jobs), callbacks=[_cb])
    bp = study.best_trial.params
    params = {
        "tenkan": int(bp.get("tenkan")),
        "kijun": int(bp.get("kijun", max(int(bp.get("tenkan")), int(bp.get("r_kijun", 1)) * int(bp.get("tenkan"))))),
        "senkou_b": int(bp.get("senkou_b", max(int(bp.get("kijun", 26)), int(bp.get("r_senkou", 1)) * int(bp.get("tenkan"))))),
        "shift": int(bp.get("shift")),
        "atr_mult": float(bp.get("atr_mult")),
        "tp_mult": float(bp.get("tp_mult")) if "tp_mult" in bp else None,
    }
    return params


def _apply_on_segment(test_df: pd.DataFrame, timeframe: str, params: Dict[str, float], loss_mult: float,
                      confidence_series: Optional[pd.Series] = None) -> Dict[str, float]:
    m = pipe.backtest_long_short(
        test_df,
        int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]), int(params["shift"]), float(params["atr_mult"]),
        loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
        tp_mult=float(params.get("tp_mult")) if "tp_mult" in params else None,
        confidence_series=confidence_series,
    )
    # normalize numeric
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}


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
    span = max(0.0, float(span))
    step = max(1e-6, float(step))
    start = max(0.05, center - span)
    end = center + span + 1e-12
    n_steps = int(round((end - start) / step)) + 1
    candidates: List[float] = []
    for i in range(max(1, n_steps)):
        v = start + i * step
        if v <= 0:
            continue
        candidates.append(float(round(v, 6)))
    if center not in candidates:
        candidates.append(center)
    candidates = sorted(set(candidates))

    best_any: Dict[str, float] | None = None
    best_allowed: Dict[str, float] | None = None
    for atr in candidates:
        m = pipe.backtest_long_short(
            train_df,
            int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]), int(params["shift"]), float(atr),
            loss_mult=float(loss_mult), symbol="BTC/USDT", timeframe=timeframe,
            tp_mult=float(params.get("tp_mult")) if "tp_mult" in params else None,
        )
        rec: Dict[str, float] = {
            "atr": float(atr),
            "sharpe": float(m.get("sharpe_proxy", 0.0)),
            "dd": float(m.get("max_drawdown", 0.0)),
            "eq": float(m.get("equity_mult", 0.0)),
        }
        if (best_any is None) or (rec["sharpe"] > best_any["sharpe"]) or (rec["sharpe"] == best_any["sharpe"] and rec["eq"] > best_any["eq"]):
            best_any = rec
        allowed = (mdd_max is None) or (rec["dd"] <= float(mdd_max))
        if allowed:
            if (best_allowed is None) or (rec["sharpe"] > best_allowed["sharpe"]) or (rec["sharpe"] == best_allowed["sharpe"] and rec["eq"] > best_allowed["eq"]):
                best_allowed = rec
    chosen = best_allowed if best_allowed is not None else best_any
    if chosen is not None:
        params["atr_mult"] = float(chosen["atr"])
    return params


def _segments_from_labels(index: pd.DatetimeIndex, labels: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
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


def run_phase_wfa(
    df: pd.DataFrame,
    labels: pd.Series,
    granularity: str,
    n_trials: int,
    seed: int | None,
    jobs: int,
    loss_mult: float,
    atr_sweep: bool,
    atr_sweep_span: float,
    atr_sweep_step: float,
    mdd_max: float | None,
    progress_path: Optional[Path] = None,
    trial_log_path: Optional[Path] = None,
    k_hint: Optional[str] = None,
    checkpoint_callback: Optional[Any] = None,
    confidence_series: Optional[pd.Series] = None,
    resume_fold: Optional[int] = None,
) -> Tuple[List[FoldResult], Dict[str, float]]:
    """
    Run phase-aware Walk-Forward Analysis.

    Args:
        checkpoint_callback: Optional callback(fold, phase, trial, total_trials, best_params, best_score, completed_phases)
                            Called periodically to save progress for crash recovery.
        confidence_series: Optional P(bull) series for confidence-based position sizing (test only).
        resume_fold: Optional fold number to resume from (skips earlier folds).
    """
    timeframe = "2h"
    folds: List[FoldResult] = []

    def _year_bounds(df_: pd.DataFrame) -> Tuple[int, int]:
        years = pd.Index(sorted(df_.index.year.unique()))
        return int(years.min()), int(years.max())

    if granularity != "annual":
        raise ValueError("Only 'annual' granularity is supported in phase runner for now.")

    y0, y1 = _year_bounds(df)
    folds_total = max(0, (y1 - (y0 + 1) + 1))
    folds_done = 0
    for y in range(y0 + 1, y1 + 1):
        # Skip folds already completed (resume support)
        if resume_fold is not None and y < resume_fold:
            print(f"  [RESUME] Skipping fold {y} (already completed)")
            folds_done += 1
            continue

        train_start = pd.Timestamp(f"{y0}-01-01"); train_end = pd.Timestamp(f"{y-1}-12-31")
        test_start = pd.Timestamp(f"{y}-01-01");  test_end = pd.Timestamp(f"{y}-12-31")
        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]
        if train_df.empty or test_df.empty:
            continue

        lbl_train = labels.reindex(train_df.index, method='ffill').astype(str)
        lbl_test = labels.reindex(test_df.index, method='ffill').astype(str)
        states = sorted(set(lbl_train.dropna().astype(str).unique().tolist()))
        params_by_state: Dict[str, Dict[str, float]] = {}

        # Track completed phases for checkpoint
        completed_phases: List[str] = []

        # Fallback: global params if state subset too small
        global_params = _optimize_on_train(
            train_df, timeframe, n_trials, seed, jobs, loss_mult,
            progress_path=progress_path, folds_done=folds_done, folds_total=folds_total or 1,
            trial_log_path=trial_log_path,
            trial_context={"k": (k_hint or "K"), "fold": f"{y}", "phase_label": "GLOBAL", "seed": seed},
        )
        if atr_sweep:
            global_params = _sweep_atr_local(train_df, timeframe, global_params, loss_mult, atr_sweep_span, atr_sweep_step, mdd_max)

        completed_phases.append("GLOBAL")

        # Checkpoint after global optimization
        if checkpoint_callback is not None:
            try:
                checkpoint_callback(
                    fold=y,
                    phase="GLOBAL",
                    trial=n_trials,
                    total_trials=n_trials,
                    best_params=global_params,
                    best_score=0.0,  # Score not tracked here
                    completed_phases=completed_phases.copy(),
                )
            except Exception:
                pass

        for st in states:
            sub = train_df[lbl_train == st]
            # Require a minimal sample to optimize per-state; else fallback to global
            if len(sub) < 200:
                params_by_state[st] = dict(global_params)
            else:
                p = _optimize_on_train(
                    sub, timeframe, n_trials, seed, jobs, loss_mult,
                    trial_log_path=trial_log_path,
                    trial_context={"k": (k_hint or "K"), "fold": f"{y}", "phase_label": str(st), "seed": seed},
                )
                if atr_sweep:
                    p = _sweep_atr_local(sub, timeframe, p, loss_mult, atr_sweep_span, atr_sweep_step, mdd_max)
                params_by_state[st] = p

            completed_phases.append(str(st))

            # Checkpoint after each phase optimization
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(
                        fold=y,
                        phase=str(st),
                        trial=n_trials,
                        total_trials=n_trials,
                        best_params=params_by_state.get(st, global_params),
                        best_score=0.0,
                        completed_phases=completed_phases.copy(),
                    )
                except Exception:
                    pass

        # Evaluate on test by contiguous segments
        segs = _segments_from_labels(test_df.index, lbl_test)
        seg_metrics: List[SegmentMetrics] = []
        for s0, s1, st in segs:
            seg_slice = test_df.loc[(test_df.index >= s0) & (test_df.index <= s1)]
            if seg_slice.empty:
                continue
            p = params_by_state.get(st, global_params)
            # Pass confidence only during test (not optimization)
            _seg_conf = None
            if confidence_series is not None:
                try:
                    _seg_conf = confidence_series.reindex(seg_slice.index, method='ffill')
                except Exception:
                    _seg_conf = None
            m = _apply_on_segment(seg_slice, timeframe, p, loss_mult, confidence_series=_seg_conf)
            seg_metrics.append(SegmentMetrics(start=s0.strftime('%Y-%m-%d'), end=s1.strftime('%Y-%m-%d'), state=str(st), metrics=m))

        # Compose fold-level metrics by stitching segments
        eq_mult = 1.0
        trades_total = 0
        sharpe_vals: List[float] = []
        min_equity_global = 1.0
        cur_equity = 1.0
        for sm in seg_metrics:
            m = sm.metrics
            eq = float(m.get("equity_mult", 1.0))
            me = float(m.get("min_equity", 1.0))
            dd = float(m.get("max_drawdown", 0.0))  # not used to compose
            tr = int(m.get("trades", 0))
            sh = float(m.get("sharpe_proxy", 0.0))
            # Update equity and global min equity
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

        folds.append(FoldResult(
            period_label=f"{y}",
            train_range=(train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
            test_range=(test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')),
            params_by_state=params_by_state,
            segments=seg_metrics,
            metrics=overall_fold,
        ))
        # Update coarse progress at end of fold
        folds_done += 1
        if progress_path is not None:
            try:
                percent = max(0.0, min(100.0, 100.0 * float(folds_done) / float(max(1, folds_total))))
                payload = {
                    "folds_done": int(folds_done),
                    "folds_total": int(folds_total),
                    "trial": None,
                    "trials_total": int(n_trials),
                    "percent": float(round(percent, 2)),
                    "phase": "fold_complete"
                }
                tmp = str(progress_path) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, progress_path)
            except Exception:
                pass

    # Aggregate across folds (compound equity, accumulate trades, max drawdown via min-equity stitching)
    eq_mult_all = 1.0
    min_equity_global = 1.0
    cur_equity = 1.0
    trades_total = 0
    sharpes_all: List[float] = []
    for fr in folds:
        m = fr.metrics
        eq = float(m.get("equity_mult", 1.0))
        me = float(1.0 - float(m.get("max_drawdown", 0.0)))  # back to min_equity proxy
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
    }
    return folds, overall


def _shutdown_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global _global_runner
    print(f"\n[CHECKPOINT] Signal {signum} received, saving checkpoint...")
    if _global_runner is not None:
        try:
            _global_runner.shutdown()
            print("[CHECKPOINT] Checkpoint saved. Safe to restart.")
        except Exception as e:
            print(f"[CHECKPOINT] Warning: {e}")
    raise SystemExit(1)


def main() -> int:
    global _global_runner

    ap = argparse.ArgumentParser(description="BTC-only phase-aware WFA (annual) on fused 2h")
    ap.add_argument("--labels-csv", required=True, help="Path to labels CSV with columns: timestamp,label")
    ap.add_argument("--granularity", choices=["annual"], default="annual")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--loss-mult", type=float, default=3.0)
    ap.add_argument("--atr-sweep", action="store_true")
    ap.add_argument("--atr-sweep-span", type=float, default=1.0)
    ap.add_argument("--atr-sweep-step", type=float, default=0.2)
    ap.add_argument("--mdd-max", type=float, default=None)
    ap.add_argument("--use-fused", action="store_true")
    ap.add_argument("--out-dir", default="outputs/wfa_phase")
    ap.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint interval in minutes")
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint system")
    ap.add_argument("--confidence-sizing", action="store_true",
                    help="Enable confidence-based position sizing (1%% base + up to 1%% boost from P(bull))")
    ap.add_argument("--confidence-csv", default=None,
                    help="CSV with timestamp,p_bull columns. If not set, tries to read p_bull from --labels-csv")
    # Accept unknown args to be robust against wrapper scripts that append extra flags
    args, _unknown = ap.parse_known_args()

    if args.use_fused:
        os.environ["USE_FUSED_H2"] = "1"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint system
    use_checkpoint = RobustRunner is not None and not args.no_checkpoint
    runner: Optional[RobustRunner] = None

    if use_checkpoint:
        runner = RobustRunner(
            out_dir,
            checkpoint_interval_minutes=args.checkpoint_interval,
            min_free_gb=10.0,
        )
        _global_runner = runner

        # Register signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGINT, _shutdown_handler)
            signal.signal(signal.SIGTERM, _shutdown_handler)
        except Exception:
            pass  # Windows may not support all signals

        # Register atexit handler
        def _atexit_save():
            global _global_runner
            if _global_runner is not None:
                try:
                    _global_runner.shutdown()
                except Exception:
                    pass
        atexit.register(_atexit_save)

        # Check for resume state
        resume_state = runner.get_resume_state()
        resume_fold = None
        if resume_state:
            resume_fold = resume_state.get('fold')
            print(f"[CHECKPOINT] Resuming from: fold={resume_fold}, "
                  f"phase={resume_state.get('phase')}, trial={resume_state.get('trial')}")
        else:
            print(f"[CHECKPOINT] Starting fresh run with checkpoints every {args.checkpoint_interval} min")
    else:
        resume_fold = None
        if args.no_checkpoint:
            print("[CHECKPOINT] Disabled by --no-checkpoint flag")
        else:
            print("[CHECKPOINT] Not available (checkpoint_manager not found)")

    # Single-instance lock to avoid duplicate runs on same out_dir
    lock_path = out_dir / ".lock"
    try:
        if lock_path.exists():
            # If lock older than 24h, ignore; else exit
            try:
                mtime = lock_path.stat().st_mtime
                age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0
            except Exception:
                age_hours = 0.0
            if age_hours < 24.0:
                print(f"Another runner appears active for {out_dir}. Lock present: {lock_path}")
                return 2
        # create/refresh lock
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"))
    except Exception:
        pass

    # Load data and labels
    df = _load_btc_fused("2h")
    df = pipe.ensure_utc_index(df)

    labels_df = pd.read_csv(args.labels_csv, parse_dates=["timestamp"])  # type: ignore[arg-type]
    if "timestamp" not in labels_df.columns:
        raise RuntimeError(f"Missing 'timestamp' in {args.labels_csv}")
    if "label" not in labels_df.columns and "state" in labels_df.columns:
        labels_df = labels_df.rename(columns={"state": "label"})
    if "label" not in labels_df.columns:
        raise RuntimeError("Labels CSV must contain a 'label' column")
    labels = labels_df.set_index("timestamp").sort_index()["label"].astype(str)

    progress_path = out_dir / "PROGRESS.json"
    # Initialize progress file early (atomic)
    try:
        payload = {
            "folds_done": 0,
            "folds_total": 0,
            "trial": 0,
            "trials_total": int(args.trials),
            "percent": 0.0,
            "phase": "init",
            "checkpoint_enabled": use_checkpoint,
        }
        tmp = str(progress_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, progress_path)
    except Exception:
        pass
    # Prepare per-trial JSONL path for live heatmaps
    k_hint = None
    try:
        name = Path(args.labels_csv).stem  # e.g., K3
        if name.upper().startswith("K"):
            k_hint = name.upper()
    except Exception:
        pass
    jsonl_path: Optional[Path] = None
    try:
        jsonl_dir = ROOT / "outputs" / "trial_logs" / "phase" / (k_hint or "K")
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_dir / "trials_from_wfa.jsonl"
    except Exception:
        jsonl_path = None

    # Create checkpoint callback for run_phase_wfa
    def checkpoint_callback(fold: int, phase: str, trial: int, total_trials: int,
                           best_params: Dict[str, Any], best_score: float,
                           completed_phases: List[str]) -> None:
        if runner is not None:
            runner.save_progress(
                seed=args.seed,
                fold=fold,
                phase=phase,
                trial=trial,
                total_trials=total_trials,
                best_params=best_params,
                best_score=best_score,
                completed_phases=completed_phases,
            )

    # Load confidence series for position sizing (if enabled)
    _conf_series: Optional[pd.Series] = None
    if args.confidence_sizing:
        try:
            from src.confidence_sizing import load_confidence_from_labels
            # Try dedicated confidence CSV first, then labels CSV
            _conf_csv = args.confidence_csv or args.labels_csv
            _conf_series = load_confidence_from_labels(_conf_csv)
            if _conf_series is not None:
                print(f"[SIZING] Confidence-based sizing ACTIVE: {len(_conf_series)} P(bull) values from {_conf_csv}")
                print(f"[SIZING] Range: P(bull) min={_conf_series.min():.3f}, max={_conf_series.max():.3f}, mean={_conf_series.mean():.3f}")
            else:
                print(f"[SIZING] Warning: --confidence-sizing enabled but no p_bull column in {_conf_csv}")
                print(f"[SIZING] Falling back to fixed 1% sizing")
        except Exception as e:
            print(f"[SIZING] Error loading confidence: {e}. Using fixed 1% sizing.")

    folds, overall = run_phase_wfa(
        df,
        labels,
        args.granularity,
        int(args.trials),
        int(args.seed),
        int(args.jobs),
        float(args.loss_mult),
        bool(args.atr_sweep),
        float(args.atr_sweep_span),
        float(args.atr_sweep_step),
        (None if args.mdd_max is None else float(args.mdd_max)),
        progress_path=progress_path,
        trial_log_path=jsonl_path,
        k_hint=k_hint,
        checkpoint_callback=checkpoint_callback if use_checkpoint else None,
        confidence_series=_conf_series,
        resume_fold=resume_fold,
    )

    ts = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    k_suffix = (k_hint or "K").upper()
    out_json = out_dir / f"WFA_phase_{k_suffix}_BTC_fused_{ts}.json"

    payload = {
        "granularity": args.granularity,
        "trials": int(args.trials),
        "jobs": int(args.jobs),
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
                "segments": [
                    {
                        "start": sm.start,
                        "end": sm.end,
                        "state": sm.state,
                        "metrics": sm.metrics,
                    }
                    for sm in fr.segments
                ],
                "metrics": fr.metrics,
            }
            for fr in folds
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    try:
        print(
            f"Overall: equity x {overall['equity_mult']:.3f}, MDD {overall['max_drawdown']:.2%}, "
            f"trades {overall['trades']}, Sharpe~{overall['sharpe_proxy_mean']:.2f}"
        )
    except Exception:
        pass
    try:
        # Mark progress 100% at completion
        payload_done = {
            "folds_done": payload.get("overall", {}).get("folds", 0),
            "folds_total": payload.get("overall", {}).get("folds", 0),
            "trial": None,
            "trials_total": int(args.trials),
            "percent": 100.0,
            "phase": "complete",
        }
        tmp = str(progress_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload_done, f, ensure_ascii=False)
        os.replace(tmp, progress_path)
    except Exception:
        pass

    # Mark seed complete and cleanup
    if runner is not None:
        runner.complete_seed(args.seed, overall)
        runner.shutdown()
        _global_runner = None

    try:
        # remove lock at the end
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


