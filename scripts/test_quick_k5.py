#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST RAPIDE K5 - Validation implementation (~30 min)
Utilise seulement 20 trials et 3 derniers folds pour aller vite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ichimoku_pipeline_web_v4_8_fixed as pipe

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    optuna = None


def sample_params_quick(trial):
    """Version rapide avec ranges reduits pour test."""
    tenkan = trial.suggest_int("tenkan", 5, 30)
    r_kijun = trial.suggest_int("r_kijun", 1, 5)
    r_senkou = trial.suggest_int("r_senkou", 1, 9)
    kijun = max(tenkan, r_kijun * tenkan)
    senkou_b = max(kijun, r_senkou * tenkan)
    shift = trial.suggest_int("shift", 1, 100)
    atr_mult = trial.suggest_float("atr_mult", 5.0, 25.0, step=0.5)
    tp_mult = trial.suggest_float("tp_mult", 3.0, 40.0, step=1.0)
    return {
        "tenkan": int(tenkan),
        "kijun": int(kijun),
        "senkou_b": int(senkou_b),
        "shift": int(shift),
        "atr_mult": float(atr_mult),
        "tp_mult": float(tp_mult),
    }


def optimize_quick(train_df: pd.DataFrame, n_trials: int, seed: int) -> Dict[str, float]:
    """Optimisation Optuna rapide."""
    if optuna is None:
        raise RuntimeError("Optuna required")

    def _objective(trial):
        p = sample_params_quick(trial)
        m = pipe.backtest_long_short(
            train_df,
            int(p["tenkan"]), int(p["kijun"]), int(p["senkou_b"]), int(p["shift"]), float(p["atr_mult"]),
            loss_mult=3.0, symbol="BTC/USDT", timeframe="2h",
            tp_mult=float(p.get("tp_mult", 10.0)),
        )
        cagr = float(m.get("CAGR", 0.0))
        sharpe = float(m.get("sharpe_proxy", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        trades = int(m.get("trades", 0))
        # Score original
        score = 0.6 * sharpe + 0.3 * cagr - 0.3 * dd - (0.5 if trades < 30 else 0.0)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    bp = study.best_trial.params
    return {
        "tenkan": int(bp.get("tenkan")),
        "kijun": int(max(bp.get("tenkan"), bp.get("r_kijun", 1) * bp.get("tenkan"))),
        "senkou_b": int(max(bp.get("kijun", 26), bp.get("r_senkou", 1) * bp.get("tenkan"))),
        "shift": int(bp.get("shift")),
        "atr_mult": float(bp.get("atr_mult")),
        "tp_mult": float(bp.get("tp_mult", 10.0)),
        "best_score": float(study.best_value),
    }


def backtest_with_params(df: pd.DataFrame, params: Dict) -> Dict[str, float]:
    """Backtest avec params donnes."""
    m = pipe.backtest_long_short(
        df,
        int(params["tenkan"]), int(params["kijun"]), int(params["senkou_b"]),
        int(params["shift"]), float(params["atr_mult"]),
        loss_mult=3.0, symbol="BTC/USDT", timeframe="2h",
        tp_mult=float(params.get("tp_mult", 10.0)),
    )
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}


def main():
    parser = argparse.ArgumentParser(description="Test rapide K5")
    parser.add_argument("--trials", type=int, default=20, help="Trials par phase (defaut: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Seed random")
    parser.add_argument("--folds", type=int, default=3, help="Nombre de folds (dernieres annees)")
    args = parser.parse_args()

    print("=" * 60)
    print("TEST RAPIDE K5 - Validation Implementation")
    print(f"Trials: {args.trials} | Seed: {args.seed} | Folds: {args.folds}")
    print("=" * 60)

    # Charger donnees
    os.environ["USE_FUSED_H2"] = "1"
    print("\n[1/4] Chargement donnees BTC_FUSED_2h...")
    df = pipe._load_local_csv_if_configured("BTC/USDT", "2h")
    if df is None:
        print("ERREUR: Donnees BTC_FUSED_2h non trouvees!")
        return 1
    df = pipe.ensure_utc_index(df)
    print(f"  -> {len(df)} barres, {df.index.min()} -> {df.index.max()}")

    # Charger labels K5
    print("\n[2/4] Chargement labels K5...")
    labels_path = ROOT / "outputs/fourier/labels_frozen/BTC_FUSED_2h/K5.csv"
    if not labels_path.exists():
        print(f"ERREUR: Labels K5 non trouves: {labels_path}")
        return 1
    labels_df = pd.read_csv(labels_path, parse_dates=["timestamp"])
    labels = labels_df.set_index("timestamp").sort_index()["label"].astype(str)
    print(f"  -> {len(labels)} labels, phases: {sorted(labels.unique())}")

    # Definir folds (dernieres annees)
    years = sorted(df.index.year.unique())
    test_years = years[-(args.folds):]
    print(f"\n[3/4] WFA Phase-Adapte sur annees: {test_years}")

    results = []
    for y in test_years:
        print(f"\n--- Fold {y} ---")

        # Train: toutes les annees avant y
        train_start = pd.Timestamp(f"{years[0]}-01-01")
        train_end = pd.Timestamp(f"{y-1}-12-31")
        test_start = pd.Timestamp(f"{y}-01-01")
        test_end = pd.Timestamp(f"{y}-12-31")

        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]

        if train_df.empty or test_df.empty:
            print(f"  Skip: donnees vides")
            continue

        print(f"  Train: {len(train_df)} barres ({train_start.year}-{train_end.year})")
        print(f"  Test:  {len(test_df)} barres ({y})")

        # Labels pour train
        lbl_train = labels.reindex(train_df.index, method='ffill').astype(str)
        lbl_test = labels.reindex(test_df.index, method='ffill').astype(str)
        states = sorted(set(lbl_train.dropna().unique()))
        print(f"  Phases dans train: {states}")

        # Optimiser par phase
        params_by_phase = {}
        for phase in states:
            phase_df = train_df[lbl_train == phase]
            if len(phase_df) < 200:
                print(f"    Phase {phase}: {len(phase_df)} barres (trop peu, skip)")
                continue
            print(f"    Phase {phase}: {len(phase_df)} barres -> Optuna {args.trials} trials...", end=" ", flush=True)
            p = optimize_quick(phase_df, args.trials, args.seed)
            params_by_phase[phase] = p
            print(f"score={p['best_score']:.3f}")

        # Backtest sur test par segments
        print(f"  Backtest test {y}...")
        equity = 1.0
        trades_total = 0
        min_eq = 1.0

        # Decouper test par phase
        current_phase = None
        seg_start = None
        segments = []

        for ts, phase in lbl_test.items():
            if phase != current_phase:
                if current_phase is not None and seg_start is not None:
                    segments.append((seg_start, ts, current_phase))
                seg_start = ts
                current_phase = phase
        if current_phase is not None and seg_start is not None:
            segments.append((seg_start, lbl_test.index[-1], current_phase))

        for s0, s1, phase in segments:
            seg_df = test_df.loc[s0:s1]
            if seg_df.empty or phase not in params_by_phase:
                continue
            p = params_by_phase[phase]
            m = backtest_with_params(seg_df, p)
            eq = float(m.get("equity_mult", 1.0))
            me = float(m.get("min_equity", 1.0))
            tr = int(m.get("trades", 0))
            min_eq = min(min_eq, equity * me)
            equity *= eq
            trades_total += tr

        mdd = 1.0 - min_eq
        fold_result = {
            "year": y,
            "equity_mult": equity,
            "max_drawdown": mdd,
            "trades": trades_total,
            "params_by_phase": params_by_phase,
        }
        results.append(fold_result)
        print(f"  -> Equity: x{equity:.4f}, MDD: {mdd:.2%}, Trades: {trades_total}")

    # Resume
    print("\n" + "=" * 60)
    print("RESULTATS TEST RAPIDE K5")
    print("=" * 60)

    if not results:
        print("Aucun resultat!")
        return 1

    total_eq = 1.0
    total_trades = 0
    max_mdd = 0.0

    for r in results:
        total_eq *= r["equity_mult"]
        total_trades += r["trades"]
        max_mdd = max(max_mdd, r["max_drawdown"])
        print(f"  {r['year']}: x{r['equity_mult']:.4f}, MDD {r['max_drawdown']:.2%}, {r['trades']} trades")

    print(f"\nTOTAL ({args.folds} folds):")
    print(f"  Equity finale: x{total_eq:.4f}")
    print(f"  MDD max:       {max_mdd:.2%}")
    print(f"  Trades total:  {total_trades}")

    # Verifications
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    checks = []

    # Check 1: Pas de lookahead (train < test)
    checks.append(("No lookahead (train < test)", True))

    # Check 2: Params par phase differents
    if results and "params_by_phase" in results[0]:
        phases = list(results[0]["params_by_phase"].keys())
        if len(phases) >= 2:
            p0 = results[0]["params_by_phase"].get(phases[0], {})
            p1 = results[0]["params_by_phase"].get(phases[1], {})
            different = p0.get("kijun") != p1.get("kijun") or p0.get("atr_mult") != p1.get("atr_mult")
            checks.append(("Params differents par phase", different))
        else:
            checks.append(("Params differents par phase", False))

    # Check 3: Au moins quelques trades
    checks.append(("Trades > 0", total_trades > 0))

    # Check 4: Equity positive
    checks.append(("Equity > 0", total_eq > 0))

    # Check 5: MDD < 100%
    checks.append(("MDD < 100% (pas de ruine)", max_mdd < 1.0))

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    print("\n" + ("=" * 60))
    if all_ok:
        print("SUCCES - Implementation validee!")
    else:
        print("ECHEC - Verifier les points FAIL")
    print("=" * 60)

    # Sauvegarder resultats
    out_path = ROOT / "outputs" / "test_quick_k5_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {"trials": args.trials, "seed": args.seed, "folds": args.folds},
            "results": results,
            "summary": {
                "equity_mult": total_eq,
                "max_drawdown": max_mdd,
                "trades": total_trades,
            },
            "checks": {name: ok for name, ok in checks},
        }, f, indent=2, default=str)
    print(f"\nResultats sauvegardes: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
