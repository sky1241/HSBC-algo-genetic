"""Run a full walk-forward (out-of-sample) evaluation.

Example
-------
    python run_oos.py --data data/BTC_USDT_2h.csv --output outputs/wfa \
        --seeds 0 1 2 --n-states 3 --n-trials 30 --welch-nperseg 256 384 \
        --welch-noverlap 0.5 --lfp-horizon-days 8.0

The script glues together the IO helpers, Welch PSD feature extraction,
HMM-based regime identification, phase-aware optimisation and robust
statistics.  All outputs (CSV and plots) are stored in the chosen output
directory and are ready to be referenced from the README or research notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from src import features_fourier, io_loader, optimizer, regime_hmm, risk_sizing, stats_eval, wfa


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward OOS evaluation")
    parser.add_argument("--data", required=True, help="CSV OHLCV dataset (timestamp, open, high, low, close, volume)")
    parser.add_argument("--output", default="outputs/wfa", help="Destination directory for reports")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds for the WFA loop")
    parser.add_argument("--n-states", type=int, default=3, help="Number of HMM regimes")
    parser.add_argument("--n-trials", type=int, default=25, help="Random-search trials per phase")
    parser.add_argument(
        "--welch-nperseg",
        nargs="+",
        type=int,
        default=None,
        help="Candidate nperseg values (bars) for the Welch mini sweep",
    )
    parser.add_argument(
        "--welch-noverlap",
        type=float,
        default=0.5,
        help="Overlap ratio for Welch segments (0-0.9)",
    )
    parser.add_argument(
        "--welch-window",
        default="hann",
        help="Window function passed to scipy.signal.welch",
    )
    parser.add_argument(
        "--lfp-horizon-days",
        type=float,
        default=5.0,
        help="Low-frequency horizon in days to compute the LFP ratio",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=96,
        help="Window (bars) used for the volatility proxy",
    )
    parser.add_argument(
        "--fs-per-day",
        type=float,
        default=None,
        help="Override sampling frequency in bars per day (auto when omitted)",
    )
    parser.add_argument(
        "--price-col",
        default="close",
        help="Name of the column containing close prices",
    )
    parser.add_argument("--min-train-years", type=int, default=1, help="Minimum number of in-sample years before scoring")
    parser.add_argument("--min-train-size", type=int, default=200, help="Minimum number of samples in the training set")
    parser.add_argument("--start-year", type=int, default=None, help="First year to include in the analysis")
    parser.add_argument("--end-year", type=int, default=None, help="Last year to include in the analysis")
    parser.add_argument(
        "--baseline",
        nargs=5,
        type=float,
        metavar=("TENKAN", "KIJUN", "SENKOU_B", "SHIFT", "ATR_MULT"),
        help="Override baseline parameters (9 26 52 26 2.0)",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=None,
        help="Sampling frequency per year (auto-detected when omitted)",
    )
    return parser.parse_args(argv)


def _make_config(args: argparse.Namespace, periods_per_year: int) -> wfa.WalkForwardConfig:
    default_cfg = wfa.WalkForwardConfig()
    baseline = optimizer.BASELINE_PARAMS.copy()
    if args.baseline is not None:
        tenkan, kijun, senkou_b, shift, atr_mult = args.baseline
        baseline = {
            "tenkan": int(tenkan),
            "kijun": int(kijun),
            "senkou_b": int(senkou_b),
            "shift": int(shift),
            "atr_mult": float(atr_mult),
        }
    welch_grid = tuple(args.welch_nperseg) if args.welch_nperseg else default_cfg.welch_nperseg_grid
    return wfa.WalkForwardConfig(
        n_states=args.n_states,
        n_trials=args.n_trials,
        welch_nperseg_grid=welch_grid,
        welch_noverlap=args.welch_noverlap,
        welch_window=args.welch_window,
        lfp_horizon_days=args.lfp_horizon_days,
        volatility_window=args.volatility_window,
        fs_per_day=args.fs_per_day,
        price_col=args.price_col,
        min_train_size=args.min_train_size,
        min_train_years=args.min_train_years,
        start_year=args.start_year,
        end_year=args.end_year,
        baseline_params=baseline,
        periods_per_year=periods_per_year,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    data_path = Path(args.data)
    df = io_loader.load_ohlcv_csv(data_path)
    df = io_loader.ensure_ohlcv_dataframe(df)
    periods_per_year = args.periods_per_year or io_loader.infer_periods_per_year(df.index)
    config = _make_config(args, periods_per_year)
    fs_per_day = config.fs_per_day or features_fourier.estimate_fs_per_day(df.index)
    feature_cfg = features_fourier.FourierConfig(
        price_col=args.price_col,
        fs_per_day=fs_per_day,
        window=args.welch_window,
        nperseg_grid=tuple(args.welch_nperseg) if args.welch_nperseg else config.welch_nperseg_grid,
        noverlap_ratio=args.welch_noverlap,
        lfp_horizon_days=args.lfp_horizon_days,
        volatility_window=args.volatility_window,
    )
    hmm_cfg = regime_hmm.HMMConfig(n_states=args.n_states)
    print(f"Fourier config: {feature_cfg}")
    print(f"HMM config: {hmm_cfg}")
    preview = risk_sizing.simulate_strategy(df.head(min(200, len(df))), config.baseline_params)
    print(f"Baseline risk sizing preview: {preview.head(3).to_list()}")
    result = wfa.run_walk_forward(df, args.seeds, config)
    evaluation = stats_eval.evaluate_results(result.returns, periods_per_year, args.output)
    summary = evaluation.summary
    global_summary = summary[summary["phase"] == "global"]
    if not global_summary.empty:
        print("\nRésumé global (médiane / IQR):")
        for _, row in global_summary.iterrows():
            print(
                f"- {row['strategy']} {row['metric']}: median={row['median']:.4f} "
                f"iqr={row['iqr']:.4f} (n={int(row['count'])})"
            )
    else:
        print("Aucun résultat disponible pour la synthèse globale.")
    print(f"\nRapports sauvegardés dans: {Path(args.output).resolve()}")
    if result.skipped_folds:
        skipped_df = pd.DataFrame(result.skipped_folds)
        skipped_path = Path(args.output) / "skipped_folds.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Folds ignorés: {len(skipped_df)} (voir {skipped_path})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
