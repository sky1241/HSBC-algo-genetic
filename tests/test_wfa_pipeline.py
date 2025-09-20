from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import io_loader, stats_eval, wfa
import run_oos


def _make_mock_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    base = 100 + rng.standard_normal(len(index)).cumsum()
    high = base + rng.uniform(0.1, 1.0, size=len(index))
    low = base - rng.uniform(0.1, 1.0, size=len(index))
    open_ = base + rng.normal(0, 0.2, size=len(index))
    close = base
    volume = rng.uniform(100, 200, size=len(index))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def test_walk_forward_pipeline(tmp_path: Path) -> None:
    df = _make_mock_dataset()
    csv_path = tmp_path / "mock.csv"
    df.reset_index().to_csv(csv_path, index=False)
    loaded = io_loader.load_ohlcv_csv(csv_path)
    config = wfa.WalkForwardConfig(
        n_states=2,
        n_trials=5,
        welch_nperseg_grid=(64,),
        welch_noverlap=0.5,
        lfp_horizon_days=10.0,
        volatility_window=32,
        min_train_size=200,
        min_train_years=1,
        periods_per_year=365,
        fs_per_day=1.0,
    )
    result = wfa.run_walk_forward(loaded, seeds=[0, 1], config=config)
    assert not result.returns.empty
    assert {"phaseaware", "baseline"} <= set(result.returns["strategy"].unique())
    reports_dir = tmp_path / "reports"
    evaluation = stats_eval.evaluate_results(result.returns, 365, reports_dir)
    assert (reports_dir / "metrics_by_seed.csv").exists()
    assert not evaluation.summary.empty
    assert (evaluation.tests.empty or set(evaluation.tests["metric"]).issubset(stats_eval.METRIC_COLUMNS))


def test_run_oos_main(tmp_path: Path) -> None:
    df = _make_mock_dataset()
    csv_path = tmp_path / "mock.csv"
    df.reset_index().to_csv(csv_path, index=False)
    out_dir = tmp_path / "out"
    args = [
        "--data",
        str(csv_path),
        "--output",
        str(out_dir),
        "--seeds",
        "0",
        "1",
        "--n-states",
        "2",
        "--n-trials",
        "5",
        "--welch-nperseg",
        "64",
        "--welch-noverlap",
        "0.5",
        "--welch-window",
        "hann",
        "--lfp-horizon-days",
        "10.0",
        "--volatility-window",
        "32",
        "--fs-per-day",
        "1.0",
        "--periods-per-year",
        "365",
        "--start-year",
        "2019",
        "--end-year",
        "2022",
    ]
    exit_code = run_oos.main(args)
    assert exit_code == 0
    assert (out_dir / "wilcoxon_comparison.csv").exists()


def test_run_oos_skips_evaluation_when_returns_invalid(tmp_path: Path, capsys) -> None:
    df = _make_mock_dataset().loc["2018":"2019"]
    csv_path = tmp_path / "short.csv"
    df.reset_index().to_csv(csv_path, index=False)
    out_dir = tmp_path / "short_out"
    args = [
        "--data",
        str(csv_path),
        "--output",
        str(out_dir),
        "--seeds",
        "0",
        "--n-states",
        "2",
        "--n-trials",
        "1",
        "--welch-nperseg",
        "64",
        "--min-train-size",
        "5000",
        "--fs-per-day",
        "1.0",
        "--periods-per-year",
        "365",
    ]
    exit_code = run_oos.main(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "DataFrame des retours vide" in captured.out
    assert not (out_dir / "wilcoxon_comparison.csv").exists()
    skipped_path = out_dir / "skipped_folds.csv"
    assert skipped_path.exists()
    skipped_df = pd.read_csv(skipped_path)
    assert not skipped_df.empty
