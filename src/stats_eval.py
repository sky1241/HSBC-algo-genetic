"""Statistical evaluation helpers for walk-forward analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRIC_COLUMNS = ["sharpe", "calmar", "cagr", "mdd", "per_month"]


@dataclass(slots=True)
class EvaluationResult:
    metrics: pd.DataFrame
    metrics_long: pd.DataFrame
    monthly_returns: pd.DataFrame
    summary: pd.DataFrame
    tests: pd.DataFrame


def compute_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Compute Sharpe, Calmar, CAGR, MDD and mean monthly return."""

    if returns is None or len(returns) == 0:
        return {metric: float("nan") for metric in METRIC_COLUMNS}
    returns = returns.dropna()
    if returns.empty:
        return {metric: float("nan") for metric in METRIC_COLUMNS}
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = float(np.sqrt(periods_per_year) * mean / std) if std > 0 else float("nan")
    equity = (1.0 + returns).cumprod()
    final = equity.iloc[-1]
    periods = len(returns)
    years = periods / float(periods_per_year) if periods_per_year > 0 else float("nan")
    if final <= 0 or years <= 0:
        cagr = float("nan")
    else:
        cagr = float(final ** (1.0 / years) - 1.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    mdd = float(drawdown.min()) if not drawdown.empty else float("nan")
    calmar = float(cagr / abs(mdd)) if mdd < 0 and np.isfinite(cagr) else float("nan")
    if isinstance(returns.index, pd.DatetimeIndex):
        monthly = (1.0 + returns).resample("ME").prod() - 1.0
        if not monthly.empty:
            monthly = monthly.tz_localize(None)
        per_month = float(monthly.mean()) if not monthly.empty else float("nan")
    else:
        per_month = float("nan")
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "cagr": cagr,
        "mdd": mdd,
        "per_month": per_month,
    }


def compute_monthly_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return monthly
    if monthly.index.tz is not None:
        monthly.index = monthly.index.tz_convert(None).tz_localize(None)
    else:
        monthly.index = monthly.index.tz_localize(None)
    monthly.index = monthly.index.to_period("M")
    return monthly


def _hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x) - np.asarray(y)
    if diff.size == 0:
        return float("nan")
    return float(np.median(diff))


def build_metrics_table(returns: pd.DataFrame, periods_per_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    monthly_records: list[dict[str, object]] = []
    for (strategy, seed, phase), group in returns.groupby(["strategy", "seed", "phase"]):
        series = group.sort_values("timestamp").set_index("timestamp")["return"].astype(float)
        metrics = compute_metrics(series, periods_per_year)
        record = {"strategy": strategy, "seed": seed, "phase": phase}
        record.update(metrics)
        records.append(record)
        monthly = compute_monthly_returns(series)
        for ts, value in monthly.items():
            monthly_records.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "phase": phase,
                    "month": ts,
                    "return": float(value),
                }
            )
    for (strategy, seed), group in returns.groupby(["strategy", "seed"]):
        series = group.sort_values("timestamp").set_index("timestamp")["return"].astype(float)
        metrics = compute_metrics(series, periods_per_year)
        record = {"strategy": strategy, "seed": seed, "phase": "global"}
        record.update(metrics)
        records.append(record)
        monthly = compute_monthly_returns(series)
        for ts, value in monthly.items():
            monthly_records.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "phase": "global",
                    "month": ts,
                    "return": float(value),
                }
            )
    metrics_df = pd.DataFrame.from_records(records)
    monthly_df = pd.DataFrame.from_records(monthly_records)
    return metrics_df, monthly_df


def aggregate_metrics(metrics_long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_long.groupby(["strategy", "phase", "metric"])  # type: ignore[arg-type]
        .agg(
            median=("value", "median"),
            q1=("value", lambda x: x.quantile(0.25)),
            q3=("value", lambda x: x.quantile(0.75)),
            iqr=("value", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            count=("value", "count"),
        )
        .reset_index()
    )
    return summary


def compare_strategies(metrics_long: pd.DataFrame, phase: str = "global") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategies = metrics_long["strategy"].unique()
    if "phaseaware" not in strategies or "baseline" not in strategies:
        return pd.DataFrame(columns=["metric", "wilcoxon_stat", "wilcoxon_pvalue", "hodges_lehmann", "n"])
    for metric in METRIC_COLUMNS:
        pa = metrics_long[
            (metrics_long["strategy"] == "phaseaware")
            & (metrics_long["phase"] == phase)
            & (metrics_long["metric"] == metric)
        ]
        bl = metrics_long[
            (metrics_long["strategy"] == "baseline")
            & (metrics_long["phase"] == phase)
            & (metrics_long["metric"] == metric)
        ]
        merged = pa.merge(bl, on="seed", suffixes=("_pa", "_bl"))
        if merged.empty:
            continue
        stat, pvalue = wilcoxon(merged["value_pa"], merged["value_bl"], zero_method="wilcox")
        hl = _hodges_lehmann(merged["value_pa"].to_numpy(), merged["value_bl"].to_numpy())
        rows.append(
            {
                "metric": metric,
                "wilcoxon_stat": float(stat),
                "wilcoxon_pvalue": float(pvalue),
                "hodges_lehmann": hl,
                "n": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def _plot_boxplots(metrics_long: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_metrics = metrics_long[metrics_long["phase"] == "global"]
    for metric in METRIC_COLUMNS:
        subset = global_metrics[global_metrics["metric"] == metric]
        if subset.empty:
            continue
        data = [
            subset.loc[subset["strategy"] == strategy, "value"].dropna().to_numpy()
            for strategy in ["phaseaware", "baseline"]
        ]
        labels = ["phaseaware", "baseline"]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, tick_labels=labels, notch=True)
        plt.title(f"Distribution {metric} (global)")
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{metric}.png", dpi=150)
        plt.close()


def evaluate_results(
    returns: pd.DataFrame,
    periods_per_year: int,
    output_dir: str | Path,
) -> EvaluationResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_df, monthly_df = build_metrics_table(returns, periods_per_year)
    metrics_long = metrics_df.melt(
        id_vars=["strategy", "seed", "phase"],
        value_vars=[col for col in METRIC_COLUMNS if col in metrics_df.columns],
        var_name="metric",
        value_name="value",
    )
    summary_df = aggregate_metrics(metrics_long)
    tests_df = compare_strategies(metrics_long)
    metrics_df.to_csv(output_path / "metrics_by_seed.csv", index=False)
    metrics_long.to_csv(output_path / "metrics_long.csv", index=False)
    summary_df.to_csv(output_path / "summary_median_iqr.csv", index=False)
    tests_df.to_csv(output_path / "wilcoxon_comparison.csv", index=False)
    monthly_df.to_csv(output_path / "monthly_returns.csv", index=False)
    _plot_boxplots(metrics_long, output_path)
    return EvaluationResult(
        metrics=metrics_df,
        metrics_long=metrics_long,
        monthly_returns=monthly_df,
        summary=summary_df,
        tests=tests_df,
    )


__all__ = [
    "EvaluationResult",
    "compute_metrics",
    "compute_monthly_returns",
    "build_metrics_table",
    "aggregate_metrics",
    "compare_strategies",
    "evaluate_results",
]
