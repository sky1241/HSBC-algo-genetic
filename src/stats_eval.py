"""Statistical evaluation helpers for walk-forward analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

METRIC_COLUMNS = [
    "sharpe", "calmar", "cagr", "mdd", "per_month",
    "sortino", "ulcer", "dd_duration",
]

# Constante d'Euler-Mascheroni — utilisée par expected_max_sharpe_under_h0.
EULER_MASCHERONI = 0.5772156649015329


@dataclass(slots=True)
class EvaluationResult:
    metrics: pd.DataFrame
    metrics_long: pd.DataFrame
    monthly_returns: pd.DataFrame
    summary: pd.DataFrame
    tests: pd.DataFrame


def compute_metrics(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Compute Sharpe, Sortino, Calmar, CAGR, MDD, ulcer index, DD duration, mean monthly return."""

    if returns is None or len(returns) == 0:
        return {metric: float("nan") for metric in METRIC_COLUMNS}
    returns = returns.dropna()
    if returns.empty:
        return {metric: float("nan") for metric in METRIC_COLUMNS}
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = float(np.sqrt(periods_per_year) * mean / std) if std > 0 else float("nan")
    # Sortino: only downside std (Bailey/Estrada). Numérateur = mean - target (target=0 ici).
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        sortino = float(np.sqrt(periods_per_year) * mean / downside_std) if downside_std > 0 else float("nan")
    else:
        sortino = float("inf") if mean > 0 else float("nan")
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
    # Ulcer index = sqrt(mean(drawdown^2)) — pénalise drawdowns longs (Martin 1989).
    ulcer = float(np.sqrt(np.mean(drawdown ** 2))) if not drawdown.empty else float("nan")
    # Drawdown duration max (en périodes): plus longue série consécutive sous le peak.
    if not drawdown.empty:
        in_dd = (drawdown < 0).to_numpy()
        max_run, run = 0, 0
        for d in in_dd:
            if d:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        dd_duration = float(max_run)
    else:
        dd_duration = float("nan")
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
        "sortino": sortino,
        "ulcer": ulcer,
        "dd_duration": dd_duration,
    }


def probabilistic_sharpe_ratio(
    sharpe_observed: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> float:
    """PSR — Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    Probabilité que le VRAI SR (population) soit > sharpe_benchmark, sachant le SR
    observé sur un échantillon de n_obs points avec skew/kurt non-normaux.

    Tous les SR doivent être dans la même unité (typiquement par-période, pas annualisés —
    l'unité doit cancel dans le ratio diff / sqrt(var_sr) tant que les inputs sont cohérents).

    Args:
        sharpe_observed: SR estimé.
        n_obs: T, taille d'échantillon.
        skew: γ_3 des returns (default 0 = normal).
        kurt: γ_4 non-excess (default 3 = normal).
        sharpe_benchmark: SR* seuil. Default 0.

    Returns: probabilité ∈ [0, 1]. > 0.95 = significatif.
    """
    if n_obs <= 1:
        return float("nan")
    diff = sharpe_observed - sharpe_benchmark
    var_sr = (
        1.0
        - skew * sharpe_observed
        + (kurt - 1.0) / 4.0 * sharpe_observed ** 2
    ) / (n_obs - 1)
    if var_sr <= 0:
        return float("nan")
    z = diff / np.sqrt(var_sr)
    return float(norm.cdf(z))


def expected_max_sharpe_under_h0(n_trials: int, var_sr_iid: float = 1.0) -> float:
    """Sharpe attendu MAX sous H0 (true SR=0) pour N essais i.i.d.

    Bailey & López de Prado 2014, eq. (5):
        E[max{SR_n}] ≈ √V · ((1−γ_em)·Φ⁻¹(1−1/N) + γ_em·Φ⁻¹(1−1/(N·e)))

    Args:
        n_trials: N essais (e.g. nombre de trials Optuna / combinaisons de params testés).
        var_sr_iid: variance des SR sous H0. Si tu n'as pas mieux, 1.0.
    """
    if n_trials <= 1:
        return 0.0
    sigma = float(np.sqrt(var_sr_iid))
    term1 = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma * (term1 + term2))


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_obs: int,
    n_trials: int,
    var_sr_trials: float = 1.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Probabilité que le vrai SR > 0 APRÈS correction pour data snooping
    (n_trials configurations testées). Avec trials massifs (Optuna ≫ 100), le Sharpe
    rapporté est biaisé vers le haut — DSR donne une significativité honnête.

    Args:
        sharpe_observed: SR du modèle GAGNANT.
        n_obs: longueur série de returns du gagnant.
        n_trials: nombre de configurations testées au total (Optuna trials, grid search, etc.).
        var_sr_trials: variance des SR observés à travers les N trials.
            Si non-disponible, 1.0 reste raisonnable mais conservateur.
        skew, kurt: moments du modèle gagnant.

    Returns: probabilité ∈ [0, 1]. > 0.95 = on rejette l'hypothèse "edge dû au hasard".
    """
    sr_thr = expected_max_sharpe_under_h0(n_trials, var_sr_trials)
    return probabilistic_sharpe_ratio(
        sharpe_observed=sharpe_observed,
        n_obs=n_obs,
        skew=skew,
        kurt=kurt,
        sharpe_benchmark=sr_thr,
    )


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
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe_under_h0",
    "deflated_sharpe_ratio",
]
