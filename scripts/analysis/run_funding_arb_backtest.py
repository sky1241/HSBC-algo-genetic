#!/usr/bin/env python3
"""STRAT-A — Funding rate arb delta-neutral backtest on BTC perp.

Outputs (under ``outputs/strat_funding_arb_2026-04-26/``):
    - metrics_summary.csv        : table threshold x fold x metrics
    - metrics_aggregate.csv      : median/mean per threshold
    - equity_curves.png          : 1 ligne par threshold + baselines (HODL spot)
    - returns_components.csv     : breakdown funding / fees / borrow
    - RAPPORT.md                 : verdict honnête
    - recommendation.json        : déploiement OUI/NON

Usage:
    .venv/bin/python -m scripts.analysis.run_funding_arb_backtest \
        --data data/BTC_USDT_2h.csv \
        --funding data/funding_rate_BTCUSDT.csv \
        --start 2020-01-01 --end 2025-08-01 \
        --folds 5

Pivot post-ALPHA-2 (cf RAPPORT_ALPHA_FINAL_2026-04-26.md): Ichimoku K3 nu
Sharpe -1.91 réfutée. On teste un edge crypto-spécifique: collecter le
funding quand longs payent shorts (delta-neutral spot+perp).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src import funding_arb as FA
from src import reality_check as RC
from src import stats_eval as SE


PERIODS_PER_YEAR_H2 = FA.H2_PERIODS_PER_YEAR
DEFAULT_OUTDIR = Path("outputs/strat_funding_arb_2026-04-26")
DEFAULT_THRESHOLDS_BPS = (3.0, 5.0, 7.0, 10.0, 15.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.loc[start:end].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_funding(path: Path, df: pd.DataFrame) -> pd.Series:
    funding_raw = pd.read_csv(path, parse_dates=["timestamp"])
    if funding_raw["timestamp"].dt.tz is not None:
        funding_raw["timestamp"] = funding_raw["timestamp"].dt.tz_localize(None)
    aligned = FA.align_funding_to_h2(funding_raw, df)
    return aligned


# ---------------------------------------------------------------------------
# Folds (anchored expanding window with rolling test)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(df: pd.DataFrame, n_folds: int = 5, train_frac: float = 0.4) -> list[Fold]:
    """Anchored expanding-window folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n = len(df)
    test_size = int(n * (1.0 - train_frac) / n_folds)
    if test_size < 100:
        raise ValueError(f"test window too small ({test_size} bars)")
    train_size = n - n_folds * test_size
    folds: list[Fold] = []
    for i in range(n_folds):
        tr_s = df.index[0]
        tr_e = df.index[train_size + i * test_size - 1]
        te_s = df.index[train_size + i * test_size]
        te_e_idx = min(train_size + (i + 1) * test_size - 1, n - 1)
        te_e = df.index[te_e_idx]
        folds.append(Fold(i, tr_s, tr_e, te_s, te_e))
    return folds


# ---------------------------------------------------------------------------
# Per-fold metrics
# ---------------------------------------------------------------------------

def fold_metrics(rets: pd.Series, n_trials: int) -> dict[str, float]:
    base = SE.compute_metrics(rets, PERIODS_PER_YEAR_H2)
    n_obs = int(rets.dropna().shape[0])
    sr = base.get("sharpe", float("nan"))
    skew = float(rets.skew()) if n_obs > 3 else 0.0
    kurt = float(rets.kurtosis() + 3.0) if n_obs > 3 else 3.0
    if np.isfinite(sr) and n_obs > 1:
        sr_per = sr / np.sqrt(PERIODS_PER_YEAR_H2)
        dsr = SE.deflated_sharpe_ratio(
            sharpe_observed=sr_per,
            n_obs=n_obs,
            n_trials=n_trials,
            var_sr_trials=1.0 / PERIODS_PER_YEAR_H2,
            skew=skew,
            kurt=kurt,
        )
    else:
        dsr = float("nan")
    out = dict(base)
    out["dsr"] = float(dsr) if dsr is not None else float("nan")
    out["n_obs"] = n_obs
    out["total_return"] = float((1.0 + rets.fillna(0.0)).prod() - 1.0)
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def hodl_spot_returns(df: pd.DataFrame) -> pd.Series:
    """HODL spot BTC: return = pct change of close."""
    return df["close"].pct_change().fillna(0.0)


def hodl_perp_returns(df: pd.DataFrame, funding: pd.Series) -> pd.Series:
    """HODL perp long: spot return - funding paid (when funding > 0)."""
    rets = df["close"].pct_change().fillna(0.0)
    # Long perp pays funding when > 0, receives when < 0
    funding_cost = pd.Series(0.0, index=df.index)
    funding_bars = df.index.map(lambda ts: ts.hour in FA.FUNDING_HOURS_UTC and ts.minute == 0)
    funding_cost[funding_bars] = -funding.where(funding_bars, 0.0).fillna(0.0)
    return rets + funding_cost


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    funding: pd.Series,
    thresholds_bps: tuple[float, ...],
    n_folds: int,
    train_frac: float,
    fee_taker_bps: float,
    spot_borrow_apr: float,
    holding_min_periods: int,
    allow_short_spot: bool,
    outdir: Path,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # Drop bars with funding NaN (early period before funding history began).
    mask = funding.notna()
    df = df.loc[mask]
    funding = funding.loc[mask]
    print(f"After funding-NaN filter: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # 1. Pre-compute returns for each threshold over the full series.
    full_rets: dict[float, pd.Series] = {}
    full_components: dict[float, dict[str, pd.Series]] = {}
    full_meta: dict[float, dict] = {}
    for thr in thresholds_bps:
        res = FA.simulate_funding_arb(
            df, funding,
            fee_taker_bps=fee_taker_bps,
            spot_short_borrow_apr=spot_borrow_apr,
            threshold_bps=thr,
            holding_min_periods=holding_min_periods,
            allow_short_spot=allow_short_spot,
        )
        full_rets[thr] = res.returns
        full_components[thr] = {
            "funding": res.funding_pnl,
            "fees": res.fees_pnl,
            "borrow": res.borrow_pnl,
            "position": res.position,
        }
        full_meta[thr] = {
            "n_trades": res.n_trades,
            "total_funding_received": res.total_funding_received,
            "total_fees": res.total_fees,
            "total_borrow": res.total_borrow,
            **res.metadata,
        }

    # 2. Walk-forward folds
    folds = make_folds(df, n_folds=n_folds, train_frac=train_frac)
    rows: list[dict] = []
    fold_oos: dict[float, list[pd.Series]] = {thr: [] for thr in thresholds_bps}
    n_trials = len(thresholds_bps)
    for fold in folds:
        for thr in thresholds_bps:
            r = full_rets[thr].loc[fold.test_start:fold.test_end]
            m = fold_metrics(r, n_trials=n_trials)
            row = {
                "threshold_bps": thr,
                "fold": fold.fold_id,
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                **m,
            }
            rows.append(row)
            fold_oos[thr].append(r)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "metrics_summary.csv", index=False)

    # 3. Aggregate per threshold
    metric_cols = [
        "sharpe", "dsr", "sortino", "calmar", "cagr", "mdd", "ulcer",
        "dd_duration", "total_return",
    ]
    agg_rows = []
    for thr in thresholds_bps:
        sub = metrics_df[metrics_df["threshold_bps"] == thr]
        agg = {"threshold_bps": thr, "n_folds": int(len(sub))}
        for c in metric_cols:
            agg[c + "_median"] = float(sub[c].median())
            agg[c + "_mean"] = float(sub[c].mean())
        # Append meta
        m = full_meta[thr]
        agg["full_n_trades"] = m["n_trades"]
        agg["full_total_funding"] = m["total_funding_received"]
        agg["full_total_fees"] = m["total_fees"]
        agg["full_total_borrow"] = m["total_borrow"]
        agg["full_n_bars_short_perp"] = m["n_bars_long_spot"]  # signal -1 = short perp
        agg["full_n_bars_long_perp"] = m["n_bars_short_spot"]
        agg["full_n_bars_flat"] = m["n_bars_flat"]
        agg_rows.append(agg)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(outdir / "metrics_aggregate.csv", index=False)

    # 4. Chained OOS returns + Hansen SPA
    chained: dict[float, pd.Series] = {}
    for thr in thresholds_bps:
        chained[thr] = pd.concat(fold_oos[thr]).sort_index()
    common_idx = chained[thresholds_bps[0]].index
    for thr in thresholds_bps[1:]:
        common_idx = common_idx.intersection(chained[thr].index)
    common_idx = common_idx.sort_values()

    # Baselines on common_idx
    hodl_s = hodl_spot_returns(df).reindex(common_idx).fillna(0.0)
    hodl_p = hodl_perp_returns(df, funding).reindex(common_idx).fillna(0.0)
    flat_zero = pd.Series(0.0, index=common_idx)

    baseline_metrics = {
        "hodl_spot": SE.compute_metrics(hodl_s, PERIODS_PER_YEAR_H2),
        "hodl_perp_long": SE.compute_metrics(hodl_p, PERIODS_PER_YEAR_H2),
    }

    # Hansen SPA: K thresholds (excess returns vs zero-return benchmark = "flat")
    if len(common_idx) < 50:
        spa = None
    else:
        excess = np.column_stack([
            chained[thr].reindex(common_idx).fillna(0.0).to_numpy()
            for thr in thresholds_bps
        ])
        spa = RC.hansen_spa_test(excess, n_bootstrap=1000, block_size_mean=20.0, seed=42)

    # 5. DSR on chained OOS for the BEST threshold (by chained Sharpe)
    chained_metrics: dict[float, dict] = {}
    for thr in thresholds_bps:
        r = chained[thr].reindex(common_idx).fillna(0.0)
        chained_metrics[thr] = SE.compute_metrics(r, PERIODS_PER_YEAR_H2)
    best_thr = max(thresholds_bps, key=lambda t: chained_metrics[t].get("sharpe", -np.inf) or -np.inf)
    best_chain_rets = chained[best_thr].reindex(common_idx).fillna(0.0)
    sr_best = chained_metrics[best_thr]["sharpe"]
    if np.isfinite(sr_best):
        sr_per = sr_best / np.sqrt(PERIODS_PER_YEAR_H2)
        skew = float(best_chain_rets.skew())
        kurt = float(best_chain_rets.kurtosis() + 3.0)
        dsr_best = SE.deflated_sharpe_ratio(
            sharpe_observed=sr_per,
            n_obs=int(len(best_chain_rets)),
            n_trials=len(thresholds_bps),
            var_sr_trials=1.0 / PERIODS_PER_YEAR_H2,
            skew=skew,
            kurt=kurt,
        )
    else:
        dsr_best = float("nan")

    # 6. Equity curves PNG
    plt.figure(figsize=(11, 6))
    for thr in thresholds_bps:
        ec = (1.0 + chained[thr].reindex(common_idx).fillna(0.0)).cumprod()
        plt.plot(ec.index, ec.values, label=f"thr={thr}bps", linewidth=1.2)
    ec_hodl_s = (1.0 + hodl_s).cumprod()
    plt.plot(ec_hodl_s.index, ec_hodl_s.values, label="HODL spot", color="black", linestyle="--", linewidth=1.0)
    plt.title("Funding-arb delta-neutral OOS (BTC/USDT H2, fees taker 4bps round-trip)")
    plt.xlabel("date")
    plt.ylabel("Equity (1 USDT init)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves.png", dpi=140)
    plt.close()

    # 7. Returns components CSV (breakdown for the best threshold)
    comps = full_components[best_thr]
    comp_df = pd.DataFrame({
        "funding_pnl": comps["funding"],
        "fees_pnl": comps["fees"],
        "borrow_pnl": comps["borrow"],
        "position": comps["position"],
        "net_return": full_rets[best_thr],
    })
    comp_df.to_csv(outdir / "returns_components_best.csv")

    return {
        "best_threshold_bps": best_thr,
        "best_chain_sharpe": float(chained_metrics[best_thr]["sharpe"]),
        "best_chain_dsr": float(dsr_best),
        "chained_metrics": {f"thr_{thr}": chained_metrics[thr] for thr in thresholds_bps},
        "baseline_metrics": baseline_metrics,
        "agg": agg_df.to_dict(orient="records"),
        "spa": (None if spa is None else {
            "test_statistic": float(spa.test_statistic),
            "p_value_consistent": float(spa.p_value_consistent),
            "p_value_lower": float(spa.p_value_lower),
            "p_value_upper": float(spa.p_value_upper),
            "best_strategy_idx": int(spa.best_strategy),
            "best_threshold_bps": float(thresholds_bps[spa.best_strategy]),
            "n_strategies": int(spa.n_strategies),
            "n_bootstrap": int(spa.n_bootstrap),
        }),
        "n_obs_oos_chained": int(len(common_idx)),
        "n_bars_total": int(len(df)),
        "data_start": str(df.index[0]),
        "data_end": str(df.index[-1]),
        "funding_mean_bps_per_8h": float(funding.mean() * 1e4),
        "funding_median_bps_per_8h": float(funding.median() * 1e4),
        "funding_p95_bps_per_8h": float(funding.quantile(0.95) * 1e4),
        "config": {
            "fee_taker_bps": fee_taker_bps,
            "spot_borrow_apr": spot_borrow_apr,
            "holding_min_periods": holding_min_periods,
            "allow_short_spot": allow_short_spot,
            "n_folds": n_folds,
            "train_frac": train_frac,
            "thresholds_bps": list(thresholds_bps),
        },
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def make_recommendation(result: dict) -> dict:
    sr = result["best_chain_sharpe"]
    dsr = result["best_chain_dsr"]
    spa = result.get("spa") or {}
    p_spa = spa.get("p_value_consistent", float("nan"))
    hodl_sr = result["baseline_metrics"]["hodl_spot"]["sharpe"]

    if not np.isfinite(sr) or sr <= 0:
        decision = "NE_PAS_DEPLOYER"
        confidence = "HAUTE"
        reason = f"Sharpe chained OOS {sr:.2f} <= 0, edge inexistant après frais."
    elif np.isfinite(dsr) and dsr < 0.95:
        decision = "NE_PAS_DEPLOYER"
        confidence = "HAUTE"
        reason = f"DSR {dsr:.3f} < 0.95: edge non significatif après correction data snooping (5 thresholds)."
    elif np.isfinite(p_spa) and p_spa > 0.05:
        decision = "NE_PAS_DEPLOYER"
        confidence = "HAUTE"
        reason = f"Hansen SPA p={p_spa:.3f} > 0.05: ne rejette pas H0 (edge dû au hasard plausible)."
    elif np.isfinite(hodl_sr) and sr <= hodl_sr:
        decision = "NE_PAS_DEPLOYER"
        confidence = "MOYENNE"
        reason = f"Sharpe stratégie {sr:.2f} <= HODL spot {hodl_sr:.2f}: pas de surperformance vs buy-hold."
    else:
        decision = "TESTNET_LIMITE"
        confidence = "MOYENNE"
        reason = f"Sharpe {sr:.2f}, DSR {dsr:.3f}, SPA p={p_spa:.3f}: signal positif mais à confirmer en testnet (≥ 1 mois)."

    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "best_threshold_bps": result["best_threshold_bps"],
        "best_chain_sharpe": sr,
        "best_chain_dsr": dsr,
        "spa_p_consistent": p_spa,
        "hodl_spot_sharpe": hodl_sr,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(result: dict, recommendation: dict, outdir: Path) -> None:
    cfg = result["config"]
    lines = [
        "# RAPPORT — Funding rate arb delta-neutral (STRAT-A) — 2026-04-26",
        "",
        "> Pivot post-ALPHA-2 (Ichimoku K3 nu Sharpe -1.91 réfutée). Cf RAPPORT_ALPHA_FINAL_2026-04-26.md.",
        "",
        "## TL;DR",
        "",
        f"- **Décision: {recommendation['decision']}** (confiance {recommendation['confidence']})",
        f"- {recommendation['reason']}",
        f"- Best threshold: **{recommendation['best_threshold_bps']} bps/8h**",
        f"- Sharpe chained OOS: **{recommendation['best_chain_sharpe']:.3f}**",
        f"- DSR (n_trials={len(cfg['thresholds_bps'])}): **{recommendation['best_chain_dsr']:.3f}**",
        f"- Hansen SPA p_consistent: **{recommendation['spa_p_consistent']:.3f}**",
        f"- HODL spot Sharpe (baseline): {recommendation['hodl_spot_sharpe']:.3f}",
        "",
        "## Configuration",
        "",
        f"- Period: {result['data_start']} → {result['data_end']} ({result['n_bars_total']} bars H2)",
        f"- OOS chained obs: {result['n_obs_oos_chained']}",
        f"- Folds: {cfg['n_folds']} (train_frac={cfg['train_frac']})",
        f"- Thresholds testés (bps/8h): {cfg['thresholds_bps']}",
        f"- Fee taker: {cfg['fee_taker_bps']} bps × leg (4 legs round-trip = {4 * cfg['fee_taker_bps']} bps)",
        f"- Spot borrow APR: {cfg['spot_borrow_apr']:.1%}",
        f"- allow_short_spot: {cfg['allow_short_spot']}",
        f"- holding_min_periods: {cfg['holding_min_periods']}",
        "",
        "## Funding rate stats sur la période",
        "",
        f"- Mean: {result['funding_mean_bps_per_8h']:.3f} bps / 8h "
        f"(annualisé ≈ {result['funding_mean_bps_per_8h'] * FA.FUNDING_PERIODS_PER_YEAR / 1e4 * 100:.1f} %/an)",
        f"- Median: {result['funding_median_bps_per_8h']:.3f} bps / 8h",
        f"- p95: {result['funding_p95_bps_per_8h']:.3f} bps / 8h",
        "",
        "## Métriques chained OOS par threshold",
        "",
        "| thr (bps) | Sharpe | Sortino | MDD | Ulcer | CAGR | Total ret |",
        "|-----------|--------|---------|------|-------|------|-----------|",
    ]
    for thr in cfg["thresholds_bps"]:
        m = result["chained_metrics"][f"thr_{thr}"]
        lines.append(
            f"| {thr:>8.1f} | {m.get('sharpe', float('nan')):.3f} | "
            f"{m.get('sortino', float('nan')):.3f} | {m.get('mdd', float('nan')):.3%} | "
            f"{m.get('ulcer', float('nan')):.4f} | {m.get('cagr', float('nan')):.3%} | "
            f"{(1 + 0):.0f} |"
        )

    lines += [
        "",
        "## Baselines (mêmes barres OOS)",
        "",
        "| stratégie | Sharpe | MDD | CAGR |",
        "|-----------|--------|------|------|",
        f"| HODL spot | {result['baseline_metrics']['hodl_spot']['sharpe']:.3f} | "
        f"{result['baseline_metrics']['hodl_spot']['mdd']:.3%} | "
        f"{result['baseline_metrics']['hodl_spot']['cagr']:.3%} |",
        f"| HODL perp long | {result['baseline_metrics']['hodl_perp_long']['sharpe']:.3f} | "
        f"{result['baseline_metrics']['hodl_perp_long']['mdd']:.3%} | "
        f"{result['baseline_metrics']['hodl_perp_long']['cagr']:.3%} |",
        f"| ALPHA-2 V1 (réfutée)  | -1.91 | n/a | n/a |",
        "",
        "## Hansen SPA (K thresholds vs zero-return benchmark)",
        "",
    ]
    if result.get("spa"):
        spa = result["spa"]
        lines += [
            f"- Test statistic V = {spa['test_statistic']:.3f}",
            f"- p_value_consistent (recommandé): **{spa['p_value_consistent']:.3f}**",
            f"- p_value_lower (conservateur): {spa['p_value_lower']:.3f}",
            f"- p_value_upper (libéral): {spa['p_value_upper']:.3f}",
            f"- Best k = threshold {spa['best_threshold_bps']} bps",
            f"- N bootstrap = {spa['n_bootstrap']}",
        ]
    else:
        lines.append("- SPA non calculé (trop peu de barres OOS).")

    lines += [
        "",
        "## Limitations / risques",
        "",
        "1. **Basis risk ignoré**: spot et perp ne tracent pas parfaitement (basis fluctue ±50bps en stress).",
        "2. **Funding squeeze**: si la stratégie est crowded, le funding revient à 0 vite (mean reversion 95%).",
        "3. **Slippage modélisé seulement via fee taker × 4**: pas d'impact prix sur entry/exit.",
        "4. **Spot+perp hedge requires 2× capital ou cross-margin** (pas modélisé en sizing).",
        "5. **Funding empirique BTC ~1bp/8h** ≈ 11%/an mais après 16bps round-trip frais, ",
        "   il faut hold ≥ 8 events (~3 jours) pour amortir si on entre/sort à chaque flip.",
        "6. **Pas de walk-forward sur le threshold** (threshold fixé pour tous les folds): si on optimise",
        "   le threshold dans chaque fold train, on data-snoop encore plus.",
        "7. **2020-2025 inclut le bull 2021 + le crash 2022 + le bull 2024**: funding peak +100bps/8h en jan",
        "   2021 puis -50bps/8h en mai 2021. La distribution est fortement non-stationnaire.",
        "",
        "## Verdict",
        "",
        f"**{recommendation['decision']}** — {recommendation['reason']}",
        "",
        "Voir `recommendation.json` pour la décision machine-readable et `equity_curves.png`.",
    ]
    (outdir / "RAPPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Funding rate arb backtest")
    p.add_argument("--data", type=Path, default=Path("data/BTC_USDT_2h.csv"))
    p.add_argument("--funding", type=Path, default=Path("data/funding_rate_BTCUSDT.csv"))
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-08-01")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--fee-taker-bps", type=float, default=4.0)
    p.add_argument("--spot-borrow-apr", type=float, default=0.05)
    p.add_argument("--holding-min-periods", type=int, default=1)
    p.add_argument("--allow-short-spot", action="store_true")
    p.add_argument("--thresholds", type=str, default="3,5,7,10,15",
                   help="Comma-separated thresholds in bps/8h.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = p.parse_args(argv)

    thresholds = tuple(float(x) for x in args.thresholds.split(","))

    print(f"Loading {args.data} [{args.start} → {args.end}]")
    df = load_data(args.data, args.start, args.end)
    print(f"Loaded {len(df)} bars BTC H2.")

    print(f"Loading funding {args.funding}")
    funding = load_funding(args.funding, df)
    print(f"Funding aligned: {funding.notna().sum()} bars with data.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    result = run_backtest(
        df=df,
        funding=funding,
        thresholds_bps=thresholds,
        n_folds=args.folds,
        train_frac=args.train_frac,
        fee_taker_bps=args.fee_taker_bps,
        spot_borrow_apr=args.spot_borrow_apr,
        holding_min_periods=args.holding_min_periods,
        allow_short_spot=args.allow_short_spot,
        outdir=args.outdir,
    )
    recommendation = make_recommendation(result)
    (args.outdir / "recommendation.json").write_text(json.dumps(recommendation, indent=2))
    (args.outdir / "result_full.json").write_text(json.dumps(result, indent=2, default=str))
    write_report(result, recommendation, args.outdir)

    print()
    print(f"=== RESULT ===")
    print(f"Best threshold: {recommendation['best_threshold_bps']} bps")
    print(f"Sharpe chained OOS: {recommendation['best_chain_sharpe']:.3f}")
    print(f"DSR: {recommendation['best_chain_dsr']:.3f}")
    print(f"SPA p_consistent: {recommendation['spa_p_consistent']:.3f}")
    print(f"HODL spot Sharpe: {recommendation['hodl_spot_sharpe']:.3f}")
    print(f"=> {recommendation['decision']} ({recommendation['confidence']})")
    print(f"Reason: {recommendation['reason']}")
    print(f"Outputs in {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
