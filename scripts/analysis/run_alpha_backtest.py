#!/usr/bin/env python3
"""ALPHA-2 — Comparative walk-forward backtest of 4 Ichimoku variants on BTC H2.

Outputs (under ``outputs/wfa_alpha_2026-04-26/``):
    - metrics_summary.csv   : table strategy x fold x metrics
    - equity_curves.png     : 4 lignes (full OOS chained)
    - RAPPORT_ALPHA_BACKTEST.md
    - recommendation.json

Usage:
    .venv/bin/python -m scripts.analysis.run_alpha_backtest \
        --data data/BTC_USDT_2h.csv \
        --start 2020-01-01 --end 2025-08-01 \
        --folds 5

Tous les chiffres sont nets (frais + funding + slippage via ``src.cost_model``).
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

# Allow ``python scripts/analysis/run_alpha_backtest.py`` from repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src import alpha_strategies as A
from src import stats_eval as SE
from src import reality_check as RC


PERIODS_PER_YEAR_H2 = 365 * 12  # 4380 bars/an
DEFAULT_OUTDIR = Path("outputs/wfa_alpha_2026-04-26")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.loc[start:end].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_funding(path: Optional[Path], idx: pd.DatetimeIndex) -> Optional[pd.Series]:
    if path is None or not Path(path).exists():
        return None
    fr = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")["funding_rate"]
    fr = fr.sort_index()
    # Align timezone-naive
    if fr.index.tz is not None:
        fr.index = fr.index.tz_localize(None)
    return fr


# ---------------------------------------------------------------------------
# Walk-forward folds: 70/30 train/test (we only need OOS metrics here, so
# we keep the train slice as warm-up for indicators and report metrics on
# the test slice).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(df: pd.DataFrame, n_folds: int = 5, train_frac: float = 0.7) -> list[Fold]:
    """Anchored expanding-window folds with rolling test windows."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n = len(df)
    # We split [0, n] in n_folds+1 chunks and roll the test window forward.
    # First fold uses chunks 0..k-1 for train (k = ceil(n*train_frac/n_folds * 1)),
    # but for simplicity we use equal-sized test windows.
    test_size = int(n * (1.0 - train_frac) / n_folds)
    if test_size < 100:
        raise ValueError(f"test window too small ({test_size} bars) — increase data span")
    # Train start = global start; test start = global_start + train_size + i*test_size
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
# Per-fold simulation
# ---------------------------------------------------------------------------

def simulate_variant_full(
    df: pd.DataFrame,
    variant: A.Variant,
    funding_rates: Optional[pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """Retourne ``(returns_net, position_held)`` sur tout ``df``.

    Le calcul est fait en une fois sur la série complète pour que les
    indicators warmup proprement ; on filtrera ensuite par fold.
    """
    sig = A.generate_signals(df, variant)
    rets = A.simulate_returns(df, sig, funding_rates=funding_rates)
    # Position effectivement détenue = sig.shift(1)
    held = sig.shift(1).fillna(0.0)
    return rets, held


def fold_metrics(
    rets: pd.Series,
    held: pd.Series,
    *,
    n_trials: int,
    periods_per_year: int = PERIODS_PER_YEAR_H2,
) -> dict[str, float]:
    base = SE.compute_metrics(rets, periods_per_year)
    n_obs = int(rets.dropna().shape[0])
    sr = base.get("sharpe", float("nan"))
    skew = float(rets.skew()) if n_obs > 3 else 0.0
    kurt = float(rets.kurtosis() + 3.0) if n_obs > 3 else 3.0  # kurtosis=excess by default
    if np.isfinite(sr):
        # SR per-period = SR_annualized / sqrt(periods)
        sr_per = sr / np.sqrt(periods_per_year)
        dsr = SE.deflated_sharpe_ratio(
            sharpe_observed=sr_per,
            n_obs=n_obs,
            n_trials=n_trials,
            var_sr_trials=1.0 / periods_per_year,
            skew=skew,
            kurt=kurt,
        )
    else:
        dsr = float("nan")
    # Trade stats
    n_trades = A.count_trades(held)
    in_pos = held != 0
    nz = rets[in_pos]
    if len(nz):
        win_rate = float((nz > 0).mean())
        avg_trade = float(nz.mean())
    else:
        win_rate = float("nan")
        avg_trade = float("nan")
    out = dict(base)
    out["dsr"] = float(dsr) if dsr is not None else float("nan")
    out["n_obs"] = n_obs
    out["n_trades"] = n_trades
    out["win_rate"] = win_rate
    out["avg_trade"] = avg_trade
    out["total_return"] = float((1.0 + rets.fillna(0.0)).prod() - 1.0)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    funding_rates: Optional[pd.Series],
    n_folds: int,
    train_frac: float,
    outdir: Path,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # Pre-compute returns + position for each variant on the full df.
    full_rets: dict[str, pd.Series] = {}
    full_held: dict[str, pd.Series] = {}
    for v in A.VARIANTS:
        r, h = simulate_variant_full(df, v, funding_rates)
        full_rets[v] = r
        full_held[v] = h

    folds = make_folds(df, n_folds=n_folds, train_frac=train_frac)

    rows: list[dict] = []
    fold_oos_rets: dict[str, list[pd.Series]] = {v: [] for v in A.VARIANTS}
    for fold in folds:
        for v in A.VARIANTS:
            r = full_rets[v].loc[fold.test_start:fold.test_end]
            h = full_held[v].loc[fold.test_start:fold.test_end]
            m = fold_metrics(r, h, n_trials=len(A.VARIANTS))
            row = {
                "strategy": v,
                "fold": fold.fold_id,
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                **m,
            }
            rows.append(row)
            fold_oos_rets[v].append(r)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "metrics_summary.csv", index=False)

    # Median-OOS aggregate per strategy.
    agg_rows: list[dict] = []
    metric_cols = [
        "sharpe", "dsr", "sortino", "calmar", "cagr", "mdd", "ulcer",
        "dd_duration", "n_trades", "win_rate", "avg_trade", "total_return",
    ]
    for v in A.VARIANTS:
        sub = metrics_df[metrics_df["strategy"] == v]
        agg = {"strategy": v, "n_folds": int(len(sub))}
        for c in metric_cols:
            agg[c + "_median"] = float(sub[c].median())
            agg[c + "_mean"] = float(sub[c].mean())
        agg_rows.append(agg)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(outdir / "metrics_aggregate.csv", index=False)

    # Hansen SPA: matrix (n_bars_oos, K) of excess returns vs V1.
    chained: dict[str, pd.Series] = {}
    for v in A.VARIANTS:
        # Concaténation des fold OOS (purgés des warm-ups).
        chained[v] = pd.concat(fold_oos_rets[v]).sort_index()
    common_idx = chained["V1"].index
    for v in A.VARIANTS[1:]:
        common_idx = common_idx.intersection(chained[v].index)
    common_idx = common_idx.sort_values()
    if len(common_idx) < 50:
        spa = None
        spa_strategy_names: list[str] = []
    else:
        # SPA = test "au moins une candidate bat V1". On exclut V1 lui-même
        # (sinon sa colonne d'excès est triviale ⇒ V=0, p≈1 mécaniquement).
        spa_strategy_names = [v for v in A.VARIANTS if v != "V1"]
        excess = np.column_stack([
            (chained[v].reindex(common_idx).fillna(0.0) -
             chained["V1"].reindex(common_idx).fillna(0.0)).to_numpy()
            for v in spa_strategy_names
        ])
        spa = RC.hansen_spa_test(excess, n_bootstrap=1000, block_size_mean=20.0, seed=42)

    # Equity curves plot
    plt.figure(figsize=(11, 6))
    for v in A.VARIANTS:
        ec = (1.0 + chained[v].fillna(0.0)).cumprod()
        plt.plot(ec.index, ec.values, label=v, linewidth=1.2)
    plt.title("Equity curves chained OOS (BTC/USDT H2, costs net)")
    plt.xlabel("date")
    plt.ylabel("Equity (1 USDT initial)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves.png", dpi=140)
    plt.close()

    # Compose return dict
    result = {
        "agg": agg_df.to_dict(orient="records"),
        "metrics": metrics_df.to_dict(orient="records"),
        "spa": (None if spa is None else {
            "test_statistic": spa.test_statistic,
            "p_value_consistent": spa.p_value_consistent,
            "p_value_lower": spa.p_value_lower,
            "p_value_upper": spa.p_value_upper,
            "best_strategy_idx": spa.best_strategy,
            "best_strategy_name": spa_strategy_names[spa.best_strategy],
            "compared_against_v1": spa_strategy_names,
            "n_strategies": spa.n_strategies,
            "n_bootstrap": spa.n_bootstrap,
        }),
        "n_obs_oos_chained": int(len(common_idx)),
    }
    return result


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def make_recommendation(result: dict) -> dict:
    """Décide V2/V3/V4/none selon DSR + p-value SPA + Sharpe uplift."""
    agg = {row["strategy"]: row for row in result["agg"]}
    spa = result.get("spa")

    v1_sharpe = agg["V1"]["sharpe_median"]
    v1_dsr = agg["V1"]["dsr_median"]

    # Candidate = best Sharpe parmi V2/V3/V4
    candidates = ["V2", "V3", "V4"]
    cand_scores = {c: agg[c]["sharpe_median"] for c in candidates}
    # On filtre NaN
    cand_scores = {c: s for c, s in cand_scores.items() if np.isfinite(s)}
    if not cand_scores:
        return {"deploy": "none",
                "reason": "no candidate has a finite Sharpe",
                "config": {},
                "expected_sharpe_uplift": 0.0}

    best = max(cand_scores, key=cand_scores.get)
    best_sharpe = cand_scores[best]
    best_dsr = agg[best]["dsr_median"]
    uplift = best_sharpe - (v1_sharpe if np.isfinite(v1_sharpe) else 0.0)

    spa_p = spa["p_value_consistent"] if spa else 1.0
    best_name_spa = spa["best_strategy_name"] if spa else None

    # Decision: deploy iff uplift > 0 AND best DSR > V1 DSR AND SPA p-consistent < 0.05
    # AND the SPA winner matches our candidate.
    deploy = "none"
    reason = []
    if uplift <= 0:
        reason.append(f"Sharpe uplift {uplift:+.3f} <= 0")
    if not (np.isfinite(best_dsr) and np.isfinite(v1_dsr) and best_dsr > v1_dsr):
        reason.append(f"DSR({best})={best_dsr:.3f} ≤ DSR(V1)={v1_dsr:.3f}")
    if spa_p > 0.05:
        reason.append(f"Hansen SPA p-consistent={spa_p:.3f} > 0.05")
    if best_name_spa and best_name_spa != best and best_name_spa != "V1":
        reason.append(f"SPA winner={best_name_spa} ≠ best Sharpe candidate {best}")

    if not reason:
        deploy = best
        reason_text = "all gates passed"
    else:
        reason_text = "; ".join(reason)

    config = {
        "strategy": deploy if deploy != "none" else None,
        "ichimoku_k3": {"tenkan": 21, "kijun": 35, "senkou_b": 90, "shift": 44},
        "filters_used": {
            "V1": [],
            "V2": ["er>0.30", "adx>22"],
            "V3": ["er>0.30", "adx>22", "chop<=61.8"],
            "V4": ["range_score<0.4"],
        }.get(deploy, []),
    }
    return {
        "deploy": deploy,
        "reason": reason_text,
        "config": config,
        "expected_sharpe_uplift": float(uplift),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x: float, fmt: str = "{:+.3f}") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return fmt.format(x)


def write_markdown(
    result: dict,
    recommendation: dict,
    df: pd.DataFrame,
    outdir: Path,
    n_folds: int,
) -> None:
    spa = result.get("spa")
    agg = {row["strategy"]: row for row in result["agg"]}
    lines: list[str] = []
    lines.append("# RAPPORT ALPHA-2 — Backtest comparatif Ichimoku K3 + filtres régime")
    lines.append("")
    lines.append("Auteur : agent ALPHA-2 (2026-04-26).  ")
    lines.append("Asset : **BTC/USDT perpetuel**, granularité **H2**, "
                 f"période **{df.index.min().date()} → {df.index.max().date()}** "
                 f"({len(df)} bars).  ")
    lines.append("Coûts nets : Binance VIP 0 (taker 4 bps, maker 2 bps), "
                 "funding +0.01 %/8 h (proxy historique 2020-2025), "
                 "slippage spread 0.5 bp + composante ATR.")
    lines.append("")
    lines.append("## 1. Hypothèses testées")
    lines.append("")
    lines.append("- **V1 — Baseline Ichimoku K3** : Tenkan=21, Kijun=35, Senkou B=90, shift=+44. "
                 "Long si `tenkan>kijun` ET `close>max(senkA,senkB)` ; symétrique au short.")
    lines.append("- **V2 — V1 + filtre ER+ADX** : entrée seulement si `efficiency_ratio(close,30)>0.30` ET "
                 "`adx(14)>22` (Kaufman/Wilder).")
    lines.append("- **V3 — V2 + veto CHOP** : en plus, `choppiness_index(14)<=61.8` (Dreiss).")
    lines.append("- **V4 — V1 + range_score<0.4** : score composite `src.range_detector.range_score` "
                 "(ADX, CI, ER, BBW squeeze, Hurst rolling).")
    lines.append("")
    lines.append("## 2. Méthodologie")
    lines.append("")
    lines.append(f"- **Walk-forward** : {n_folds} folds, train/test 70 / 30, expanding train, OOS rolling.")
    lines.append("- Métriques calculées **par fold** puis médiane multi-fold.")
    lines.append("- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) avec `n_trials=4` "
                 "(les 4 stratégies comparées).")
    lines.append("- **Hansen SPA** (consistent, lower, upper) : 1000 répliques bootstrap stationnaire, "
                 "block size moyenne 20 bars. Test sur returns OOS chaînés des 5 folds.")
    lines.append("- Coûts via `src.cost_model.apply_costs_to_returns` (frais + funding + slippage).")
    lines.append("- `periods_per_year = 365 * 12 = 4380` pour annualiser le Sharpe sur barres H2.")
    lines.append("")
    lines.append("## 3. Tableau métriques (médiane OOS multi-fold)")
    lines.append("")
    lines.append("| Stratégie | Sharpe | DSR | Sortino | MDD | Ulcer | n_trades | Win rate | Total OOS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in A.VARIANTS:
        r = agg[v]
        lines.append(
            f"| {v} | {_fmt(r['sharpe_median'])} | {_fmt(r['dsr_median'])} "
            f"| {_fmt(r['sortino_median'])} | {_fmt(r['mdd_median'], '{:+.2%}')} "
            f"| {_fmt(r['ulcer_median'], '{:.4f}')} "
            f"| {_fmt(r['n_trades_median'], '{:.0f}')} "
            f"| {_fmt(r['win_rate_median'], '{:.2%}')} "
            f"| {_fmt(r['total_return_median'], '{:+.2%}')} |"
        )
    lines.append("")
    lines.append("Lecture : Sharpe et Sortino sont annualisés. DSR proche de 1.0 = "
                 "edge significatif après data snooping ; proche de 0.5 = edge incertain.")
    lines.append("")
    lines.append("## 4. Hansen SPA test")
    lines.append("")
    if spa is None:
        lines.append("*(non calculable : trop peu d'observations OOS communes.)*")
    else:
        lines.append(f"- Test statistic V = {spa['test_statistic']:.4f}  "
                     f"(observations OOS chaînées : {result['n_obs_oos_chained']})")
        lines.append(f"- **p-value SPA_consistent** = {spa['p_value_consistent']:.3f}  "
                     f"(seuil rejet H0 < 0.05 ⇒ edge réel)")
        lines.append(f"- p-value SPA_lower (conservateur) = {spa['p_value_lower']:.3f}  ")
        lines.append(f"- p-value SPA_upper (libéral) = {spa['p_value_upper']:.3f}  ")
        lines.append(f"- **Stratégie best (max f̄_k)** : {spa['best_strategy_name']} "
                     f"(index {spa['best_strategy_idx']})")
    lines.append("")
    lines.append("## 5. Recommandation finale")
    lines.append("")
    deploy = recommendation["deploy"]
    if deploy == "none":
        lines.append(f"### NE PAS DÉPLOYER")
        lines.append("")
        lines.append(f"Raison : {recommendation['reason']}.")
        lines.append("")
        lines.append("Aucune stratégie filtrée n'a passé simultanément les trois gates : "
                     "Sharpe > V1, DSR > V1, p-value Hansen SPA < 0.05. "
                     "Cf hypothèse H_0 du rapport recherche : Ichimoku seul + filtre régime "
                     "n'apporte pas d'edge net statistiquement détectable sur cet asset/timeframe.")
    else:
        lines.append(f"### Déploiement recommandé : **{deploy}**")
        lines.append("")
        lines.append(f"Raison : {recommendation['reason']}.")
        lines.append(f"Sharpe uplift attendu vs V1 : "
                     f"{recommendation['expected_sharpe_uplift']:+.3f}.")
        lines.append("")
        lines.append("**Conditions de déploiement** :")
        lines.append("- Phase testnet ≥ 6 semaines avant taille pleine (López de Prado min track record).")
        lines.append("- Surveillance funding rate live : si pic > 0.1 %/8 h, suspendre l'entrée.")
        lines.append("- Stop global : MDD live > 1.5 × MDD backtest ⇒ kill-switch.")
        lines.append("- Re-tuning trimestriel des seuils ER/ADX/CHOP en walk-forward.")
    lines.append("")
    lines.append("## 6. Limitations (honnêteté impérative)")
    lines.append("")
    lines.append("- **1 seul asset** (BTC/USDT). Aucune généralisation à ETH/altcoins.")
    lines.append("- **1 seul timeframe** (H2). H1/H4 peuvent renverser les conclusions.")
    lines.append("- **Frais & funding constants** : la dynamique de funding crowded est ignorée. "
                 "Live, les funding extrêmes peuvent éroder 1-3 bps additionnels par bar long.")
    lines.append("- **n_trials=4** est sous-estimé si on considère qu'on a aussi joué sur les seuils "
                 "(0.30, 22, 61.8, 0.4). Le DSR rapporté est donc optimiste sur l'effet snooping.")
    lines.append("- **Pas de stop-loss explicite** dans la simulation : la position est toujours "
                 "long/short selon le signal, sans gestion fine de drawdown intra-trade. "
                 "Live, l'interaction stop-loss + filtre régime peut différer.")
    lines.append("- **Slippage proxy** : 0.5 bp + composante ATR ; un live BTC stressé "
                 "(news, liquidations) peut taper 5-10 bps.")
    lines.append("- **5 folds** seulement : intervalle de confiance large sur la médiane.")
    lines.append("")
    lines.append("## 7. Reproductibilité")
    lines.append("")
    lines.append("Commande exacte :")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python -m scripts.analysis.run_alpha_backtest \\")
    lines.append("    --data data/BTC_USDT_2h.csv \\")
    lines.append("    --funding data/funding_rate_BTCUSDT.csv \\")
    lines.append("    --start 2020-01-01 --end 2025-08-01 --folds 5")
    lines.append("```")
    lines.append("")
    lines.append("Tests : `pytest tests/test_alpha_backtest.py -v`")
    lines.append("")
    md_path = outdir / "RAPPORT_ALPHA_BACKTEST.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/BTC_USDT_2h.csv")
    p.add_argument("--funding", default="data/funding_rate_BTCUSDT.csv")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-08-01")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data), args.start, args.end)
    if df.empty:
        raise SystemExit("Empty data window — check --start/--end")
    funding = load_funding(Path(args.funding) if args.funding else None, df.index)

    print(f"[run_alpha_backtest] Loaded {len(df)} bars "
          f"{df.index.min()} → {df.index.max()}, "
          f"funding={'yes' if funding is not None else 'NO (proxy 0.01%/8h)'}")

    result = run_backtest(df, funding, n_folds=args.folds,
                          train_frac=args.train_frac, outdir=outdir)
    recommendation = make_recommendation(result)

    write_markdown(result, recommendation, df, outdir, n_folds=args.folds)
    (outdir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[run_alpha_backtest] Aggregate (median OOS multi-fold):")
    for row in result["agg"]:
        print(f"  {row['strategy']}  Sharpe={row['sharpe_median']:+.3f}  "
              f"DSR={row['dsr_median']:.3f}  MDD={row['mdd_median']:+.2%}  "
              f"trades={row['n_trades_median']:.0f}")
    print(f"[run_alpha_backtest] Recommendation: deploy={recommendation['deploy']} "
          f"(uplift={recommendation['expected_sharpe_uplift']:+.3f})")
    print(f"[run_alpha_backtest] Outputs in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
