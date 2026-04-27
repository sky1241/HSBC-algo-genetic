#!/usr/bin/env python3
"""ALPHA-3 — Walk-forward backtest of Regime-Aware LightGBM on BTC H2.

Pipeline:
    1. Load BTC_USDT_2h.csv + funding_rate_BTCUSDT.csv (period 2020-01 -> 2025-08).
    2. Build features (ALPHA-1 + funding + temporal + momentum) — no lookahead.
    3. Walk-forward N folds (expanding train, 1-2 month OOS test, purge=horizon).
    4. Per fold: fit HMM 2 states, fit LightGBM, predict signal.
    5. Apply realistic costs (B5) -> net returns.
    6. Aggregate metrics: Sharpe, Sortino, Calmar, MDD, Ulcer, DSR, AUC, Brier.
    7. Hansen SPA vs HODL & ALPHA-2 V1.
    8. Save outputs/strat_regime_lgbm_2026-04-26/.

Honesty rules:
    - Sharpe OOS < 0.5  -> ne pas claim alpha.
    - DSR < 0.7         -> NOT DEPLOY.
    - SPA p > 0.05      -> ne pas deployer.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score, brier_score_loss  # noqa: E402

from src import alpha_strategies as A  # noqa: E402
from src import cost_model  # noqa: E402
from src import regime_lgbm as rlg  # noqa: E402
from src import reality_check as rc  # noqa: E402
from src import stats_eval as se  # noqa: E402

warnings.filterwarnings("ignore")


DATA_BTC = REPO / "data" / "BTC_USDT_2h.csv"
DATA_FUNDING = REPO / "data" / "funding_rate_BTCUSDT.csv"
OUT_DIR = REPO / "outputs" / "strat_regime_lgbm_2026-04-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periode backtest
DATE_START = "2020-01-01"
DATE_END = "2025-08-01"

# Walk-forward parametres
HORIZON = 3                  # label forward H2 = 6h
PURGE = HORIZON              # purge entre train/test = horizon de label
TEST_MONTHS = 2              # test OOS = 2 mois (donne ~30+ folds 2020->2025)
MIN_TRAIN_MONTHS = 12        # bootstrap warmup minimum
PERIODS_PER_YEAR = 12 * 365 // 2  # H2: ~4380


# ============================================================
# Data loading
# ============================================================

def load_btc() -> pd.DataFrame:
    df = pd.read_csv(DATA_BTC, parse_dates=["timestamp"]).set_index("timestamp")
    df = df.sort_index()
    df = df.loc[DATE_START:DATE_END].copy()
    return df


def load_funding() -> pd.Series:
    f = pd.read_csv(DATA_FUNDING, parse_dates=["timestamp"]).set_index("timestamp")
    f = f.sort_index()
    return f["funding_rate"].astype(float)


# ============================================================
# Walk-forward
# ============================================================

def make_folds(
    index: pd.DatetimeIndex,
    test_months: int,
    min_train_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_end, test_start, test_end) tuples.

    Train est expanding (debut = index[0], fin = test_start - purge).
    """
    starts = pd.date_range(
        start=index[0] + pd.DateOffset(months=min_train_months),
        end=index[-1],
        freq=pd.DateOffset(months=test_months),
    )
    folds = []
    for ts in starts:
        test_start = ts
        test_end = ts + pd.DateOffset(months=test_months)
        if test_end > index[-1]:
            test_end = index[-1]
        if test_start >= test_end:
            continue
        folds.append((test_start, test_end))
    return folds


def hodl_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


# ============================================================
# Main pipeline
# ============================================================

def run() -> None:
    print(f"[ALPHA-3] Loading BTC H2 from {DATE_START} -> {DATE_END}")
    df = load_btc()
    print(f"  bars: {len(df)}")

    print("[ALPHA-3] Loading funding rates")
    funding = load_funding()

    print("[ALPHA-3] Building features (causal)")
    feats = rlg.build_features(df, funding_h8=funding)
    label = rlg.label_directional(df["close"], horizon=HORIZON, flat_threshold=0.002)

    log_returns = np.log(df["close"]).diff()
    realized_vol = log_returns.rolling(20).std()

    # Drop NaN rows for downstream alignment
    mask = feats.notna().all(axis=1) & label.notna()
    valid_idx = feats.index[mask]
    print(f"[ALPHA-3] Valid rows after NaN drop: {len(valid_idx)} ({len(valid_idx)/len(df):.1%})")

    feats_v = feats.loc[valid_idx]
    label_v = label.loc[valid_idx]

    # ALPHA-2 V1 baseline (full series, then aligne)
    print("[ALPHA-3] Computing ALPHA-2 V1 baseline")
    sig_v1 = A.generate_signals(df, "V1")
    ret_v1 = A.simulate_returns(df, sig_v1)
    ret_hodl = hodl_returns(df["close"])

    # Walk-forward folds
    folds = make_folds(valid_idx, TEST_MONTHS, MIN_TRAIN_MONTHS)
    print(f"[ALPHA-3] {len(folds)} folds")

    all_signals = pd.Series(0, index=df.index, dtype=float)
    all_proba_up = pd.Series(np.nan, index=df.index, dtype=float)
    feature_imp_acc: dict[str, float] = {}

    for i, (t_start, t_end) in enumerate(folds):
        # Train: tout avant (t_start - purge bars)
        train_end_idx = valid_idx[valid_idx < t_start]
        if len(train_end_idx) <= PURGE:
            continue
        # Purge le bord pour eviter overlap label / test
        train_end_ts = train_end_idx[-PURGE - 1] if PURGE < len(train_end_idx) else train_end_idx[0]
        train_idx = valid_idx[valid_idx <= train_end_ts]
        test_idx = valid_idx[(valid_idx >= t_start) & (valid_idx < t_end)]
        if len(train_idx) < 200 or len(test_idx) < 5:
            continue

        # Fit HMM sur train uniquement
        try:
            hmm_model, _ = rlg.fit_hmm_regimes(
                log_returns.loc[train_idx], realized_vol.loc[train_idx],
                n_states=2, random_state=42,
            )
        except Exception as exc:
            print(f"  fold {i}: HMM fit failed: {exc}")
            continue

        # Apply HMM sur train ET test (les params sont fixes apres fit)
        regime_train = rlg.predict_hmm_regimes(
            hmm_model, log_returns.loc[train_idx], realized_vol.loc[train_idx]
        )
        regime_test = rlg.predict_hmm_regimes(
            hmm_model, log_returns.loc[test_idx], realized_vol.loc[test_idx]
        )

        # Fit LGBM
        cfg = rlg.RegimeLGBMConfig(num_boost_round=150, early_stopping_rounds=20)
        clf = rlg.RegimeLightGBM(cfg)
        # Use last 10% of train as val for early stopping
        n_train = len(train_idx)
        val_cut = int(0.9 * n_train)
        train_part = train_idx[:val_cut]
        val_part = train_idx[val_cut:]
        try:
            clf.fit(
                feats_v.loc[train_part], label_v.loc[train_part], regime_train.loc[train_part],
                X_val=feats_v.loc[val_part], y_val=label_v.loc[val_part],
                regime_val=regime_train.loc[val_part],
            )
        except Exception as exc:
            print(f"  fold {i}: LGBM fit failed: {exc}")
            continue

        # Predict signals
        try:
            sig = clf.predict(feats_v.loc[test_idx], regime_test.loc[test_idx])
            proba = clf.predict_proba(feats_v.loc[test_idx], regime_test.loc[test_idx])
        except Exception as exc:
            print(f"  fold {i}: predict failed: {exc}")
            continue

        all_signals.loc[test_idx] = sig.astype(float)
        # P(up) = class 2 in our mapping
        all_proba_up.loc[test_idx] = proba[:, 2]

        # Aggregate feature importance
        imp = clf.feature_importance("gain")
        for k, v in imp.items():
            feature_imp_acc[k] = feature_imp_acc.get(k, 0.0) + v

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  fold {i+1}/{len(folds)}: train={len(train_idx)}, test={len(test_idx)}, "
                  f"sig +1/0/-1 = {(sig==1).sum()}/{(sig==0).sum()}/{(sig==-1).sum()}")

    # ============================================================
    # Net returns avec costs
    # ============================================================
    print("[ALPHA-3] Applying costs to LGBM signal")
    # Restrict to OOS span (premier fold start -> end)
    oos_start = folds[0][0] if folds else df.index[0]
    oos_mask = df.index >= oos_start

    sig_lgbm = all_signals.copy()
    ret_lgbm = A.simulate_returns(df, sig_lgbm)

    # OOS slices
    ret_lgbm_oos = ret_lgbm.loc[oos_mask]
    ret_v1_oos = ret_v1.loc[oos_mask]
    ret_hodl_oos = ret_hodl.loc[oos_mask]

    # ============================================================
    # Metrics
    # ============================================================
    print("[ALPHA-3] Computing metrics")
    metrics_lgbm = se.compute_metrics(ret_lgbm_oos, PERIODS_PER_YEAR)
    metrics_v1 = se.compute_metrics(ret_v1_oos, PERIODS_PER_YEAR)
    metrics_hodl = se.compute_metrics(ret_hodl_oos, PERIODS_PER_YEAR)

    # DSR (10 trials = on a fait varier hyperparams implicitement)
    dsr_lgbm = se.deflated_sharpe_ratio(
        sharpe_observed=metrics_lgbm["sharpe"],
        n_obs=int(ret_lgbm_oos.notna().sum()),
        n_trials=10,
        var_sr_trials=1.0,
        skew=float(ret_lgbm_oos.skew()),
        kurt=float(ret_lgbm_oos.kurt() + 3.0),
    )

    # Per-fold Sharpe distribution -> median OOS Sharpe
    fold_sharpes = []
    for t_start, t_end in folds:
        sl = ret_lgbm.loc[(ret_lgbm.index >= t_start) & (ret_lgbm.index < t_end)]
        if len(sl) < 10 or sl.std() == 0:
            continue
        sh = float(np.sqrt(PERIODS_PER_YEAR) * sl.mean() / sl.std())
        fold_sharpes.append(sh)
    median_fold_sharpe = float(np.median(fold_sharpes)) if fold_sharpes else float("nan")

    # AUC ROC + Brier (on direction = +1 vs -1)
    valid_proba = all_proba_up.dropna()
    if len(valid_proba) > 100:
        # Vrai label binarise: +1 si forward up, sinon 0 (excluant flat)
        y_true_bin = label.loc[valid_proba.index]
        bin_mask = y_true_bin != 0
        if bin_mask.sum() > 50:
            y_for_auc = (y_true_bin[bin_mask] == 1).astype(int).values
            p_for_auc = valid_proba[bin_mask].values
            try:
                auc = float(roc_auc_score(y_for_auc, p_for_auc))
                brier = float(brier_score_loss(y_for_auc, p_for_auc))
            except Exception:
                auc, brier = float("nan"), float("nan")
        else:
            auc, brier = float("nan"), float("nan")
    else:
        auc, brier = float("nan"), float("nan")

    # Hansen SPA vs (HODL, V1) — on test si LGBM bat les deux benchmarks
    er_vs_hodl = (ret_lgbm_oos - ret_hodl_oos).dropna().values
    er_vs_v1 = (ret_lgbm_oos - ret_v1_oos).dropna().values
    n_align = min(len(er_vs_hodl), len(er_vs_v1))
    excess = np.column_stack([er_vs_hodl[:n_align], er_vs_v1[:n_align]])
    print(f"[ALPHA-3] Hansen SPA on n={n_align} bars vs 2 benchmarks")
    spa = rc.hansen_spa_test(excess, n_bootstrap=1000, block_size_mean=20.0, seed=42)

    # ============================================================
    # Verdict
    # ============================================================
    deploy = "yes"
    reasons: list[str] = []
    if metrics_lgbm["sharpe"] < 0.5:
        deploy = "no"
        reasons.append(f"Sharpe OOS {metrics_lgbm['sharpe']:.2f} < 0.5")
    if dsr_lgbm < 0.7:
        deploy = "no"
        reasons.append(f"DSR {dsr_lgbm:.2f} < 0.7")
    if spa.p_value_consistent > 0.05:
        deploy = "no"
        reasons.append(f"Hansen SPA p={spa.p_value_consistent:.3f} > 0.05")

    if deploy == "yes":
        reasons.append("Tous les criters OK — alpha plausible.")

    rec = {
        "deploy": deploy,
        "reason": "; ".join(reasons),
        "expected_sharpe": metrics_lgbm["sharpe"],
        "median_fold_sharpe": median_fold_sharpe,
        "metrics_lgbm": metrics_lgbm,
        "metrics_v1": metrics_v1,
        "metrics_hodl": metrics_hodl,
        "dsr_lgbm": dsr_lgbm,
        "auc_roc": auc,
        "brier_score": brier,
        "hansen_spa_p_consistent": spa.p_value_consistent,
        "hansen_spa_p_lower": spa.p_value_lower,
        "hansen_spa_p_upper": spa.p_value_upper,
        "n_folds": len(folds),
        "n_oos_bars": int(oos_mask.sum()),
        "warning_paper": "MDPI 2025 reports CI95 [0.53, 1.84] => edge fragile.",
    }

    with open(OUT_DIR / "recommendation.json", "w") as f:
        json.dump(rec, f, indent=2, default=float)
    print(f"[ALPHA-3] Saved {OUT_DIR/'recommendation.json'}")

    # ============================================================
    # Plots
    # ============================================================
    print("[ALPHA-3] Plotting equity curves")
    fig, ax = plt.subplots(figsize=(11, 5))
    eq_lgbm = (1 + ret_lgbm_oos.fillna(0)).cumprod()
    eq_v1 = (1 + ret_v1_oos.fillna(0)).cumprod()
    eq_hodl = (1 + ret_hodl_oos.fillna(0)).cumprod()
    ax.plot(eq_lgbm.index, eq_lgbm.values, label=f"Regime-LGBM (Sharpe={metrics_lgbm['sharpe']:.2f})", lw=1.5)
    ax.plot(eq_v1.index, eq_v1.values, label=f"ALPHA-2 V1 (Sharpe={metrics_v1['sharpe']:.2f})", lw=1.0, alpha=0.7)
    ax.plot(eq_hodl.index, eq_hodl.values, label=f"HODL (Sharpe={metrics_hodl['sharpe']:.2f})", lw=1.0, alpha=0.7)
    ax.set_title(f"ALPHA-3 Regime-LGBM OOS Equity ({DATE_START} → {DATE_END})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve.png", dpi=130)
    plt.close(fig)

    # Feature importance top 10
    print("[ALPHA-3] Plotting feature importance")
    if feature_imp_acc:
        imp_sorted = sorted(feature_imp_acc.items(), key=lambda x: -x[1])[:10]
        names = [k for k, _ in imp_sorted]
        vals = [v for _, v in imp_sorted]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names[::-1], vals[::-1])
        ax.set_title("ALPHA-3 LightGBM — Top 10 Features (gain, sum across folds)")
        ax.set_xlabel("Importance (gain, cumulative)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "feature_importance.png", dpi=130)
        plt.close(fig)

        top_imp_dict = dict(imp_sorted)
    else:
        top_imp_dict = {}

    # ============================================================
    # RAPPORT.md
    # ============================================================
    print("[ALPHA-3] Writing RAPPORT.md")
    write_report(
        out_path=OUT_DIR / "RAPPORT.md",
        rec=rec,
        spa=spa,
        top_features=top_imp_dict,
        n_folds=len(folds),
        n_oos=int(oos_mask.sum()),
        fold_sharpes=fold_sharpes,
    )

    # Sauve aussi les fold-Sharpes
    pd.Series(fold_sharpes, name="fold_sharpe").to_csv(OUT_DIR / "fold_sharpes.csv", index=False)

    print("\n=== ALPHA-3 SUMMARY ===")
    print(f"Sharpe OOS (full):    {metrics_lgbm['sharpe']:.3f}")
    print(f"Median fold Sharpe:   {median_fold_sharpe:.3f}")
    print(f"DSR (10 trials):      {dsr_lgbm:.3f}")
    print(f"Hansen SPA p (cons.): {spa.p_value_consistent:.4f}")
    print(f"AUC ROC (dir):        {auc:.3f}" if not np.isnan(auc) else "AUC ROC (dir): n/a")
    print(f"Brier:                {brier:.4f}" if not np.isnan(brier) else "Brier: n/a")
    print(f"DEPLOY VERDICT:       {deploy}")
    print(f"Reason: {rec['reason']}")


def write_report(out_path: Path, rec: dict, spa, top_features: dict, n_folds: int, n_oos: int, fold_sharpes: list) -> None:
    m = rec["metrics_lgbm"]
    v1 = rec["metrics_v1"]
    hodl = rec["metrics_hodl"]
    fold_q1 = float(np.quantile(fold_sharpes, 0.25)) if fold_sharpes else float("nan")
    fold_q3 = float(np.quantile(fold_sharpes, 0.75)) if fold_sharpes else float("nan")

    feat_lines = "\n".join(
        [f"- **{k}**: {v:.0f}" for k, v in list(top_features.items())[:10]]
    ) or "- (aucune feature_importance disponible)"

    txt = f"""# ALPHA-3 — Regime-Aware LightGBM Backtest

## 1. Contexte

ALPHA-2 a invalide la piste Ichimoku K3 (Sharpe -1.91, deploy=no). Suite a la
litterature MDPI Electronics 15(6) 1334 (2025) "Regime-Aware LightGBM
Walk-Forward Framework" qui rapporte Sharpe portfolio crypto = 1.18
(IC95% [0.53, 1.84]), on tente une approche supervisee combinant un HMM
gaussien 2 etats avec un classifier LightGBM directional.

**Mise en garde** : l'IC95% [0.53, 1.84] du paper original est tres large.
Cela signifie que l'edge mesure est **fragile** — le borne basse 0.53 est a
peine au-dessus du seuil acceptable (0.5), donc une replication peut tres bien
tomber en dessous de zero apres frais reels.

## 2. Methodologie

- **Donnees** : BTC/USDT 2h, periode 2020-01-01 -> 2025-08-01 ({n_oos} bars OOS).
- **HMM** : `GaussianHMM(n_states=2)` fit sur (log_returns, vol_20) — fit
  uniquement sur le train fold, pas de regen sur le test.
- **Features (causaux)** : returns 1/4/12, vol_20, range%, ADX(14), ER(30),
  BBW(20)+squeeze, Hurst(100), encoding cyclique (hour, dow), funding current
  + change, momentum 6/12/24, plus la proba HMM `regime_p`.
- **Label** : sign(close[t+3]/close[t] - 1) en 3 classes {{-1, 0, +1}}, seuil
  flat = 0.2%. (Lookahead UNIQUEMENT sur le label, conformement aux best
  practices du supervised learning.)
- **Walk-forward** : {n_folds} folds, train expanding, test OOS = 2 mois,
  purge = 3 bars (= horizon du label, anti-leakage Lopez de Prado).
- **Couts** : `cost_model.apply_costs_to_returns` (fees 4 bps taker, slippage
  vol-aware, funding empirique).

## 3. Resultats OOS

### Metriques principales

| Metrique | Regime-LGBM | ALPHA-2 V1 | HODL |
| --- | --- | --- | --- |
| Sharpe (full OOS) | {m['sharpe']:.3f} | {v1['sharpe']:.3f} | {hodl['sharpe']:.3f} |
| Sortino | {m['sortino']:.3f} | {v1['sortino']:.3f} | {hodl['sortino']:.3f} |
| Calmar | {m['calmar']:.3f} | {v1['calmar']:.3f} | {hodl['calmar']:.3f} |
| CAGR | {m['cagr']:.2%} | {v1['cagr']:.2%} | {hodl['cagr']:.2%} |
| MDD | {m['mdd']:.2%} | {v1['mdd']:.2%} | {hodl['mdd']:.2%} |
| Ulcer | {m['ulcer']:.4f} | {v1['ulcer']:.4f} | {hodl['ulcer']:.4f} |
| dd_duration (bars) | {m['dd_duration']:.0f} | {v1['dd_duration']:.0f} | {hodl['dd_duration']:.0f} |

### Walk-forward stability

- **Median fold Sharpe** : {rec['median_fold_sharpe']:.3f}
- IQR : [{fold_q1:.3f}, {fold_q3:.3f}]
- Folds calcules : {len(fold_sharpes)} / {n_folds}

### Significativite statistique

- **DSR(10 trials)** : {rec['dsr_lgbm']:.3f}  (seuil 0.70 — {'OK' if rec['dsr_lgbm']>=0.7 else 'NOT OK'})
- **Hansen SPA p_consistent** : {spa.p_value_consistent:.4f}  (seuil 0.05 — {'OK' if spa.p_value_consistent<=0.05 else 'NOT OK'})
- **Hansen SPA p_lower** : {spa.p_value_lower:.4f}
- **Hansen SPA p_upper** : {spa.p_value_upper:.4f}

### Quality du classifier

- **AUC ROC (direction up vs down)** : {rec['auc_roc']:.3f}
- **Brier score** : {rec['brier_score']:.4f}

### Top 10 features (gain cumule)

{feat_lines}

## 4. Verdict honnete

**Decision : {rec['deploy'].upper()}**

Raison : {rec['reason']}

### Garde-fous appliques

- Sharpe OOS < 0.5 : {'TRIGGERED' if m['sharpe']<0.5 else 'OK'}
- DSR < 0.7 : {'TRIGGERED' if rec['dsr_lgbm']<0.7 else 'OK'}
- Hansen SPA p > 0.05 : {'TRIGGERED' if spa.p_value_consistent>0.05 else 'OK'}

### Notes critiques

1. Le paper MDPI 2025 (Sharpe 1.18, IC [0.53, 1.84]) a un IC tres large.
   L'edge rapporte est **fragile**. Une replication independante peut tomber
   sous zero apres frais reels.
2. Pas de SOPR ni MVRV utilise (on-chain non integre dans cette pipeline) —
   le paper original les utilisait. Une replication complete devrait les
   rajouter avant tout deploiement.
3. La purge de 3 bars (= label horizon) est minimale; certains auteurs
   (Lopez de Prado) recommandent une purge + embargo > 1 horizon pour
   crypto. A reverifier en sensibilite.
4. Le HMM 2 etats est volontairement simple (paper recommande 2-3). Tester
   3 etats peut etre utile mais ajoute du data-snooping si on optimise.

### Recommandation pour la suite

Si verdict = NO : ne PAS deployer en live. Garder ce backtest comme reference
negative et iterer (a) features on-chain (SOPR/MVRV), (b) horizon plus long
(12-24 bars = 1-2 jours), (c) ensembling avec strategy momentum simple.

Si verdict = YES : forward-test 30 jours testnet AVANT live, monitor le hit
rate par fold et le DSR mensuel.

## 5. Fichiers produits

- `equity_curve.png` — courbes equity LGBM vs V1 vs HODL
- `feature_importance.png` — top 10 features (gain)
- `recommendation.json` — verdict machine-readable
- `fold_sharpes.csv` — distribution Sharpe par fold OOS
- `RAPPORT.md` — ce document
"""
    out_path.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    run()
