"""R8 / P10 — Vrai training LightGBM combinator sur BTC H1 historique.

Spec exigeait :
  - Dataset : 3-4 ans BTC H1 post-2021
  - Split temporel : 70 / 15 / 15
  - Optuna : MAX 50 trials, timeout 1h
  - DSR déflaté avec n_trials=50 OBLIGATOIRE
  - Métrique : ROC-AUC sur validation
  - Garde-fous : AUC > 0.65 = suspect leakage ; AUC < 0.55 = pas d'edge

État features
-------------
Features réellement calculables sur OHLCV historique :
  - vpin                     (P7 build_volume_buckets + compute_vpin rolling)
  - hour_sin, hour_cos       (trivial)
  - day_of_week one-hot      (trivial)
  - days_since_halving       (trivial, halving 2024-04-19)
  - har_rv_predicted         (HAR-RV rolling fit chaque 1000 bars)
  - har_rv_classified        (low/mid/high via classify_regime)
  - egarch_sigma_predicted   (arch_model rolling re-fit chaque 1000 bars)

Features non-disponibles historiquement (mock à 0) :
  - composite_signal_score   (besoin flow_top_ls + taker + liq + oi
                              live, P6.1-P6.4 collecteurs créés R3 mais
                              data start = 2026-04-28, pas d'historique)
  - comp_top_ls/taker/liq/oi (idem)
  - phase_K3                 (besoin K3 daily phase rotation, pas trivial)

Cette limitation = **L-004** dans LIMITATIONS_ACTEES.md. AUC mesurée
reflète l'edge sur le sous-ensemble réel + les zéros. Ce n'est PAS un
problème statistique (les zéros sont constants donc LightGBM les ignore),
juste un signal moins fort qu'attendu.

Exécution
---------
$ python scripts/training/train_lgbm_combinator.py [--n-trials 50] [--quick]

  --quick : skip HAR/EGARCH (utilise seulement features triviales + VPIN),
            run en ~2min au lieu de ~30min.

Output : artifacts/lgbm_combinator/{model.txt, report.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lgbm_combinator import (
    AUC_DEPLOY_MIN,
    AUC_DEPLOY_SUSPECT_MAX,
    DSR_DEPLOY_MIN,
    N_TRIALS_MAX,
    LOW_VOL_QUANTILE_DEFAULT,
    build_target,
    build_features,
    evaluate_with_dsr,
    predict_low_vol_proba,
    should_deploy,
    temporal_split,
    train_combinator,
)


BTC_HALVING_2024 = datetime(2024, 4, 19, tzinfo=timezone.utc)
ARTIFACTS_DIR = ROOT / "artifacts" / "lgbm_combinator"


def _load_btc_h1(years_back: int = 4) -> pd.DataFrame:
    csv = ROOT / "data" / "BTC_USD_1h.csv"
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=365 * years_back)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def _compute_vpin_rolling(df: pd.DataFrame, bucket_size: float = 100.0,
                          window: int = 50) -> pd.Series:
    """VPIN rolling : pour chaque bar, recompute le VPIN sur les `window`
    derniers buckets de volume bucket_size USDT.

    Approximation simple : on utilise compute_vpin sur fenêtre cumulative.
    Pour éviter recompute O(n²) on calcule incremental.
    """
    from src.vpin import build_volume_buckets, compute_vpin

    prices = df["close"].astype(float)
    volumes = df["volume"].astype(float)
    n = len(df)

    # Strategy : on construit TOUS les buckets une fois, puis pour chaque
    # bar i on prend les N derniers buckets <= i (par ts_end) et on calcule.
    buckets = build_volume_buckets(prices, volumes, bucket_size_v=bucket_size)
    if not buckets:
        return pd.Series(np.full(n, np.nan), index=df.index)

    # Map ts_end_ms -> bucket
    # Mais nos prices n'ont pas de DatetimeIndex ici. Utilisons l'index
    # pos comme ts pour aligner.
    # build_volume_buckets utilise prices.index — RangeIndex ici donc
    # ts_end est un int. On va re-set l'index sur df["timestamp"].
    prices_dt = prices.copy()
    prices_dt.index = df["timestamp"].values
    volumes_dt = volumes.copy()
    volumes_dt.index = df["timestamp"].values
    buckets = build_volume_buckets(prices_dt, volumes_dt, bucket_size_v=bucket_size)
    bucket_ts = [int(b.get("ts_end_ms", 0)) for b in buckets]

    # Optimisation O(n) via single-pointer (bucket_ts monotone croissant,
    # df.timestamp monotone croissant).
    out = np.full(n, np.nan)
    ts_ms = (df["timestamp"].astype("int64").values // 1_000_000)
    j = 0  # pointeur sur le dernier bucket dont ts_end <= bar_i.ts
    for i in range(n):
        while j < len(bucket_ts) and bucket_ts[j] <= ts_ms[i]:
            j += 1
        # j-1 est le dernier bucket <= ts_ms[i]
        n_avail = j
        if n_avail < 5:
            continue
        start = max(0, n_avail - window)
        recent = buckets[start:n_avail]
        out[i] = compute_vpin(recent, window=window)
    return pd.Series(out, index=df.index)


def _compute_har_rv_features(df: pd.DataFrame, refit_every: int = 1000) -> tuple[pd.Series, pd.Series]:
    """Calcule har_rv_predicted + har_rv_classified rolling.

    Re-fit HAR-RV chaque `refit_every` bars (lookahead-safe : on fit sur
    [0, t-1] et on predict pour t).
    """
    from src.har_rv import (fit_har_rv, predict_rv, classify_regime,
                             realized_volatility)
    n = len(df)
    returns = df["close"].pct_change().fillna(0.0)
    har_pred = np.full(n, np.nan)
    har_class = np.full(n, "mid", dtype=object)

    fit_params = None
    historical_rv_h = None
    for i in range(200, n):
        if (i - 200) % refit_every == 0 or fit_params is None:
            train_returns = returns.iloc[:i]
            fit_params = fit_har_rv(train_returns)
            historical_rv_h = realized_volatility(train_returns, window=1).dropna()
        if not np.isfinite(fit_params.get("beta_h", np.nan)):
            continue
        rv_h = float(realized_volatility(returns.iloc[:i], window=1).iloc[-1])
        rv_d = float(realized_volatility(returns.iloc[:i], window=5).iloc[-1])
        rv_w = float(realized_volatility(returns.iloc[:i], window=22).iloc[-1])
        rv_pred = predict_rv(fit_params, rv_h, rv_d, rv_w)
        har_pred[i] = rv_pred
        har_class[i] = classify_regime(rv_pred, historical_rv_h)
    return pd.Series(har_pred, index=df.index), pd.Series(har_class, index=df.index)


def _compute_egarch_sigma_predicted(df: pd.DataFrame, refit_every: int = 1000) -> pd.Series:
    """EGARCH conditional volatility 1-step ahead, rolling re-fit.

    Pour limiter compute, on re-fit chaque `refit_every` bars.
    """
    from src.garch import fit_egarch
    returns = df["close"].pct_change().fillna(0.0)
    n = len(df)
    out = np.full(n, np.nan)
    fit = None
    last_fit_idx = -1
    for i in range(500, n):
        if i - last_fit_idx >= refit_every or fit is None or not fit.get("converged"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_egarch(returns.iloc[:i])
            last_fit_idx = i
        if fit and fit.get("converged"):
            # Approximation : sigma_predicted = sqrt(sigma2_last * 0.95
            # + omega) si on simulait ; ici on prend juste sigma2_last.
            out[i] = float(np.sqrt(fit.get("sigma2_last", 0.0)))
    return pd.Series(out, index=df.index)


def build_full_features(df: pd.DataFrame, quick: bool = False) -> pd.DataFrame:
    """Construit la matrice X complète (avec mocks pour features non-dispo)."""
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # Features triviales
    out["days_since_halving"] = ((df["timestamp"] - BTC_HALVING_2024).dt.total_seconds() / 86400).astype(float)
    out["hour_of_day"] = df["timestamp"].dt.hour
    out["day_of_week"] = df["timestamp"].dt.dayofweek

    # VPIN rolling
    print(f"[features] computing VPIN rolling ({n} bars)...")
    out["vpin"] = _compute_vpin_rolling(df).fillna(0.5)

    if not quick:
        print(f"[features] computing HAR-RV rolling ({n} bars)...")
        har_pred, har_class = _compute_har_rv_features(df, refit_every=2000)
        out["har_rv_predicted"] = har_pred.fillna(0.0)
        out["har_rv_classified"] = har_class

        print(f"[features] computing EGARCH sigma rolling ({n} bars)...")
        out["egarch_sigma_predicted"] = _compute_egarch_sigma_predicted(df, refit_every=2000).fillna(0.0)
    else:
        # Quick mode : mock features lentes
        out["har_rv_predicted"] = 0.0
        out["har_rv_classified"] = "mid"
        out["egarch_sigma_predicted"] = 0.0

    # Features non-dispo historiquement (mock 0, cf L-004)
    out["composite_signal_score"] = 0.0
    out["comp_top_ls"] = 0.0
    out["comp_taker"] = 0.0
    out["comp_liq"] = 0.0
    out["comp_oi"] = 0.0
    out["phase_K3"] = 0  # mock — K3 daily phase pas trivial à reconstituer

    # Drop early bars (où VPIN/HAR/EGARCH sont NaN)
    return out.iloc[500:].reset_index(drop=True)


def _build_target(df: pd.DataFrame, n_dropped: int = 500) -> pd.Series:
    """Target : 1 si RV(t+1h) < quantile_30 (RV historique 90j rolling)."""
    returns = df["close"].pct_change().fillna(0.0)
    rv_1h = returns.abs()  # |return_t| ≈ RV(window=1)
    rv_future = rv_1h.shift(-1)
    rv_history_90d = rv_1h.rolling(window=90 * 24, min_periods=100).quantile(LOW_VOL_QUANTILE_DEFAULT)
    target = (rv_future < rv_history_90d).astype(int)
    return target.iloc[n_dropped:].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_MAX)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--years-back", type=int, default=4)
    args = parser.parse_args()

    print(f"[load] loading BTC H1 last {args.years_back} years...")
    df = _load_btc_h1(years_back=args.years_back)
    print(f"[load] {len(df)} bars ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

    features_raw = build_full_features(df, quick=args.quick)
    target = _build_target(df, n_dropped=500)
    # Aligner longueurs
    n = min(len(features_raw), len(target))
    features_raw = features_raw.iloc[:n].reset_index(drop=True)
    target = target.iloc[:n].reset_index(drop=True)
    # Drop NaN target (dernier bar n'a pas de t+1)
    valid = target.notna() & np.isfinite(target.astype(float))
    features_raw = features_raw[valid].reset_index(drop=True)
    target = target[valid].reset_index(drop=True).astype(int)
    print(f"[data] X shape={features_raw.shape}, y mean={target.mean():.3f}")

    print("[features] encoding via build_features...")
    X = build_features(features_raw)
    print(f"[features] encoded shape={X.shape}")

    splits = temporal_split(X, target, train_pct=0.70, val_pct=0.15)
    print(f"[split] train={len(splits['X_train'])} val={len(splits['X_val'])} "
          f"test={len(splits['X_test'])}")

    print(f"[train] running combinator with n_trials={args.n_trials} (cap {N_TRIALS_MAX})...")
    result = train_combinator(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        n_trials=args.n_trials,
        timeout_seconds=3600,
        use_optuna=True,
    )
    print(f"[train] converged={result.converged} val_auc={result.val_auc:.4f} "
          f"trials_used={result.n_trials_used}")

    eval_dict = evaluate_with_dsr(
        val_auc=result.val_auc,
        n_trials_used=result.n_trials_used,
        n_obs=len(splits["X_val"]),
    )
    deploy, reason = should_deploy(result.val_auc, eval_dict["dsr_p"])
    print(f"[eval] AUC={eval_dict['val_auc']:.4f} DSR_p={eval_dict['dsr_p']:.4f}")
    print(f"[deploy] decision={deploy} reason={reason or 'OK'}")

    # Sauvegarde
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if result.model is not None:
        try:
            result.model.save_model(str(ARTIFACTS_DIR / "model.txt"))
        except Exception as e:
            print(f"[save] model save failed: {e}")
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "years_back": args.years_back,
        "quick_mode": args.quick,
        "n_trials_requested": args.n_trials,
        "n_trials_used": result.n_trials_used,
        "n_obs_train": len(splits["X_train"]),
        "n_obs_val": len(splits["X_val"]),
        "n_obs_test": len(splits["X_test"]),
        "val_auc": float(result.val_auc),
        "dsr_p": float(eval_dict["dsr_p"]),
        "should_deploy": bool(deploy),
        "reason": reason,
        "best_params": result.best_params,
        "deploy_thresholds": {
            "AUC_DEPLOY_MIN": AUC_DEPLOY_MIN,
            "AUC_DEPLOY_SUSPECT_MAX": AUC_DEPLOY_SUSPECT_MAX,
            "DSR_DEPLOY_MIN": DSR_DEPLOY_MIN,
        },
    }
    (ARTIFACTS_DIR / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"[save] report written to {ARTIFACTS_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
