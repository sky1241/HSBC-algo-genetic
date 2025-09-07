from __future__ import annotations

import sys
from pathlib import Path
import zipfile
import argparse
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Import helpers from repo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_hmm_regimes import build_feature_matrix  # type: ignore


def read_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])  # type: ignore[arg-type]
    if "timestamp" not in df.columns:
        raise RuntimeError(f"Missing 'timestamp' in {csv_path}")
    df = df.set_index("timestamp").sort_index()
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"Missing OHLC columns in {csv_path}: need {need}")
    return df


def run_hmm_for_k(X: pd.DataFrame, dates: pd.DatetimeIndex, K: int, seed: int, train_frac: float):
    n = len(X)
    split = int(n * train_frac)
    Xtr, Xte = X.iloc[:split].to_numpy(), X.iloc[split:].to_numpy()

    hmm = GaussianHMM(
        n_components=K,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        random_state=int(seed),
    )
    hmm.fit(Xtr)
    ll_tr = float(hmm.score(Xtr))
    ll_te = float(hmm.score(Xte)) if len(Xte) else np.nan

    # Param count approx (means + diag covars + transitions + startprob)
    p = K * X.shape[1] + K * X.shape[1] + K * (K - 1) + (K - 1)
    aic = 2 * p - 2 * ll_tr
    bic = p * np.log(max(1, len(Xtr))) - 2 * ll_tr

    states = hmm.predict(X.to_numpy())
    pred = pd.DataFrame({"timestamp": dates, "state": states})
    return aic, bic, ll_te, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC_FUSED")
    ap.add_argument("--tf", default="2h")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=30)  # 30×2 batches
    ap.add_argument("--win-bars", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--k-major", action="store_true", help="Loop K outermost and checkpoint per-K backups")
    args = ap.parse_args()

    sym, tf = args.symbol, args.tf
    csv = Path("data") / f"{sym}_{tf}.csv"
    if not csv.exists():
        # Fallback if fused missing
        alt = Path("data") / f"BTC_USDT_{tf}.csv"
        if alt.exists():
            csv = alt
        else:
            print(f"Missing data: {csv} (and fallback {alt})")
            return 1

    print(f"[INFO] Loading: {csv}", flush=True)
    df = read_ohlcv_csv(csv)
    bars_per_day = 12 if tf.lower() == "2h" else 1
    print(f"[INFO] Building features (win={args.win_bars}, bpd={bars_per_day}) …", flush=True)
    X = build_feature_matrix(df, bars_per_day=bars_per_day, win_bars=args.win_bars)

    out_root = Path("outputs") / "fourier" / "hmm" / f"{sym}_{tf}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Two batches of 30 distinct seeds each (60 total)
    seeds_b1 = [123 + 13 * i for i in range(args.n_seeds)]
    seeds_b2 = [650 + 17 * i for i in range(args.n_seeds)]
    seeds = seeds_b1 + seeds_b2

    k_list = list(range(args.k_min, args.k_max + 1))
    print(f"[INFO] Running {len(seeds)} seeds × {len(k_list)} K values …", flush=True)

    all_rows = []

    if args.k_major:
        # K‑major loop: finalize a full per‑K backup when each K is done across all seeds
        rows_per_seed: dict[int, list[dict[str, float]]] = {int(s): [] for s in seeds}
        by_k_root = out_root / "by_K"
        by_k_root.mkdir(parents=True, exist_ok=True)

        for ki, K in enumerate(k_list, 1):
            print(f"=== K {K} ({ki}/{len(k_list)}) ===", flush=True)
            rows_k = []
            for si, seed in enumerate(seeds, 1):
                try:
                    aic, bic, ll, pred = run_hmm_for_k(X, X.index, K, seed, args.train_frac)
                    seed_dir = out_root / f"seed_{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    pred.to_csv(seed_dir / f"HMM_PRED_{K}.csv", index=False)
                    row = {"seed": seed, "K": K, "AIC": aic, "BIC": bic, "LL_OOS": ll}
                    rows_k.append(row)
                    rows_per_seed[int(seed)].append(row)
                    all_rows.append(row)
                    print(f"[K {K}] seed={seed} AIC={aic:.0f} BIC={bic:.0f} LL_OOS={ll:.2f}", flush=True)
                except Exception as e:
                    print(f"[K {K}] seed={seed} ERROR: {e}", flush=True)

            # Per‑K checkpoint: CSV + quick stats + zip of predictions for this K
            k_dir = by_k_root / f"K{K}"
            k_dir.mkdir(parents=True, exist_ok=True)

            df_k = pd.DataFrame(rows_k)
            df_k.to_csv(k_dir / f"HMM_K_SELECTION_K{K}_ALL_SEEDS.csv", index=False)

            if not df_k.empty:
                agg_k = (
                    df_k.agg({"AIC": ["mean", "median"], "BIC": ["mean", "median"], "LL_OOS": ["mean", "median"]})
                    .T
                    .rename(columns={"mean": "mean", "median": "median"})
                )
                agg_k.to_csv(k_dir / f"AGG_K{K}.csv")

            # Zip all prediction files for this K across seeds
            zip_path = k_dir / f"PRED_K{K}.zip"
            try:
                with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for seed in seeds:
                        seed_dir = out_root / f"seed_{seed}"
                        pred_path = seed_dir / f"HMM_PRED_{K}.csv"
                        if pred_path.exists():
                            zf.write(pred_path, arcname=f"seed_{seed}.csv")
            except Exception as e:
                print(f"[WARN] Zip failed for K={K}: {e}", flush=True)

        # After all K done, write per‑seed summaries for completeness
        for seed, rows in rows_per_seed.items():
            seed_dir = out_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(seed_dir / f"HMM_K_SELECTION_seed{seed}.csv", index=False)
    else:
        # Original seed‑major loop
        for si, seed in enumerate(seeds, 1):
            print(f"=== Seed {seed} ({si}/{len(seeds)}) ===", flush=True)
            seed_dir = out_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for K in k_list:
                try:
                    aic, bic, ll, pred = run_hmm_for_k(X, X.index, K, seed, args.train_frac)
                    pred.to_csv(seed_dir / f"HMM_PRED_{K}.csv", index=False)
                    rows.append({"seed": seed, "K": K, "AIC": aic, "BIC": bic, "LL_OOS": ll})
                    print(f"[seed {seed}] K={K} AIC={aic:.0f} BIC={bic:.0f} LL_OOS={ll:.2f}", flush=True)
                except Exception as e:
                    print(f"[seed {seed}] K={K} ERROR: {e}", flush=True)
            pd.DataFrame(rows).to_csv(seed_dir / f"HMM_K_SELECTION_seed{seed}.csv", index=False)
            all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    agg = (
        df_all.groupby("K")
        .agg(
            AIC_mean=("AIC", "mean"),
            AIC_median=("AIC", "median"),
            BIC_mean=("BIC", "mean"),
            BIC_median=("BIC", "median"),
            LL_mean=("LL_OOS", "mean"),
            LL_median=("LL_OOS", "median"),
            count=("LL_OOS", "count"),
        )
        .reset_index()
        .sort_values(["BIC_median", "AIC_median"])
    )
    agg.to_csv(out_root / "HMM_K_SELECTION_AGG.csv", index=False)

    # Markdown summary
    try:
        md = []
        md.append(f"### HMM K‑selection (seeds={len(seeds)}) — {sym} {tf}")
        md.append("")
        md.append(agg.to_markdown(index=False))
        best_k = int(agg.iloc[0]["K"])
        md.append("")
        md.append(f"Recommandation (BIC_median): K={best_k}")
        (Path("docs") / f"HMM_{sym}_{tf}.md").write_text("\n".join(md), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] MD write failed: {e}", flush=True)

    print(f"[DONE] Wrote: {out_root/'HMM_K_SELECTION_AGG.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


