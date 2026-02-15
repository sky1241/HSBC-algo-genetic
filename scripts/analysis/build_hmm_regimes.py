#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estime un HMM (GaussianHMM, diag cov) sur features spectrales + Ichimoku et
sélectionne K ∈ {3,4,5,6} via AIC/BIC + log-vraisemblance out-of-sample (walk-forward).

Entrées:
- data/<SYMBOL>_2h.csv (timestamp, open, high, low, close, volume)

Sorties:
- outputs/fourier/hmm/<SYMBOL>_2h/HMM_K_SELECTION.csv  (K, AIC, BIC, LL_OOS)
- outputs/fourier/hmm/<SYMBOL>_2h/HMM_PRED_<K>.csv     (dates, state)
- docs/HMM_<SYMBOL>_2h.md (résumé + reco K)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Import helpers from repo
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.plot_phase_price import read_ohlcv  # type: ignore
from scripts.phase_aware_module import scale_ichimoku  # type: ignore
from scripts.fourier_core import compute_welch_psd  # type: ignore


def _stft_window_features(x: np.ndarray, fs: float, n: int) -> dict:
    """Features spectrales sur fenêtre: centroid, entropy, low/high band-power."""
    freqs, psd = compute_welch_psd(x[-n:], fs=fs)
    if len(freqs) == 0:
        return {"centroid": np.nan, "entropy": np.nan, "low_share": np.nan, "high_share": np.nan}
    # Avoid epsilon bias: derive probabilities from strictly positive mass only
    total = float(psd.sum())
    if total <= 0:
        return {"centroid": float("nan"), "entropy": float("nan"), "low_share": float("nan"), "high_share": float("nan")}
    p = psd / total
    p = np.where(p > 0, p, 0.0)
    centroid = float((freqs * p).sum())
    entropy = float(-(p * np.log(p)).sum())
    # low/high split around 1/10 day^-1 (≈ cycles > 10 jours en H2)
    low_mask = freqs < (1.0 / 10.0)
    low_share = float(psd[low_mask].sum() / psd.sum())
    high_share = 1.0 - low_share
    return {"centroid": centroid, "entropy": entropy, "low_share": low_share, "high_share": high_share}


def _ichimoku_features(df: pd.DataFrame, bars_per_day: int = 12) -> pd.DataFrame:
    t, k, sb = scale_ichimoku(bars_per_day=bars_per_day)
    # Tenkan/Kijun de phase_aware_module ≈ rolling middle of highs/lows
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(t, min_periods=t).max() + low.rolling(t, min_periods=t).min()) / 2.0
    kijun  = (high.rolling(k, min_periods=k).max() + low.rolling(k, min_periods=k).min()) / 2.0
    kumo_top = ((tenkan + kijun) / 2.0).rolling(sb, min_periods=sb).max()
    kumo_bot = ((tenkan + kijun) / 2.0).rolling(sb, min_periods=sb).min()
    m_ichi = tenkan / kijun - 1.0
    dist_kumo = (df["close"] - (kumo_top + kumo_bot) / 2.0) / df["close"].abs().rolling(k, min_periods=k).mean()
    out = pd.DataFrame({
        "M_ICH": m_ichi,
        "CloudSpan": (kumo_top - kumo_bot),
        "DistKumo": dist_kumo,
    }, index=df.index)
    return out


def build_feature_matrix(df: pd.DataFrame, bars_per_day: int = 12, win_bars: int = 256) -> pd.DataFrame:
    close = df["close"].astype(float)
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    feats = []
    for i in range(len(close)):
        if i < win_bars:
            feats.append({"centroid": np.nan, "entropy": np.nan, "low_share": np.nan, "high_share": np.nan})
            continue
        seg = ret[i - win_bars : i]
        feats.append(_stft_window_features(seg, fs=1.0, n=len(seg)))
    spec = pd.DataFrame(feats, index=df.index)
    ichi = _ichimoku_features(df, bars_per_day=bars_per_day)
    X = pd.concat([spec, ichi], axis=1)
    # standardize per-column (robust: median/MAD)
    med = X.median()
    mad = (X - med).abs().median()
    denom = mad.replace(0, np.nan).fillna(1.0)
    Xs = (X - med) / (denom + 1e-9)
    return Xs.dropna()


def k_selection(X: pd.DataFrame, dates: pd.DatetimeIndex, K_list: List[int], train_frac: float = 0.7) -> pd.DataFrame:
    n = len(X)
    split = int(n * train_frac)
    Xtr, Xte = X.iloc[:split].to_numpy(), X.iloc[split:].to_numpy()
    rows = []
    for K in K_list:
        hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, tol=1e-3, random_state=42)
        hmm.fit(Xtr)
        ll_tr = float(hmm.score(Xtr))
        ll_te = float(hmm.score(Xte)) if len(Xte) else np.nan
        # AIC/BIC: 2p - 2LL ; p approx params (means + covars + trans + startprob)
        p = K * X.shape[1]          # means
        p += K * X.shape[1]         # diag covars
        p += K * (K - 1)            # transitions (row-stochastic)
        p += (K - 1)                # startprob
        aic = 2 * p - 2 * ll_tr
        bic = p * np.log(max(1, len(Xtr))) - 2 * ll_tr
        rows.append({"K": K, "AIC": aic, "BIC": bic, "LL_OOS": ll_te})
        # write per‑bar states OOS for inspection
        states = hmm.predict(X)
        pred = pd.DataFrame({"timestamp": dates, "state": states})
        outp = Path("outputs") / "fourier" / "hmm" / f"BTC_USDT_2h" / f"HMM_PRED_{K}.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(outp, index=False)
    return pd.DataFrame(rows).sort_values(["BIC", "AIC"]).reset_index(drop=True)


def main() -> int:
    sym, tf = "BTC_USDT", "2h"
    csv = Path("data") / f"{sym}_{tf}.csv"
    if not csv.exists():
        print("Missing:", csv)
        return 1
    df = read_ohlcv(csv)
    X = build_feature_matrix(df, bars_per_day=12, win_bars=256)
    sel = k_selection(X, X.index, K_list=[3,4,5,6])
    out_dir = Path("outputs") / "fourier" / "hmm" / f"{sym}_{tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "HMM_K_SELECTION.csv"
    sel.to_csv(out_csv, index=False)
    # write a short MD summary
    best = sel.iloc[0]
    md = [
        f"### HMM K‑selection — {sym} {tf}",
        "",
        sel.to_markdown(index=False),
        "",
        f"Recommandation: K={int(best['K'])} (BIC minimal), LL_OOS={best['LL_OOS']:.2f}",
    ]
    (Path("docs") / f"HMM_{sym}_{tf}.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote:", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


