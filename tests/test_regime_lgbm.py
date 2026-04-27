"""Tests pour ALPHA-3 RegimeLightGBM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import regime_lgbm as rlg


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2000
    idx = pd.date_range("2022-01-01", periods=n, freq="2h")

    # Two regimes: low vol (sigma=0.005), high vol (sigma=0.02), 500 bars each
    sigma = np.tile(np.repeat([0.005, 0.02], 500), 2)
    rets = rng.normal(0.0, sigma, size=n)
    close = 10000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    vol = rng.uniform(10, 100, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# 1. HMM converges on synthetic 2-regime data
# ---------------------------------------------------------------------------

def test_hmm_fit_converges(synth_df):
    log_close = np.log(synth_df["close"])
    log_returns = log_close.diff()
    realized_vol = log_returns.rolling(20).std()

    model, prob = rlg.fit_hmm_regimes(log_returns, realized_vol, n_states=2)

    # Le modele est fit
    assert model.means_.shape == (2, 2)
    assert prob.notna().sum() > 0
    # Les probas sont dans [0, 1]
    valid = prob.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
    # Les deux moyennes de vol doivent differer (le HMM separe high vs low vol)
    vol_means = model.means_[:, 1]
    assert abs(vol_means[0] - vol_means[1]) > 1e-4, (
        "HMM should separate the two synthetic vol regimes"
    )


# ---------------------------------------------------------------------------
# 2. build_features: NO LOOKAHEAD invariant
# ---------------------------------------------------------------------------

def test_build_features_no_lookahead(synth_df):
    """Truncating df at index t doit donner les memes features pour les bars 0..t
    qu'avec le df complet.
    """
    full_feats = rlg.build_features(synth_df)
    # Tronque a la moitie
    cut = len(synth_df) // 2
    trunc_df = synth_df.iloc[:cut].copy()
    trunc_feats = rlg.build_features(trunc_df)

    # Les features sur les bars communes [0..cut-1] doivent etre identiques
    common_idx = trunc_feats.index
    cols_to_check = [
        "ret_1", "ret_4", "ret_12", "vol_20",
        "adx_14", "er_30", "bbw_20", "hurst_100",
        "mom_6", "mom_12", "mom_24",
    ]
    for col in cols_to_check:
        a = full_feats.loc[common_idx, col]
        b = trunc_feats[col]
        # NaN-aware compare
        mask = a.notna() & b.notna()
        assert mask.sum() > 100, f"too few comparable values for {col}"
        diff = (a[mask] - b[mask]).abs()
        assert diff.max() < 1e-9, (
            f"LOOKAHEAD detected in {col}: max diff={diff.max():.2e}"
        )


# ---------------------------------------------------------------------------
# 3. label_directional returns {-1, 0, +1}
# ---------------------------------------------------------------------------

def test_label_directional_values(synth_df):
    y = rlg.label_directional(synth_df["close"], horizon=3, flat_threshold=0.001)
    unique = set(np.unique(y))
    assert unique.issubset({-1, 0, 1}), f"unexpected label values {unique}"
    # Le label = sign(forward return), donc on doit avoir un mix des trois
    assert (y == 1).sum() > 0
    assert (y == -1).sum() > 0


# ---------------------------------------------------------------------------
# 4. LightGBM fit + predict shape
# ---------------------------------------------------------------------------

def test_lgbm_fit_predict_shapes(synth_df):
    feats = rlg.build_features(synth_df)
    y = rlg.label_directional(synth_df["close"], horizon=3, flat_threshold=0.001)

    log_returns = np.log(synth_df["close"]).diff()
    realized_vol = log_returns.rolling(20).std()
    _, regime = rlg.fit_hmm_regimes(log_returns, realized_vol, n_states=2)

    # Drop NaN rows from features+label+regime
    aligned = pd.concat(
        [feats, y.rename("y"), regime.rename("r")], axis=1
    ).dropna()
    if len(aligned) < 200:
        pytest.skip("not enough valid data for LGBM smoke test")

    X = aligned.drop(columns=["y", "r"])
    y_align = aligned["y"]
    r_align = aligned["r"]

    train_size = int(0.7 * len(aligned))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y_align.iloc[:train_size], y_align.iloc[train_size:]
    r_train, r_test = r_align.iloc[:train_size], r_align.iloc[train_size:]

    cfg = rlg.RegimeLGBMConfig(num_boost_round=30, early_stopping_rounds=10)
    clf = rlg.RegimeLightGBM(cfg)
    clf.fit(X_train, y_train, r_train)
    preds = clf.predict(X_test, r_test)
    proba = clf.predict_proba(X_test, r_test)

    assert preds.shape == (len(X_test),)
    assert proba.shape == (len(X_test), 3)
    assert set(np.unique(preds)).issubset({-1, 0, 1})
    # Probas somment a 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    # Feature importance accessible
    imp = clf.feature_importance("gain")
    assert "regime_p" in imp
    assert len(imp) == X_train.shape[1] + 1  # +1 pour regime_p


# ---------------------------------------------------------------------------
# 5. Smoke pipeline bout-en-bout
# ---------------------------------------------------------------------------

def test_pipeline_smoke(synth_df):
    """Pipeline minimal: features -> HMM -> label -> LightGBM -> signal."""
    feats = rlg.build_features(synth_df)
    log_returns = np.log(synth_df["close"]).diff()
    realized_vol = log_returns.rolling(20).std()
    _, regime = rlg.fit_hmm_regimes(log_returns, realized_vol, n_states=2)
    y = rlg.label_directional(synth_df["close"], horizon=3, flat_threshold=0.001)

    aligned = pd.concat(
        [feats, y.rename("y"), regime.rename("r")], axis=1
    ).dropna()
    assert len(aligned) > 100, "Not enough data after alignment"

    split = int(0.8 * len(aligned))
    X = aligned.drop(columns=["y", "r"])
    cfg = rlg.RegimeLGBMConfig(num_boost_round=20, early_stopping_rounds=5)
    clf = rlg.RegimeLightGBM(cfg)
    clf.fit(
        X.iloc[:split], aligned["y"].iloc[:split], aligned["r"].iloc[:split]
    )
    sig = clf.predict(X.iloc[split:], aligned["r"].iloc[split:])
    assert len(sig) == len(aligned) - split
