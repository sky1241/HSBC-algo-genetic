import numpy as np
import pandas as pd
import pytest

from src.regime_hmm import apply_hmm, fit_hmm, rules_fourier


def _build_cluster(rng, loc, scale, size):
    return rng.normal(loc, scale, size=(size, 2))


def test_fit_hmm_rejects_rare_states():
    rng = np.random.default_rng(42)
    data = np.vstack([
        _build_cluster(rng, loc=-4.0, scale=0.2, size=60),
        _build_cluster(rng, loc=4.0, scale=0.2, size=60),
        _build_cluster(rng, loc=0.0, scale=0.05, size=3),
    ])
    train_df = pd.DataFrame(data, columns=["f1", "f2"])
    durations = pd.Series([1.0] * 120 + [1.0] * 3, index=train_df.index)
    trades = pd.Series([5.0] * 120 + [1.0] * 3, index=train_df.index)

    with pytest.raises(ValueError, match="rare states"):
        fit_hmm(
            train_df,
            K=[3],
            seed=0,
            durations=durations,
            trade_counts=trades,
            min_duration_frac=0.05,
            min_trade_frac=0.05,
            min_trades=5.0,
            min_obs=10,
        )


def test_apply_hmm_oos_prediction_without_lookahead():
    rng = np.random.default_rng(0)
    train_data = np.vstack([
        _build_cluster(rng, loc=-1.0, scale=0.3, size=40),
        _build_cluster(rng, loc=1.0, scale=0.3, size=40),
    ])
    test_data = np.vstack([
        _build_cluster(rng, loc=-1.0, scale=0.25, size=10),
        _build_cluster(rng, loc=1.0, scale=0.25, size=10),
    ])

    train_df = pd.DataFrame(train_data, columns=["feat1", "feat2"])
    train_df["duration"] = 1.0
    train_df["n_trades"] = 5.0
    train_df["P1"] = np.linspace(100, 200, len(train_df))
    train_df["LFP"] = np.linspace(0.6, 0.9, len(train_df))

    test_df = pd.DataFrame(test_data, columns=["feat1", "feat2"])
    test_df["duration"] = 1.0
    test_df["n_trades"] = 5.0
    test_df["P1"] = np.linspace(90, 210, len(test_df))
    test_df["LFP"] = np.linspace(0.55, 0.95, len(test_df))

    cfg = {
        "feature_cols": ["feat1", "feat2"],
        "duration_col": "duration",
        "trade_count_col": "n_trades",
        "K": [2],
        "seed": 123,
        "min_duration_frac": 0.01,
        "min_trade_frac": 0.01,
        "min_trades": 5.0,
        "min_obs": 5,
        "fourier_cols": {"P1": "P1", "LFP": "LFP"},
        "return_train_states": True,
    }

    result = apply_hmm(train_df, test_df, cfg)
    assert result["model"] is not None
    assert len(result["oos_states"]) == len(test_df)
    assert list(result["oos_states"].index) == list(test_df.index)

    train_summary = result["train_summary"]
    assert pytest.approx(train_summary["duration"].sum()) == train_df["duration"].sum()
    assert pytest.approx(train_summary["trades"].sum()) == train_df["n_trades"].sum()
    assert pytest.approx(train_summary["duration_frac"].sum()) == 1.0
    assert pytest.approx(train_summary["trade_frac"].sum()) == 1.0

    scaler = result["scaler"]
    expected_median = train_df[["feat1", "feat2"]].median()
    assert scaler["median"].equals(expected_median)

    # The decoded training states are available for diagnostics only.
    assert len(result["train_states"]) == len(train_df)


def test_apply_hmm_fallback_rules_fourier():
    rng = np.random.default_rng(1234)
    train_values = np.vstack([
        _build_cluster(rng, loc=-2.0, scale=0.2, size=30),
        _build_cluster(rng, loc=2.0, scale=0.2, size=5),
    ])
    test_values = np.vstack([
        _build_cluster(rng, loc=-2.0, scale=0.2, size=5),
        _build_cluster(rng, loc=2.0, scale=0.2, size=5),
    ])

    train_df = pd.DataFrame(train_values, columns=["feat1", "feat2"])
    train_df["duration"] = [2.0] * 30 + [0.1] * 5
    train_df["n_trades"] = [10.0] * 30 + [0.5] * 5
    train_df["P1"] = np.linspace(80, 160, len(train_df))
    train_df["LFP"] = np.linspace(0.5, 0.95, len(train_df))

    test_df = pd.DataFrame(test_values, columns=["feat1", "feat2"])
    test_df["P1"] = np.linspace(70, 170, len(test_df))
    test_df["LFP"] = np.linspace(0.45, 0.97, len(test_df))

    cfg = {
        "feature_cols": ["feat1", "feat2"],
        "duration_col": "duration",
        "trade_count_col": "n_trades",
        "K": [3],
        "seed": 7,
        "min_duration_frac": 0.15,
        "min_trade_frac": 0.15,
        "min_trades": 6.0,
        "min_obs": 10,
        "fourier_cols": {"P1": "P1", "LFP": "LFP"},
        "quantiles": (0.25, 0.75),
    }

    result = apply_hmm(train_df, test_df, cfg)
    assert result["model"] is None
    assert result["fallback"] == "rules_fourier"
    assert "rare" in result["reason"].lower()

    # Compute the expected Fourier-based regimes manually (train quantiles only).
    p1_low, p1_high = train_df["P1"].quantile(0.25), train_df["P1"].quantile(0.75)
    lfp_low, lfp_high = train_df["LFP"].quantile(0.25), train_df["LFP"].quantile(0.75)
    expected_states = []
    for _, row in test_df.iterrows():
        if row["LFP"] >= lfp_high and row["P1"] >= p1_high:
            expected_states.append("trend")
        elif row["LFP"] <= lfp_low or row["P1"] <= p1_low:
            expected_states.append("range")
        else:
            expected_states.append("transition")
    assert list(result["oos_states"]) == expected_states

    # Direct call to the fallback for completeness.
    fallback_direct = rules_fourier(train_df, test_df, cfg)
    pd.testing.assert_series_equal(result["oos_states"], fallback_direct)
