from __future__ import annotations

import numpy as np
import pandas as pd

from src import regime_hmm


def _make_features(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    p1 = 20 + rng.normal(0, 1.5, size=n).cumsum()
    lfp = 0.5 + 0.05 * rng.normal(size=n)
    vol = 0.02 + 0.01 * rng.random(size=n)
    return pd.DataFrame(
        {
            "P1_period": p1,
            "LFP_ratio": lfp,
            "volatility": vol,
        },
        index=index,
    )


def test_apply_hmm_returns_states() -> None:
    features = _make_features(240, seed=1)
    train = features.iloc[:160]
    test = features.iloc[160:]
    cfg = {
        "feature_cols": ["P1_period", "LFP_ratio", "volatility"],
        "K": (2, 3),
        "seed": 42,
        "fourier_cols": {"P1": "P1_period", "LFP": "LFP_ratio"},
        "quantiles": (0.25, 0.75),
        "return_train_states": True,
    }
    result = regime_hmm.apply_hmm(train, test, cfg)
    assert result["model"] is not None
    assert "train_states" in result
    assert not result["oos_states"].isna().all()
    assert set(result["oos_states"].dropna().unique()) <= {0, 1, 2}


def test_apply_hmm_fallback_rules_fourier() -> None:
    features = _make_features(120, seed=2)
    train = features.iloc[:80]
    test = features.iloc[80:]
    cfg = {
        "feature_cols": ["P1_period", "LFP_ratio", "volatility"],
        "K": (3,),
        "min_obs": 500,  # Force rejection of all HMM candidates
        "fourier_cols": {"P1": "P1_period", "LFP": "LFP_ratio"},
        "quantiles": (0.25, 0.75),
        "return_train_states": True,
    }
    result = regime_hmm.apply_hmm(train, test, cfg)
    assert result["model"] is None
    assert result.get("fallback") == "rules_fourier"
    assert set(result["oos_states"].unique()) <= {"trend", "range", "transition", "unknown"}
    assert set(result["train_states"].unique()) <= {"trend", "range", "transition", "unknown"}
