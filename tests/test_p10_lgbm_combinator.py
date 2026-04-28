"""P10 — Tests LightGBM combinator P(low_vol_next_hour)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lgbm_combinator import (
    AUC_DEPLOY_MIN,
    AUC_DEPLOY_SUSPECT_MAX,
    DSR_DEPLOY_MIN,
    N_TRIALS_MAX,
    build_features,
    build_target,
    evaluate_with_dsr,
    predict_low_vol_proba,
    should_deploy,
    temporal_split,
    train_combinator,
)


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _synth_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regimes = rng.choice(["low", "mid", "high"], size=n, p=[0.3, 0.5, 0.2])
    df = pd.DataFrame({
        "har_rv_predicted": rng.uniform(0.005, 0.05, n),
        "har_rv_classified": regimes,
        "egarch_sigma_predicted": rng.uniform(0.01, 0.06, n),
        "vpin": rng.uniform(0.0, 1.0, n),
        "composite_signal_score": rng.uniform(-1.0, 1.0, n),
        "comp_top_ls": rng.normal(0, 1, n),
        "comp_taker": rng.normal(0, 1, n),
        "comp_liq": rng.normal(0, 1, n),
        "comp_oi": rng.normal(0, 1, n),
        "phase_K3": rng.integers(0, 3, n),
        "days_since_halving": rng.integers(0, 1500, n),
        "hour_of_day": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
    })
    return df


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_features_shape_matches_spec():
    """X doit contenir : 9 numériques + 3 one-hot regime + 3 one-hot phase
    + 2 cyclic hour + 7 one-hot dow = 24 colonnes."""
    df = _synth_df(100)
    X = build_features(df)
    expected_cols = {
        "har_rv_predicted", "egarch_sigma_predicted", "vpin",
        "composite_signal_score", "comp_top_ls", "comp_taker",
        "comp_liq", "comp_oi", "days_since_halving",
        "har_regime_low", "har_regime_mid", "har_regime_high",
        "phase_K3_0", "phase_K3_1", "phase_K3_2",
        "hour_sin", "hour_cos",
        "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    }
    assert set(X.columns) == expected_cols
    assert len(X) == 100


def test_features_missing_column_raises():
    df = _synth_df(50).drop(columns=["vpin"])
    with pytest.raises(KeyError, match="vpin"):
        build_features(df)


def test_features_cyclic_hour_encoding_correct():
    """hour=0 → sin=0, cos=1 ; hour=6 → sin=1, cos≈0."""
    df = _synth_df(10)
    df.loc[0, "hour_of_day"] = 0
    df.loc[1, "hour_of_day"] = 6
    X = build_features(df)
    assert X["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert X["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert X["hour_sin"].iloc[1] == pytest.approx(1.0, abs=1e-9)
    assert X["hour_cos"].iloc[1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# build_target
# ---------------------------------------------------------------------------


def test_target_binary_only():
    """y doit être ∈ {0, 1} uniquement."""
    rv_future = pd.Series(np.random.uniform(0.005, 0.05, 200))
    rv_history = pd.Series(np.random.uniform(0.005, 0.05, 200))
    y = build_target(rv_future, rv_history, quantile=0.30)
    assert set(y.unique()).issubset({0, 1})


def test_target_low_quantile_yields_few_positives():
    """quantile=0.30 sur distribution uniforme → ~30% de 1s."""
    rng = np.random.default_rng(0)
    rv_future = pd.Series(rng.uniform(0, 1, 5000))
    rv_history = pd.Series(rng.uniform(0, 1, 5000))
    y = build_target(rv_future, rv_history, quantile=0.30)
    assert 0.20 < y.mean() < 0.40


def test_target_invalid_quantile():
    with pytest.raises(ValueError):
        build_target(pd.Series([0.01]), pd.Series([0.01]), quantile=0.0)
    with pytest.raises(ValueError):
        build_target(pd.Series([0.01]), pd.Series([0.01]), quantile=1.0)


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


def test_temporal_split_no_leakage():
    """Test set ne doit contenir aucun index < val_end."""
    df = _synth_df(1000)
    X = build_features(df)
    y = pd.Series(np.random.randint(0, 2, 1000))
    splits = temporal_split(X, y, train_pct=0.70, val_pct=0.15)
    # Indices : train=[0,699], val=[700,849], test=[850,999]
    assert splits["X_train"].index.max() == 699
    assert splits["X_val"].index.min() == 700
    assert splits["X_val"].index.max() == 849
    assert splits["X_test"].index.min() == 850
    # Pas d'overlap d'index
    assert set(splits["X_train"].index).isdisjoint(set(splits["X_val"].index))
    assert set(splits["X_val"].index).isdisjoint(set(splits["X_test"].index))


def test_temporal_split_invalid_pcts():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError):
        temporal_split(X, y, train_pct=0.9, val_pct=0.2)  # somme >= 1


# ---------------------------------------------------------------------------
# train_combinator (Optuna max 50)
# ---------------------------------------------------------------------------


def test_optuna_max_50_trials_hard_cap():
    """N_TRIALS_MAX = 50 : even if user requests 100, le clamp limite à 50."""
    assert N_TRIALS_MAX == 50

    df = _synth_df(500, seed=3)
    X = build_features(df)
    y = pd.Series(np.random.RandomState(3).randint(0, 2, 500))
    splits = temporal_split(X, y)

    with patch("src.lgbm_combinator.HAS_OPTUNA", True), \
         patch("src.lgbm_combinator.optuna") as mock_optuna:
        # Mock Optuna study qui retourne best_params constants
        mock_study = MagicMock()
        mock_study.best_params = {
            "lr": 0.05, "num_leaves": 31, "ff": 0.8, "bf": 0.8, "min_data": 20,
        }
        mock_study.trials = [MagicMock()] * 50
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = MagicMock()

        train_combinator(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
            n_trials=100,  # demande 100, cap à 50
        )
        # study.optimize doit être appelé avec n_trials=50 (capped)
        call_kwargs = mock_study.optimize.call_args.kwargs
        assert call_kwargs.get("n_trials") == 50


def test_optuna_real_clamp_runs_actual_study():
    """F-tests INSTR-4 : run RÉELLEMENT Optuna (sans mock) avec n_trials=3
    + cap N_TRIALS_MAX. Vérifie n_trials_used <= cap.

    Complète le test mock-only précédent en exerçant le path Optuna réel.
    Utilise un dataset minimal et n_trials petit pour rester rapide (~10s).
    """
    df = _synth_df(400, seed=99)
    X = build_features(df)
    # Cible faiblement corrélée pour avoir un fit qui converge
    y = (df["har_rv_predicted"] < df["har_rv_predicted"].quantile(0.30)).astype(int)
    splits = temporal_split(X, y)

    requested_trials = 3  # bien < N_TRIALS_MAX donc pas clamped
    result = train_combinator(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        n_trials=requested_trials,
        timeout_seconds=60,
        use_optuna=True,
    )
    # Optuna a vraiment tourné (pas de mock)
    assert result.n_trials_used <= requested_trials  # ≤ 3 (peut être moins si timeout)
    assert result.n_trials_used >= 1  # au moins 1 trial complété
    assert result.n_trials_used <= N_TRIALS_MAX  # respecte le cap absolu
    assert 0.0 <= result.val_auc <= 1.0
    assert result.converged is True


def test_optuna_real_clamp_high_request_clamped_to_max():
    """F-tests INSTR-4 : n_trials=N_TRIALS_MAX+5 → réellement clamped à
    N_TRIALS_MAX. NB: on utilise dataset très petit + timeout court pour
    éviter de tourner 50 trials réels (déjà couvert par test_optuna_max
    via mock). Ici on vérifie juste que train_combinator NE LAISSE PAS
    passer n_trials > N_TRIALS_MAX.
    """
    df = _synth_df(150, seed=33)
    X = build_features(df)
    y = (df["har_rv_predicted"] < df["har_rv_predicted"].quantile(0.30)).astype(int)
    splits = temporal_split(X, y)

    result = train_combinator(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        n_trials=N_TRIALS_MAX + 5,  # demande au-dessus du cap
        timeout_seconds=2,  # short timeout : ne va pas faire 50 trials
        use_optuna=True,
    )
    # Quel que soit le n_trials atteint (limited by timeout), ne dépasse jamais cap
    assert result.n_trials_used <= N_TRIALS_MAX


def test_train_combinator_fallback_no_optuna():
    """Si optuna absent → fallback default LGBM, n_trials_used=1, val_auc défini."""
    df = _synth_df(800, seed=11)
    X = build_features(df)
    # Cible corrélée à har_rv pour avoir un edge réel (sinon AUC ~0.5)
    y = (df["har_rv_predicted"] < df["har_rv_predicted"].quantile(0.30)).astype(int)
    splits = temporal_split(X, y)

    with patch("src.lgbm_combinator.HAS_OPTUNA", False):
        result = train_combinator(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
            use_optuna=False,
        )
    assert result.converged is True
    assert result.n_trials_used == 1
    assert 0.0 <= result.val_auc <= 1.0
    assert result.model is not None


def test_predict_low_vol_proba_returns_floats_in_unit_interval():
    """Avec un modèle entraîné, les probas sont dans [0,1]."""
    df = _synth_df(500, seed=5)
    X = build_features(df)
    y = (df["har_rv_predicted"] < df["har_rv_predicted"].quantile(0.30)).astype(int)
    splits = temporal_split(X, y)
    res = train_combinator(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        use_optuna=False,
    )
    proba = predict_low_vol_proba(res.model, splits["X_test"])
    assert proba.shape == (len(splits["X_test"]),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_low_vol_proba_no_model_returns_neutral():
    """Modèle None → 0.5 partout (pas de signal)."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    proba = predict_low_vol_proba(None, X)
    assert (proba == 0.5).all()


# ---------------------------------------------------------------------------
# evaluate_with_dsr (mock pour confirmer n_trials=50)
# ---------------------------------------------------------------------------


def test_dsr_deflated_called_with_n_trials_50():
    """evaluate_with_dsr doit appeler deflated_sharpe_ratio avec n_trials<=50."""
    with patch("src.lgbm_combinator.deflated_sharpe_ratio") as mock_dsr:
        mock_dsr.return_value = 0.97
        evaluate_with_dsr(val_auc=0.60, n_trials_used=50, n_obs=300)
        call_kwargs = mock_dsr.call_args.kwargs
        assert call_kwargs.get("n_trials") == 50


def test_dsr_clamped_when_n_trials_exceeds_max():
    """Si n_trials_used > N_TRIALS_MAX (50), on clamp à 50."""
    with patch("src.lgbm_combinator.deflated_sharpe_ratio") as mock_dsr:
        mock_dsr.return_value = 0.97
        evaluate_with_dsr(val_auc=0.60, n_trials_used=200, n_obs=300)
        call_kwargs = mock_dsr.call_args.kwargs
        assert call_kwargs.get("n_trials") == 50


# ---------------------------------------------------------------------------
# should_deploy gate
# ---------------------------------------------------------------------------


def test_low_auc_returns_no_deploy_flag():
    """AUC < 0.55 → no deploy."""
    deploy, reason = should_deploy(val_auc=0.52, dsr_p=0.99)
    assert deploy is False
    assert "auc" in reason.lower() and ("0.55" in reason or "edge" in reason.lower())


def test_high_auc_returns_no_deploy_flag_suspect_leakage():
    """AUC > 0.65 → suspect leakage → no deploy."""
    deploy, reason = should_deploy(val_auc=0.70, dsr_p=0.99)
    assert deploy is False
    assert "suspect" in reason.lower() or "leakage" in reason.lower() or "0.65" in reason


def test_low_dsr_returns_no_deploy_flag():
    """AUC OK mais DSR < 0.95 → no deploy (snooping)."""
    deploy, reason = should_deploy(val_auc=0.60, dsr_p=0.80)
    assert deploy is False
    assert "dsr" in reason.lower() or "snoop" in reason.lower()


def test_acceptable_metrics_deploy():
    """AUC ∈ [0.55, 0.65] et DSR > 0.95 → deploy=True."""
    deploy, reason = should_deploy(val_auc=0.60, dsr_p=0.96)
    assert deploy is True
    assert reason == ""


def test_constants_match_spec():
    assert AUC_DEPLOY_MIN == 0.55
    assert AUC_DEPLOY_SUSPECT_MAX == 0.65
    assert DSR_DEPLOY_MIN == 0.95


# ---------------------------------------------------------------------------
# Branchement signal_engine
# ---------------------------------------------------------------------------


def test_signal_engine_combinator_blocks_only_when_both_agree():
    """SignalEngine._low_vol_combinator_blocks() : True ssi HAR=low ET proba > seuil."""
    sys.path.insert(0, str(ROOT / "binance_bot"))
    from services.signal_engine import SignalEngine

    # HAR low + proba haute → bloque
    eng_block = SignalEngine(
        regime_gate_fn=lambda: "low",
        low_vol_combinator_fn=lambda: 0.85,
        low_vol_combinator_threshold=0.6,
    )
    assert eng_block._low_vol_combinator_blocks() is True

    # HAR mid + proba haute → ne bloque pas (HAR pas low)
    eng_har_ok = SignalEngine(
        regime_gate_fn=lambda: "mid",
        low_vol_combinator_fn=lambda: 0.85,
    )
    assert eng_har_ok._low_vol_combinator_blocks() is False

    # HAR low + proba basse → ne bloque pas (combinator pas confiant)
    eng_proba_low = SignalEngine(
        regime_gate_fn=lambda: "low",
        low_vol_combinator_fn=lambda: 0.30,
    )
    assert eng_proba_low._low_vol_combinator_blocks() is False

    # Sans combinator_fn → False
    eng_no_fn = SignalEngine(regime_gate_fn=lambda: "low")
    assert eng_no_fn._low_vol_combinator_blocks() is False

    # Combinator raise → False (safe fallback)
    def _bad():
        raise RuntimeError("LGBM model failed")
    eng_raise = SignalEngine(
        regime_gate_fn=lambda: "low",
        low_vol_combinator_fn=_bad,
    )
    assert eng_raise._low_vol_combinator_blocks() is False
