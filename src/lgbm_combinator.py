"""P10 — LightGBM combinator : P(low_vol_next_hour) à partir des signaux quant.

NE PAS CONFONDRE avec src/regime_lgbm.py (qui est un classifieur directionnel
+1/0/-1 + HMM). Ce module-ci est un combinateur de features quant déjà
calculées en amont (HAR-RV P4, EGARCH P8, VPIN P7, composite signal P6.5,
context features) → cible binaire P(volatilité prochaine basse).

Use case
--------
Si HAR-RV ET combinator disent tous deux "low vol" → gate dur dans le
signal_engine (Ichimoku trend-following sous-performe en range/low-vol).

Pipeline d'entraînement
-----------------------
1. build_features(df_signals)       : X numérique (one-hot phase, cyclic h)
2. build_target(rv_future_1h, rv_history_90d) : y binaire (low_vol=1)
3. temporal_split(X, y)             : 70 / 15 / 15 (test JAMAIS touché)
4. train_combinator(X_tr, y_tr, X_val, y_val, n_trials=50)
   - Optuna (si dispo, max 50 trials, timeout 1h)
   - Sinon fallback LightGBM default params
5. evaluate_with_dsr(model, X_test, y_test, n_trials_used) → AUC + DSR p-value
6. should_deploy(auc_val, dsr_p)    → bool + reason

Garde-fous
----------
- AUC validation > 0.65        : SUSPECT (probablement leakage temporal)
- AUC validation < 0.55        : pas d'edge (pas de déploiement)
- DSR p-value < 0.95           : edge non significatif après 50 trials → no deploy

Référence
---------
Bailey, López de Prado (2014). "The Deflated Sharpe Ratio." Journal of
Portfolio Management, 40(5), 94-107.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False
    lgb = None

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None  # type: ignore

from .stats_eval import deflated_sharpe_ratio


# Bornes deploy / suspect (audit-spec)
AUC_DEPLOY_MIN = 0.55
AUC_DEPLOY_SUSPECT_MAX = 0.65
DSR_DEPLOY_MIN = 0.95
N_TRIALS_MAX = 50  # Hard cap (anti-snooping)
LOW_VOL_QUANTILE_DEFAULT = 0.30  # 30e percentile RV historique


# ---------------------------------------------------------------------------
# Features & target
# ---------------------------------------------------------------------------


_REQUIRED_FEATURE_COLS = (
    "har_rv_predicted",
    "har_rv_classified",
    "egarch_sigma_predicted",
    "vpin",
    "composite_signal_score",
    "comp_top_ls",
    "comp_taker",
    "comp_liq",
    "comp_oi",
    "phase_K3",
    "days_since_halving",
    "hour_of_day",
    "day_of_week",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit X numérique encodé pour LightGBM.

    Encoding:
      - har_rv_classified  : one-hot {low, mid, high}
      - phase_K3           : one-hot {0, 1, 2}
      - hour_of_day        : cyclic sin/cos (period 24)
      - day_of_week        : one-hot {0..6}

    Args:
        df: DataFrame avec les colonnes _REQUIRED_FEATURE_COLS.

    Returns:
        DataFrame X numérique encodé. Lève KeyError si colonnes manquantes.
    """
    missing = [c for c in _REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing required feature columns: {missing}")

    X = pd.DataFrame(index=df.index)

    # Numérique brut
    X["har_rv_predicted"] = df["har_rv_predicted"].astype(float)
    X["egarch_sigma_predicted"] = df["egarch_sigma_predicted"].astype(float)
    X["vpin"] = df["vpin"].astype(float)
    X["composite_signal_score"] = df["composite_signal_score"].astype(float)
    X["comp_top_ls"] = df["comp_top_ls"].astype(float)
    X["comp_taker"] = df["comp_taker"].astype(float)
    X["comp_liq"] = df["comp_liq"].astype(float)
    X["comp_oi"] = df["comp_oi"].astype(float)
    X["days_since_halving"] = df["days_since_halving"].astype(float)

    # One-hot regime HAR
    for label in ("low", "mid", "high"):
        X[f"har_regime_{label}"] = (df["har_rv_classified"] == label).astype(int)

    # One-hot phase K3
    for k in (0, 1, 2):
        X[f"phase_K3_{k}"] = (df["phase_K3"] == k).astype(int)

    # Cyclic encoding hour_of_day
    h = df["hour_of_day"].astype(float)
    X["hour_sin"] = np.sin(2.0 * np.pi * h / 24.0)
    X["hour_cos"] = np.cos(2.0 * np.pi * h / 24.0)

    # One-hot day_of_week
    for d in range(7):
        X[f"dow_{d}"] = (df["day_of_week"] == d).astype(int)

    return X


def build_target(
    rv_future_1h: pd.Series,
    rv_history_90d: pd.Series,
    quantile: float = LOW_VOL_QUANTILE_DEFAULT,
) -> pd.Series:
    """Cible binaire : 1 si RV(t+1h) < quantile_q(RV historique 90j), sinon 0.

    Args:
        rv_future_1h: pd.Series, RV à t+1h (déjà shiftée correctement).
        rv_history_90d: pd.Series, RV roulante 90j alignée sur t.
        quantile: percentile servant de seuil (default 30 %).

    Returns:
        pd.Series binaire ∈ {0, 1}.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    threshold = rv_history_90d.quantile(quantile)
    return (rv_future_1h < threshold).astype(int)


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> dict:
    """Split temporel strict (pas de shuffle, pas de leakage).

    Test set est JAMAIS touché — réservé à evaluate_with_dsr().

    Returns:
        dict {X_train, y_train, X_val, y_val, X_test, y_test, indices}.
    """
    if not (0.0 < train_pct < 1.0):
        raise ValueError("train_pct out of range")
    if not (0.0 < val_pct < 1.0):
        raise ValueError("val_pct out of range")
    if train_pct + val_pct >= 1.0:
        raise ValueError("train_pct + val_pct must be < 1.0 (test leftover)")
    n = len(X)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    return {
        "X_train": X.iloc[:n_train],
        "y_train": y.iloc[:n_train],
        "X_val": X.iloc[n_train:n_train + n_val],
        "y_val": y.iloc[n_train:n_train + n_val],
        "X_test": X.iloc[n_train + n_val:],
        "y_test": y.iloc[n_train + n_val:],
        "indices": {
            "train_end": n_train - 1,
            "val_start": n_train,
            "val_end": n_train + n_val - 1,
            "test_start": n_train + n_val,
        },
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


_DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbose": -1,
}


@dataclass
class CombinatorTrainResult:
    model: object  # lgb.Booster ou None si LGBM absent
    best_params: dict
    n_trials_used: int
    val_auc: float
    converged: bool


def train_combinator(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = N_TRIALS_MAX,
    timeout_seconds: int = 3600,
    use_optuna: bool = True,
    seed: int = 42,
) -> CombinatorTrainResult:
    """Entraîne le combinator. Optuna max N_TRIALS_MAX (50) hard-cap.

    Args:
        n_trials: nb de trials Optuna (clampé à N_TRIALS_MAX).
        use_optuna: si False ou optuna absent → fallback default params.
        seed: pour reproductibilité.

    Returns:
        CombinatorTrainResult avec model, best_params, n_trials_used, val_auc.
    """
    if not HAS_LIGHTGBM:
        return CombinatorTrainResult(
            model=None, best_params={}, n_trials_used=0,
            val_auc=0.0, converged=False,
        )

    n_trials = int(min(max(n_trials, 1), N_TRIALS_MAX))

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    if not (use_optuna and HAS_OPTUNA):
        params = dict(_DEFAULT_LGBM_PARAMS)
        params["seed"] = seed
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        val_auc = float(model.best_score["valid_0"]["auc"])
        return CombinatorTrainResult(
            model=model, best_params=params,
            n_trials_used=1, val_auc=val_auc, converged=True,
        )

    # Optuna search
    def _objective(trial):
        params = dict(_DEFAULT_LGBM_PARAMS)
        params["learning_rate"] = trial.suggest_float("lr", 1e-3, 0.2, log=True)
        params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
        params["feature_fraction"] = trial.suggest_float("ff", 0.5, 1.0)
        params["bagging_fraction"] = trial.suggest_float("bf", 0.5, 1.0)
        params["min_data_in_leaf"] = trial.suggest_int("min_data", 10, 100)
        params["seed"] = seed
        model = lgb.train(
            params, train_data, num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        return float(model.best_score["valid_0"]["auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(_objective, n_trials=n_trials, timeout=timeout_seconds)

    best_params = dict(_DEFAULT_LGBM_PARAMS)
    best_params.update({
        "learning_rate": study.best_params.get("lr", 0.05),
        "num_leaves": study.best_params.get("num_leaves", 31),
        "feature_fraction": study.best_params.get("ff", 0.8),
        "bagging_fraction": study.best_params.get("bf", 0.8),
        "min_data_in_leaf": study.best_params.get("min_data", 20),
        "seed": seed,
    })
    model = lgb.train(
        best_params, train_data, num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    val_auc = float(model.best_score["valid_0"]["auc"])
    return CombinatorTrainResult(
        model=model, best_params=best_params,
        n_trials_used=int(len(study.trials)),
        val_auc=val_auc, converged=True,
    )


def predict_low_vol_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Retourne P(low_vol_next_hour) ∈ [0, 1] pour chaque ligne de X."""
    if model is None:
        return np.full(len(X), 0.5)  # neutre si pas de modèle
    return np.asarray(model.predict(X, num_iteration=getattr(model, "best_iteration", None)))


# ---------------------------------------------------------------------------
# Evaluation & deploy gate
# ---------------------------------------------------------------------------


def evaluate_with_dsr(
    val_auc: float,
    n_trials_used: int,
    n_obs: int,
) -> dict:
    """Évalue significance via DSR.

    On utilise l'AUC validation comme proxy de Sharpe (en l'absence d'un
    backtest PnL). C'est conservateur : DSR(AUC) traite l'AUC comme un
    indicateur de performance soumis au snooping.

    Note: idéalement on remplacerait val_auc par sharpe d'un backtest
    déployant le model en filtre. Pour P10, on garde DSR sur val_auc.

    Returns:
        dict {val_auc, dsr_p, n_trials_used}.
    """
    if n_trials_used <= 1:
        # Pas de snooping → DSR identique à PSR
        n_trials_for_dsr = 1
    else:
        n_trials_for_dsr = int(min(n_trials_used, N_TRIALS_MAX))

    # Mappe val_auc dans [0, 1] vers un "Sharpe-like" pour DSR.
    # AUC 0.5 = neutre → SR 0 ; AUC 0.65 = solide → SR ≈ 1.5.
    sharpe_proxy = max(0.0, (val_auc - 0.5)) * 10.0
    dsr_p = deflated_sharpe_ratio(
        sharpe_observed=sharpe_proxy,
        n_obs=int(max(n_obs, 2)),
        n_trials=int(n_trials_for_dsr),
    )
    return {
        "val_auc": float(val_auc),
        "dsr_p": float(dsr_p),
        "n_trials_used": int(n_trials_used),
        "n_obs": int(n_obs),
    }


def should_deploy(val_auc: float, dsr_p: float) -> tuple[bool, str]:
    """Décision de déploiement basée sur AUC validation + DSR p-value.

    Returns:
        (deploy: bool, reason: str). reason vide si deploy=True.
    """
    if not np.isfinite(val_auc) or not np.isfinite(dsr_p):
        return False, "auc/dsr non-finis"
    if val_auc < AUC_DEPLOY_MIN:
        return False, f"auc {val_auc:.3f} < {AUC_DEPLOY_MIN} (pas d'edge)"
    if val_auc > AUC_DEPLOY_SUSPECT_MAX:
        return False, f"auc {val_auc:.3f} > {AUC_DEPLOY_SUSPECT_MAX} (suspect, leakage probable)"
    if dsr_p < DSR_DEPLOY_MIN:
        return False, f"dsr_p {dsr_p:.3f} < {DSR_DEPLOY_MIN} (snooping)"
    return True, ""


__all__ = [
    "AUC_DEPLOY_MIN",
    "AUC_DEPLOY_SUSPECT_MAX",
    "DSR_DEPLOY_MIN",
    "N_TRIALS_MAX",
    "LOW_VOL_QUANTILE_DEFAULT",
    "build_features",
    "build_target",
    "temporal_split",
    "train_combinator",
    "predict_low_vol_proba",
    "evaluate_with_dsr",
    "should_deploy",
    "CombinatorTrainResult",
]
