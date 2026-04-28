"""P4 — HAR-RV régime de volatilité (Corsi 2009).

Heterogeneous Autoregressive of Realized Volatility — modèle simple et robuste
pour prévoir la volatilité réalisée court terme via une cascade de fenêtres
1h / 5h / 22h. Pas d'hyperparamètres réglables → pas de risque de data snooping.

Référence:
    Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
    Volatility." Journal of Financial Econometrics, 7(2), 174-196.

Validation crypto:
    Bergsli, L. Ø., Lind, A. F., Molnár, P., & Polasik, M. (2022).
    "Forecasting volatility of Bitcoin." Research in International Business
    and Finance.

Modèle (Eq. 1 du papier original, version log) :
    log(RV_{t+1}) = β0 + β_h × log(RV_1h(t)) + β_d × log(RV_5h(t))
                                            + β_w × log(RV_22h(t)) + ε_{t+1}

Implémentation :
    - Realized Volatility = sqrt(sum(r_i^2)) sur fenêtre rolling.
    - OLS via np.linalg.lstsq (pas de dep statsmodels).
    - Régime classifié via quantiles 30%/70% sur RV_1h des 90 derniers jours.
"""
from __future__ import annotations

from typing import Literal, Mapping, Optional

import numpy as np
import pandas as pd


# Fenêtres canoniques Corsi 2009 (en bars 1h pour application crypto H1).
_WINDOW_HOURLY = 1
_WINDOW_DAILY = 5    # ~ 1 trading day pour stocks; ici 5h
_WINDOW_WEEKLY = 22  # ~ 1 trading week stocks; 22h crypto

_QUANTILE_LOW = 0.30
_QUANTILE_HIGH = 0.70
_REGIME_WINDOW_BARS = 90 * 24  # 90 jours × 24h

RegimeLabel = Literal["low", "mid", "high"]


def realized_volatility(
    returns: pd.Series,
    window: int = _WINDOW_WEEKLY,
) -> pd.Series:
    """RV rolling = sqrt(somme cumulative des r² sur `window` bars).

    Args:
        returns: série de returns par bar (1h typiquement).
        window: nombre de bars à agréger.

    Returns:
        pd.Series alignée sur returns.index, NaN aux premiers bars (< window).
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    sq = returns.astype(float).pow(2)
    rv = sq.rolling(window=window, min_periods=window).sum().pow(0.5)
    return rv


def _build_har_features(
    returns_1h: pd.Series,
    w_h: int = _WINDOW_HOURLY,
    w_d: int = _WINDOW_DAILY,
    w_w: int = _WINDOW_WEEKLY,
) -> pd.DataFrame:
    """Construit les features HAR (RV_1h, RV_5h, RV_22h) + target log(RV_{t+1}).

    Args:
        returns_1h: série de returns 1h.
        w_h, w_d, w_w: fenêtres horaire / daily / weekly.

    Returns:
        DataFrame avec colonnes ["log_rv_h", "log_rv_d", "log_rv_w", "log_rv_next"]
        nettoyé des NaN.
    """
    rv_h = realized_volatility(returns_1h, window=w_h).replace(0, np.nan)
    rv_d = realized_volatility(returns_1h, window=w_d).replace(0, np.nan)
    rv_w = realized_volatility(returns_1h, window=w_w).replace(0, np.nan)
    df = pd.DataFrame({
        "log_rv_h": np.log(rv_h),
        "log_rv_d": np.log(rv_d),
        "log_rv_w": np.log(rv_w),
        "log_rv_next": np.log(rv_h.shift(-1)),  # cible: RV_h à t+1
    })
    return df.dropna()


def fit_har_rv(returns_1h: pd.Series) -> Mapping[str, float]:
    """Fit OLS du modèle HAR-RV sur historique returns 1h.

    Args:
        returns_1h: série temporelle returns 1h sur l'historique de train.

    Returns:
        dict avec clés "beta_0", "beta_h", "beta_d", "beta_w" + "n_obs".
        Tous les coefficients sont des floats finis. Si train trop court,
        retourne dict avec coefs = NaN et n_obs=0.
    """
    df = _build_har_features(returns_1h)
    n = len(df)
    if n < 30:
        return {"beta_0": np.nan, "beta_h": np.nan, "beta_d": np.nan,
                "beta_w": np.nan, "n_obs": 0}
    X = np.column_stack([
        np.ones(n),
        df["log_rv_h"].to_numpy(),
        df["log_rv_d"].to_numpy(),
        df["log_rv_w"].to_numpy(),
    ])
    y = df["log_rv_next"].to_numpy()
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "beta_0": float(coefs[0]),
        "beta_h": float(coefs[1]),
        "beta_d": float(coefs[2]),
        "beta_w": float(coefs[3]),
        "n_obs": int(n),
    }


def predict_rv(
    model_params: Mapping[str, float],
    current_rv_1h: float,
    current_rv_5h: float,
    current_rv_22h: float,
) -> float:
    """Prédit RV_h(t+1) via le modèle HAR-RV fitté.

    Args:
        model_params: dict retourné par fit_har_rv (clés beta_0/h/d/w).
        current_rv_1h: RV horaire actuelle (linéaire, pas log).
        current_rv_5h: RV 5h actuelle.
        current_rv_22h: RV 22h actuelle.

    Returns:
        RV prédite (linéaire, strictement positive). 0.0 si inputs invalides.
    """
    if any(v <= 0 or not np.isfinite(v) for v in
           (current_rv_1h, current_rv_5h, current_rv_22h)):
        return 0.0
    b0 = float(model_params.get("beta_0", np.nan))
    bh = float(model_params.get("beta_h", np.nan))
    bd = float(model_params.get("beta_d", np.nan))
    bw = float(model_params.get("beta_w", np.nan))
    if any(not np.isfinite(b) for b in (b0, bh, bd, bw)):
        return 0.0
    log_pred = (
        b0
        + bh * np.log(current_rv_1h)
        + bd * np.log(current_rv_5h)
        + bw * np.log(current_rv_22h)
    )
    return float(np.exp(log_pred))


def classify_regime(
    rv_predicted: float,
    historical_rv_1h: pd.Series,
    quantile_low: float = _QUANTILE_LOW,
    quantile_high: float = _QUANTILE_HIGH,
) -> RegimeLabel:
    """Classifie la RV prédite via les quantiles 30%/70% des 90j rolling.

    Args:
        rv_predicted: RV prédite par predict_rv (linéaire, > 0).
        historical_rv_1h: pd.Series des RV_1h des 90 derniers jours
            (= 90 × 24 = 2160 bars typiquement).
        quantile_low/high: seuils de classification.

    Returns:
        "low"  si rv_predicted < quantile_low(historical)
        "high" si rv_predicted > quantile_high(historical)
        "mid"  sinon (ou si historical insuffisant pour estimer les quantiles)
    """
    if not isinstance(historical_rv_1h, pd.Series) or historical_rv_1h.empty:
        return "mid"
    series = historical_rv_1h.dropna()
    if len(series) < 30:
        return "mid"
    q_low = float(series.quantile(quantile_low))
    q_high = float(series.quantile(quantile_high))
    if not np.isfinite(rv_predicted) or rv_predicted <= 0:
        return "mid"
    if rv_predicted < q_low:
        return "low"
    if rv_predicted > q_high:
        return "high"
    return "mid"


__all__ = [
    "realized_volatility",
    "fit_har_rv",
    "predict_rv",
    "classify_regime",
]
