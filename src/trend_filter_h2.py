"""P-MTF-2 — Filtre directionnel basé sur la tendance H2 Ichimoku.

Module pur, stateless. Entrée : DataFrame Ichimoku H2 (sortie de
`calculate_ichimoku`). Sortie : direction ∈ {long, short, flat}.

Convention :
    close > cloud_top    → tendance haussière → "long" (longs autorisés)
    close < cloud_bottom → tendance baissière → "short" (shorts autorisés)
    sinon                 → indécision        → "flat" (pas de trade)

Ce filtre est consommé par `execution_runner_15m` via `trend_gate_fn`
injecté dans `SignalEngine` (cf P-MTF-3 et P-MTF-5).

Référence : Ichimoku Kinkō Hyō, Goichi Hosoda 1969 — le nuage (Kumo)
représente une zone de support/résistance dynamique. Trading dans le
nuage = tendance indéterminée, recommandé de s'abstenir.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def classify_h2_trend(
    df_ichimoku_h2: pd.DataFrame,
    margin_pct: float = 0.0,
) -> dict:
    """Classify la tendance H2 depuis un DF Ichimoku.

    Args:
        df_ichimoku_h2: DataFrame issu de `calculate_ichimoku()`. Doit
            contenir au minimum les colonnes `close`, `cloud_top`,
            `cloud_bottom`. La dernière ligne est utilisée pour la
            décision (présent vs nuage projeté).
        margin_pct: marge proportionnelle autour du nuage. 0 = strict
            (close > cloud_top → long). 0.005 = 0.5% de marge (close
            doit être > cloud_top × 1.005 pour être considéré long).
            Permet d'éviter les faux signaux dans les zones frontalières.

    Returns:
        dict {
            "direction": "long" | "short" | "flat",
            "last_close": float | None,
            "cloud_top": float | None,
            "cloud_bottom": float | None,
            "distance_pct": float | None,  # (close - cloud_mid) / cloud_mid
            "computed_at_iso": str (UTC ISO 8601),
            "reason": str (court : "above_cloud", "below_cloud",
                             "inside_cloud", "no_data", "nan_values"),
        }

    Garantie : ne lève jamais d'exception. En cas de DF vide, NaN ou
    colonnes manquantes, retourne `direction="flat"` avec `reason`
    explicite. C'est un fail-safe : "flat" = pas de trade autorisé.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    base: dict = {
        "direction": "flat",
        "last_close": None,
        "cloud_top": None,
        "cloud_bottom": None,
        "distance_pct": None,
        "computed_at_iso": now_iso,
        "reason": "no_data",
    }

    if df_ichimoku_h2 is None or df_ichimoku_h2.empty:
        return base

    required = {"close", "cloud_top", "cloud_bottom"}
    if not required.issubset(df_ichimoku_h2.columns):
        base["reason"] = "missing_columns"
        return base

    last = df_ichimoku_h2.iloc[-1]
    close = last.get("close")
    cloud_top = last.get("cloud_top")
    cloud_bottom = last.get("cloud_bottom")

    # NaN check : au démarrage de Ichimoku, les premières N bougies ont
    # des NaN sur cloud_top/bottom à cause du shift. Si la DERNIÈRE
    # bougie a un NaN, c'est qu'il manque trop d'historique → flat.
    if (
        close is None
        or cloud_top is None
        or cloud_bottom is None
        or pd.isna(close)
        or pd.isna(cloud_top)
        or pd.isna(cloud_bottom)
    ):
        base["reason"] = "nan_values"
        return base

    close = float(close)
    cloud_top = float(cloud_top)
    cloud_bottom = float(cloud_bottom)
    cloud_mid = (cloud_top + cloud_bottom) / 2.0

    base["last_close"] = close
    base["cloud_top"] = cloud_top
    base["cloud_bottom"] = cloud_bottom
    base["distance_pct"] = (close - cloud_mid) / cloud_mid if cloud_mid > 0 else 0.0

    # Application de la marge proportionnelle
    upper_threshold = cloud_top * (1.0 + margin_pct)
    lower_threshold = cloud_bottom * (1.0 - margin_pct)

    if close > upper_threshold:
        base["direction"] = "long"
        base["reason"] = "above_cloud"
    elif close < lower_threshold:
        base["direction"] = "short"
        base["reason"] = "below_cloud"
    else:
        base["direction"] = "flat"
        base["reason"] = "inside_cloud"

    return base


def is_h2_trend_stale(
    h2_trend: Optional[dict],
    max_age_hours: float = 3.0,
    now: Optional[datetime] = None,
) -> bool:
    """Renvoie True si le snapshot H2 trend est trop vieux pour être fiable.

    Utilisé par execution_runner_15m pour décider de skipper le cycle si
    h2_trend_runner n'a pas tourné depuis trop longtemps (ex: timer raté
    après un reboot, échec réseau Binance...).

    Args:
        h2_trend: dict retourné par classify_h2_trend (ou None).
        max_age_hours: seuil au-delà duquel on considère le snapshot obsolète.
        now: datetime (UTC) pour le test (default = datetime.now(UTC)).

    Returns:
        True si stale (ou si h2_trend invalide), False sinon.
    """
    if not h2_trend or "computed_at_iso" not in h2_trend:
        return True
    try:
        ts = datetime.fromisoformat(h2_trend["computed_at_iso"])
    except (ValueError, TypeError):
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_hours = (now - ts).total_seconds() / 3600.0
    return age_hours > max_age_hours


__all__ = ["classify_h2_trend", "is_h2_trend_stale"]
