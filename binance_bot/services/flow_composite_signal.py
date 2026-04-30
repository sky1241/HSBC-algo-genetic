"""P6.5 — Composite signal Binance Flow Stack.

Agrège les 4 flux Binance (P6.1-P6.4) en un score continu [-1, +1] qui
peut bloquer des trades adverses en mode GATE après collecte d'une baseline
30j.

Composition (pondération validée par la spec):
    score = 0.30 × z(top_ls_inverted)   # contrarian (P6.1)
          + 0.25 × z(taker_centered)    # agressivité (P6.2)
          + 0.25 × z(liq_imbalance)     # capitulation (P6.3)
          + 0.20 × z(oi_change_pct)     # engagement (P6.4)

    z() : z-score sur fenêtre rolling 30j (gestion drift).
    Output clippé à [-1, +1] (chaque composant peut tirer fortement).

Conventions de signe (TOUS contrarian → score positif = signal LONG bullish):
    top_ls_inverted = 1 - 2 × longAccount
        → +1 si 80% short (contrarian → LONG attractif)
        → -1 si 80% long  (contrarian → SHORT attractif)

    taker_centered = buy_sell_ratio - 0.5
        → positif = + d'aggressive buy → bullish
        → négatif = + d'aggressive sell → bearish
        Z-scoré pour normaliser : composant bullish/bearish purement empirique.

    liq_imbalance = (long_liq_notional - short_liq_notional) / total_liq
        → positif = + de longs liquidés → capitulation longs → bullish contrarian
        → négatif = + de shorts liquidés → capitulation shorts → bearish contrarian

    oi_change_pct_24h = (OI_now - OI_24h_ago) / OI_24h_ago
        → z-scoré pour normaliser. Engagement croissant amplifie tendance.

Mode INITIAL (log only) : 30 jours de baseline collectés, le gate ne bloque
PAS encore les trades. Le score est juste loggé pour analyse.
Mode GATE : après 30j, le gate bloque les trades adverses si |score| > 0.4.

Pour P6.5 minimum-viable, le branchement signal_engine est OPTIONNEL
(pattern callback `composite_gate_fn` similaire à P3/P4). On expose les
fonctions pures + un script standalone qui calcule le score.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional  # noqa: F401

import numpy as np
import pandas as pd


# Pondérations validées par la spec
_WEIGHT_TOP_LS = 0.30
_WEIGHT_TAKER = 0.25
_WEIGHT_LIQ = 0.25
_WEIGHT_OI = 0.20

_DEFAULT_GATE_THRESHOLD = 0.4
_DEFAULT_ZSCORE_LOOKBACK_DAYS = 30
_DEFAULT_OI_LOOKBACK_24H_BARS = 288  # 24h × 60min / 5min = 288 bars

ModeType = Literal["log", "gate"]


# ---------------------------------------------------------------------------
# Loaders jsonl → DataFrame
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, days_back: int = _DEFAULT_ZSCORE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Charge un jsonl (P6.1-P6.4) et filtre les records des `days_back` derniers jours."""
    if not path.exists():
        return pd.DataFrame()
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts = int(rec.get("ts_ms") or rec.get("bucket_start_ms") or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if ts < cutoff_ms:
                continue
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normaliser ts_ms (priorité ts_ms ; sinon bucket_start_ms)
    if "ts_ms" not in df.columns and "bucket_start_ms" in df.columns:
        df["ts_ms"] = df["bucket_start_ms"]
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Composants (chacun retourne pd.Series indexée par ts_ms)
# ---------------------------------------------------------------------------


def compute_top_ls_inverted_series(df: pd.DataFrame) -> pd.Series:
    """Convertit longAccount → score inversé contrarian [-1, +1]."""
    if df.empty or "long_account" not in df.columns:
        return pd.Series(dtype=float)
    s = 1.0 - 2.0 * df["long_account"].astype(float)
    s.index = df["ts_ms"].astype(int)
    return s.dropna()


def compute_taker_centered_series(df: pd.DataFrame) -> pd.Series:
    """Centre buy_sell_ratio autour de 1.0 (pas 0.5 comme la doc le laisserait croire).

    Note importante: dans P6.2, `buy_sell_ratio` = buyVol / sellVol (ratio,
    pas fraction). 1.0 = équilibre. Donc on retourne (ratio - 1) pour centrer.
    """
    if df.empty or "buy_sell_ratio" not in df.columns:
        return pd.Series(dtype=float)
    s = df["buy_sell_ratio"].astype(float) - 1.0
    s.index = df["ts_ms"].astype(int)
    return s.dropna()


def compute_liq_imbalance_series(df: pd.DataFrame) -> pd.Series:
    """liq_imbalance = (long_liq_notional - short_liq_notional) / total_liq ∈ [-1, +1]."""
    if df.empty or "long_liq_notional" not in df.columns or "short_liq_notional" not in df.columns:
        return pd.Series(dtype=float)
    total = df["long_liq_notional"].astype(float) + df["short_liq_notional"].astype(float)
    diff = df["long_liq_notional"].astype(float) - df["short_liq_notional"].astype(float)
    # Évite division par 0 sur buckets sans aucune liq
    s = pd.Series(np.where(total > 0, diff / total.replace(0, np.nan), 0.0))
    ts_col = "bucket_start_ms" if "bucket_start_ms" in df.columns else "ts_ms"
    s.index = df[ts_col].astype(int)
    return s.dropna()


def compute_oi_change_pct_series(
    df: pd.DataFrame,
    lookback_bars: int = _DEFAULT_OI_LOOKBACK_24H_BARS,
) -> pd.Series:
    """OI change pct vs lookback bars ago (default 24h = 288 bars × 5min).

    Returns: pd.Series de (OI_t - OI_{t-N}) / OI_{t-N}.
    """
    if df.empty or "sum_oi" not in df.columns:
        return pd.Series(dtype=float)
    oi = df["sum_oi"].astype(float)
    if len(oi) <= lookback_bars:
        return pd.Series(dtype=float)
    prev = oi.shift(lookback_bars)
    s = (oi - prev) / prev.replace(0, np.nan)
    s.index = df["ts_ms"].astype(int)
    return s.dropna()


# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------


def zscore_last(series: pd.Series, min_obs: int = 100) -> float:
    """Z-score de la dernière valeur de `series` vs distribution complète.

    Args:
        series: pd.Series de valeurs (déjà filtrée sur fenêtre rolling).
        min_obs: nombre minimum d'observations pour calculer un z-score
            fiable. En dessous, retourne 0.0 (pas de signal).

    Returns:
        Z-score de la dernière valeur. 0.0 si insuffisamment de données ou
        std nulle.
    """
    s = series.dropna()
    if len(s) < min_obs:
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return 0.0
    z = (float(s.iloc[-1]) - mean) / std
    return float(z) if np.isfinite(z) else 0.0


def clip_score(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clipping [-1, +1] (ou [lo, hi] custom).

    Convention numpy: NaN → 0.0 (indéterminé, neutre). inf → hi, -inf → lo.
    """
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    if value == float("inf"):
        return float(hi)
    if value == float("-inf"):
        return float(lo)
    return float(max(lo, min(hi, value)))


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_composite_score(
    top_ls_path: Path,
    taker_path: Path,
    liq_path: Path,
    oi_path: Path,
    days_back: int = _DEFAULT_ZSCORE_LOOKBACK_DAYS,
    min_obs_for_active: int = 100,
) -> dict:
    """Calcule le score composite à partir des 4 jsonl jeux de flux.

    Renormalisation dynamique des poids (COMPOSITE-001) :
    si une feature n'a pas assez d'observations valides pour produire un
    z-score fiable (n_obs < min_obs_for_active), elle est SKIPPÉE et les
    poids des features actives sont renormalisés à somme = 1.0. C'est la
    bonne pratique factor modeling / ESG composite scoring : préserver
    la comparabilité du score quel que soit le nombre de features dispos.

    Cas d'usage critique : depuis WS-001, le flux liq n'a plus de data
    (Binance ferme @forceOrder côté serveur). Avant ce fix, la composante
    liq valait 0.0 mais conservait son poids 0.25, ce qui réduisait
    mécaniquement le score de 25%. Maintenant, les 3 autres features
    sont renormalisées (top_ls 0.30→0.40, taker 0.25→0.333, oi 0.20→0.267)
    et le score reste comparable à un score 4-features.

    Returns:
        dict {
            "score": float ∈ [-1, +1] (clipped),
            "raw_score": float (pre-clip),
            "components": z-scores des features ACTIVES uniquement,
            "weights": pondérations renormalisées des features ACTIVES,
            "base_weights": pondérations de design (audit trail),
            "active_features": liste des features utilisées dans le score,
            "degraded": True si au moins 1 feature manquante,
            "n_obs": dict des comptes par flux (toutes les 4, pour drift).
        }
    """
    df_top_ls = _load_jsonl(top_ls_path, days_back=days_back)
    df_taker = _load_jsonl(taker_path, days_back=days_back)
    df_liq = _load_jsonl(liq_path, days_back=days_back)
    df_oi = _load_jsonl(oi_path, days_back=days_back)

    series_map = {
        "top_ls": compute_top_ls_inverted_series(df_top_ls),
        "taker": compute_taker_centered_series(df_taker),
        "liq": compute_liq_imbalance_series(df_liq),
        "oi": compute_oi_change_pct_series(df_oi),
    }
    n_obs_map = {k: int(len(s)) for k, s in series_map.items()}

    base_weights = {
        "top_ls": _WEIGHT_TOP_LS,
        "taker": _WEIGHT_TAKER,
        "liq": _WEIGHT_LIQ,
        "oi": _WEIGHT_OI,
    }

    # Une feature est "active" si suffisamment d'obs pour z-score fiable
    active_components = {
        k: zscore_last(s, min_obs=min_obs_for_active)
        for k, s in series_map.items()
        if len(s) >= min_obs_for_active
    }

    if not active_components:
        return {
            "score": 0.0,
            "raw_score": 0.0,
            "components": {},
            "weights": {},
            "base_weights": base_weights,
            "active_features": [],
            "degraded": True,
            "n_obs": n_obs_map,
        }

    # Renormalisation des poids actifs à somme = 1.0
    sum_active_weights = sum(base_weights[k] for k in active_components.keys())
    normalized_weights = {
        k: base_weights[k] / sum_active_weights for k in active_components.keys()
    }

    raw_score = sum(
        normalized_weights[k] * active_components[k] for k in active_components.keys()
    )
    score = clip_score(raw_score)

    return {
        "score": score,
        "raw_score": float(raw_score),
        "components": active_components,
        "weights": normalized_weights,
        "base_weights": base_weights,
        "active_features": list(active_components.keys()),
        "degraded": len(active_components) < len(base_weights),
        "n_obs": n_obs_map,
    }


def should_block_trade(
    score: float,
    side: str,
    mode: ModeType = "log",
    threshold: float = _DEFAULT_GATE_THRESHOLD,
) -> tuple[bool, str]:
    """Décide si un trade doit être bloqué selon le composite score.

    Conventions:
        score > 0  : signal contrarian BULLISH (favorable à LONG, défavorable à SHORT)
        score < 0  : signal contrarian BEARISH (favorable à SHORT, défavorable à LONG)

    Mode "log" : ne bloque JAMAIS (collection baseline 30j).
    Mode "gate" :
        - side="long"  + score < -threshold → BLOCK (sentiment fortement bearish)
        - side="short" + score > +threshold → BLOCK (sentiment fortement bullish)
        - sinon allow

    Args:
        score: composite [-1, +1] depuis compute_composite_score.
        side: "long" ou "short".
        mode: "log" ou "gate".
        threshold: seuil absolu (default 0.4).

    Returns:
        (blocked: bool, reason: str). reason vide si autorisé.
    """
    if mode == "log":
        return False, ""
    if mode != "gate":
        return False, ""
    side_norm = str(side).lower()
    if side_norm == "long" and score < -threshold:
        return True, f"composite_signal score={score:.3f} < -{threshold} (bearish, block long)"
    if side_norm == "short" and score > threshold:
        return True, f"composite_signal score={score:.3f} > +{threshold} (bullish, block short)"
    return False, ""


__all__ = [
    "compute_top_ls_inverted_series",
    "compute_taker_centered_series",
    "compute_liq_imbalance_series",
    "compute_oi_change_pct_series",
    "zscore_last",
    "clip_score",
    "compute_composite_score",
    "should_block_trade",
]
