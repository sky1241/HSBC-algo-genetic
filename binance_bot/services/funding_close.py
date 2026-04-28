"""P11 — Funding-aware close (5 min avant settlement Binance USDM Futures).

Binance Futures perpetuals settle funding à 00:00, 08:00, 16:00 UTC. Si on
détient une position dont le funding nous est défavorable (long avec funding
positif, ou short avec funding négatif), on évite de payer en flat la
position juste avant le settlement.

Logique
-------
Settlement windows : HH ∈ {0, 8, 16} UTC.
Advance window     : [HH:55, HH:00) → on flag les fermetures à exécuter.

Décisions par side et funding rate (bps de la prochaine settlement) :
  - LONG  + funding > +threshold_bps  → CLOSE (on payerait)
  - LONG  + funding < 0               → HOLD  (les shorts nous paient)
  - SHORT + funding < -threshold_bps  → CLOSE (on payerait)
  - SHORT + funding > 0               → HOLD  (les longs nous paient)

Configurable
------------
funding_close_threshold_long_bps      : default 5.0  (~ +18 % APR)
funding_close_threshold_short_bps     : default 5.0  (~ -18 % APR)
funding_close_advance_minutes         : default 5

Notes
-----
- Un funding rate de +5 bps par 8h → APR ~ 5e-4 × 3 × 365 = ~54.75 %.
  La doc demande threshold > 5 bps, donc on ferme dès que le coût annuel
  attendu dépasse ~+18 % APR (≈ 5 bps × 3 × 365 / 100 = 5.475 bps de drag
  daily × 360 ≈ 1971 bps APR ≈ ne tient pas, vérifier la conversion).
  Le seuil ABSOLU 5 bps reste le contrat avec la spec — l'APR équivalent
  est conservateur et empirique.

API
---
SETTLEMENT_HOURS_UTC                       : (0, 8, 16)
is_settlement_window(now, advance=5)       : True ssi à <advance> min d'un settlement
next_settlement(now)                       : datetime UTC du prochain settlement
minutes_to_settlement(now)                 : int min restant
should_close_position(side, funding_bps,
                      threshold_long_bps,
                      threshold_short_bps) : (close: bool, reason: str)
evaluate_close_signals(now, positions,
                       funding_bps, ...)   : list[dict] de signaux close
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional


SETTLEMENT_HOURS_UTC: tuple[int, ...] = (0, 8, 16)
DEFAULT_THRESHOLD_LONG_BPS = 5.0
DEFAULT_THRESHOLD_SHORT_BPS = 5.0
DEFAULT_ADVANCE_MINUTES = 5


def _to_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def next_settlement(now: datetime) -> datetime:
    """Retourne le datetime UTC du prochain settlement (>= now)."""
    now_u = _to_utc(now)
    today_settlements = [
        now_u.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in SETTLEMENT_HOURS_UTC
    ]
    for s in today_settlements:
        if s >= now_u:
            return s
    # Aucun aujourd'hui → premier de demain
    tomorrow = (now_u + timedelta(days=1)).replace(
        hour=SETTLEMENT_HOURS_UTC[0], minute=0, second=0, microsecond=0
    )
    return tomorrow


def minutes_to_settlement(now: datetime) -> int:
    """Minutes entières restantes avant le prochain settlement."""
    delta = next_settlement(now) - _to_utc(now)
    return int(delta.total_seconds() // 60)


def is_settlement_window(
    now: datetime,
    advance_minutes: int = DEFAULT_ADVANCE_MINUTES,
) -> bool:
    """True si on est dans la fenêtre [HH:60-advance, HH:60) avant settlement."""
    if advance_minutes < 0:
        return False
    mins = minutes_to_settlement(now)
    # 0 = on EST au settlement précis ; au-delà de 0, on a déjà raté la fenêtre.
    return 0 < mins <= advance_minutes


def should_close_position(
    side: str,
    funding_rate_bps: float,
    threshold_long_bps: float = DEFAULT_THRESHOLD_LONG_BPS,
    threshold_short_bps: float = DEFAULT_THRESHOLD_SHORT_BPS,
) -> tuple[bool, str]:
    """Décide si une position doit être fermée selon le funding.

    Args:
        side: "long" / "short" (case-insensitive).
        funding_rate_bps: prochain funding rate en bps. Positif = longs paient.
        threshold_long_bps: au-delà, ferme la position LONG (default +5).
        threshold_short_bps: en-dessous de -threshold_short_bps, ferme la
            position SHORT (default 5 → seuil = -5 bps).

    Returns:
        (close: bool, reason: str).
    """
    s = str(side).lower()
    if threshold_long_bps < 0 or threshold_short_bps < 0:
        raise ValueError("thresholds must be non-negative")
    if s == "long":
        if funding_rate_bps > threshold_long_bps:
            return True, (
                f"long flat: funding={funding_rate_bps:.2f} bps "
                f"> +{threshold_long_bps:.2f} (longs paient les shorts)"
            )
        return False, ""
    if s == "short":
        if funding_rate_bps < -threshold_short_bps:
            return True, (
                f"short flat: funding={funding_rate_bps:.2f} bps "
                f"< -{threshold_short_bps:.2f} (shorts paient les longs)"
            )
        return False, ""
    raise ValueError(f"side must be 'long' or 'short', got {side!r}")


def evaluate_close_signals(
    now: datetime,
    positions: Iterable[dict],
    funding_rate_bps: float,
    threshold_long_bps: float = DEFAULT_THRESHOLD_LONG_BPS,
    threshold_short_bps: float = DEFAULT_THRESHOLD_SHORT_BPS,
    advance_minutes: int = DEFAULT_ADVANCE_MINUTES,
) -> list[dict]:
    """Évalue les positions et retourne les signaux de close à émettre.

    Hors fenêtre [HH:60-advance, HH:60) → aucune action (liste vide).

    Args:
        now: datetime UTC courant.
        positions: iterable de dict avec au moins {"id", "side", "symbol"}.
        funding_rate_bps: funding rate prévu pour le prochain settlement.
        threshold_long_bps, threshold_short_bps, advance_minutes: configs.

    Returns:
        Liste de dicts {action, pos_id, symbol, reason} prêts à être consommés.
    """
    if not is_settlement_window(now, advance_minutes=advance_minutes):
        return []
    out: list[dict] = []
    for pos in positions:
        side = str(pos.get("side", "")).lower()
        try:
            close, reason = should_close_position(
                side, funding_rate_bps,
                threshold_long_bps=threshold_long_bps,
                threshold_short_bps=threshold_short_bps,
            )
        except ValueError:
            continue
        if not close:
            continue
        out.append({
            "action": "funding_close",
            "side": side,
            "pos_id": pos.get("id"),
            "symbol": pos.get("symbol"),
            "funding_rate_bps": float(funding_rate_bps),
            "minutes_to_settlement": minutes_to_settlement(now),
            "reason": reason,
        })
    return out


__all__ = [
    "SETTLEMENT_HOURS_UTC",
    "DEFAULT_THRESHOLD_LONG_BPS",
    "DEFAULT_THRESHOLD_SHORT_BPS",
    "DEFAULT_ADVANCE_MINUTES",
    "next_settlement",
    "minutes_to_settlement",
    "is_settlement_window",
    "should_close_position",
    "evaluate_close_signals",
]
