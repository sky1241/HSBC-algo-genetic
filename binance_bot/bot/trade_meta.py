"""P9 — Trade journal meta-labeling (Lopez de Prado 2018, ch. 3 "Labeling").

Capture un schéma enrichi de chaque trade clos (context + pre-trade + execution
+ exit) pour permettre à Munin (mycelium) de clusteriser les contextes
gagnants/perdants. Stockage append-only hash-chained sur data/trades_meta.jsonl
(réutilise l'AuditLog hash chain de B15).

Schema:
    {
      "trade_id": str,
      "timestamp_open": int_ms,
      "timestamp_close": int_ms,
      "symbol": "BTCUSDT" | "ETHUSDT" | "SOLUSDT",
      "side": "LONG" | "SHORT",
      "entry_price": float, "exit_price": float, "qty": float,
      "pnl_gross_usd": float, "fees_paid_usd": float,
      "funding_paid_usd": float, "pnl_net_usd": float,
      "context": {phase_K3, days_since_halving, day_of_week, hour_of_day,
                  btc_dominance},
      "pre_trade": {atr_at_entry, rv_predicted_har, regime_har, vpin_at_entry,
                    composite_signal, cloud_breakout_size_atr_units,
                    volume_relative_30d, funding_rate_at_entry_bps},
      "execution": {tf_signal_origin, n_tf_confirming, slippage_bps},
      "exit": {reason}
    }

Référence:
    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    Chapter 3 — Labeling (triple-barrier method, meta-labeling).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .audit_log import AuditLog


# Fields obligatoires top-level (pas de None autorisé pour ces clés)
_REQUIRED_TOP = (
    "trade_id",
    "timestamp_open",
    "timestamp_close",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "qty",
    "pnl_gross_usd",
    "fees_paid_usd",
    "funding_paid_usd",
    "pnl_net_usd",
    "context",
    "pre_trade",
    "execution",
    "exit",
)

_REQUIRED_CONTEXT = (
    "phase_K3",
    "days_since_halving",
    "day_of_week",
    "hour_of_day",
    "btc_dominance",
)

_REQUIRED_PRE_TRADE = (
    "atr_at_entry",
    "rv_predicted_har",
    "regime_har",
    "vpin_at_entry",
    "composite_signal",
    "cloud_breakout_size_atr_units",
    "volume_relative_30d",
    "funding_rate_at_entry_bps",
)

_REQUIRED_EXECUTION = ("tf_signal_origin", "n_tf_confirming", "slippage_bps")
_REQUIRED_EXIT = ("reason",)

_VALID_SIDES = {"LONG", "SHORT"}
_VALID_EXIT_REASONS = {
    "TP", "SL", "trailing", "EOD", "opposite_signal", "daily_cap", "manual"
}

# Anti-leak : aucun de ces termes ne doit apparaître dans une clé/valeur
# (case-insensitive) du payload pour éviter de logger secrets/PII.
_FORBIDDEN_SUBSTRINGS = (
    "api_key", "apikey", "secret", "password", "passwd",
    "private_key", "privatekey", "token", "bearer", "authorization",
)


def _check_no_pii(payload: Any, _depth: int = 0) -> Optional[str]:
    """Recurse payload; retourne le terme interdit trouvé ou None."""
    if _depth > 10:
        return None  # safety: max recursion
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(k, str):
                low = k.lower()
                for bad in _FORBIDDEN_SUBSTRINGS:
                    if bad in low:
                        return f"forbidden key '{k}'"
            err = _check_no_pii(v, _depth + 1)
            if err:
                return err
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            err = _check_no_pii(item, _depth + 1)
            if err:
                return err
    elif isinstance(payload, str):
        low = payload.lower()
        for bad in _FORBIDDEN_SUBSTRINGS:
            # On rejette uniquement si la chaîne RESSEMBLE à un secret
            # (préfixe = bad terme suivi de '=' / ':' / espace).
            if bad in low and any(sep in low for sep in ("=", ":", " ")):
                return f"forbidden secret-like value contains '{bad}'"
    return None


def build_meta_label(
    trade_id: str,
    timestamp_open: int,
    timestamp_close: int,
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl_gross_usd: float,
    fees_paid_usd: float,
    funding_paid_usd: float,
    pnl_net_usd: float,
    context: dict,
    pre_trade: dict,
    execution: dict,
    exit_info: dict,
) -> dict:
    """Construit le dict meta-label, valide les champs obligatoires.

    Lève ValueError si :
      - clé top-level manquante
      - sub-dict context/pre_trade/execution/exit avec clé manquante
      - side hors {LONG, SHORT}
      - exit reason hors enum
      - PII / secret détecté

    Les champs optionnels (None) restent dans le payload — Munin gère NaN.
    """
    side_norm = str(side).upper()
    if side_norm not in _VALID_SIDES:
        raise ValueError(f"side must be in {_VALID_SIDES}, got {side!r}")

    payload = {
        "trade_id": str(trade_id),
        "timestamp_open": int(timestamp_open),
        "timestamp_close": int(timestamp_close),
        "symbol": str(symbol),
        "side": side_norm,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "qty": float(qty),
        "pnl_gross_usd": float(pnl_gross_usd),
        "fees_paid_usd": float(fees_paid_usd),
        "funding_paid_usd": float(funding_paid_usd),
        "pnl_net_usd": float(pnl_net_usd),
        "context": dict(context),
        "pre_trade": dict(pre_trade),
        "execution": dict(execution),
        "exit": dict(exit_info),
    }

    for k in _REQUIRED_TOP:
        if k not in payload:
            raise ValueError(f"missing required top-level key: {k}")
    for k in _REQUIRED_CONTEXT:
        if k not in payload["context"]:
            raise ValueError(f"missing context key: {k}")
    for k in _REQUIRED_PRE_TRADE:
        if k not in payload["pre_trade"]:
            raise ValueError(f"missing pre_trade key: {k}")
    for k in _REQUIRED_EXECUTION:
        if k not in payload["execution"]:
            raise ValueError(f"missing execution key: {k}")
    for k in _REQUIRED_EXIT:
        if k not in payload["exit"]:
            raise ValueError(f"missing exit key: {k}")

    reason = str(payload["exit"]["reason"])
    if reason not in _VALID_EXIT_REASONS:
        raise ValueError(
            f"exit reason must be in {_VALID_EXIT_REASONS}, got {reason!r}"
        )

    err = _check_no_pii(payload)
    if err:
        raise ValueError(f"PII/secret leak detected: {err}")

    return payload


class MetaLabelLogger:
    """Wrapper AuditLog dédié aux trade meta-labels.

    Garantit append-only + hash chain (intégrité, pas confidentialité).
    """

    def __init__(self, path: str | Path = "binance_bot/data/trades_meta.jsonl"):
        self._chain = AuditLog(path)
        self.path = self._chain.path

    def write_meta_label(self, **kwargs) -> dict:
        """Construit + valide + append le meta-label. Retourne l'entry chain."""
        payload = build_meta_label(**kwargs)
        return self._chain.append(payload)

    def write_payload(self, payload: dict) -> dict:
        """Variante : payload pré-construit (déjà validé via build_meta_label)."""
        err = _check_no_pii(payload)
        if err:
            raise ValueError(f"PII/secret leak detected: {err}")
        return self._chain.append(payload)

    def verify(self):
        return self._chain.verify_chain()

    def iter_entries(self):
        return self._chain.iter_entries()


__all__ = [
    "MetaLabelLogger",
    "build_meta_label",
]
