"""P7 / R5 — VPIN gate state machine — Option A (reset wins).

VPIN (Volume-synchronized Probability of Informed Trading, Easley-LdP-O'Hara
2012) détecte la toxicité du flow d'order book : valeur élevée → trading
informé asymétrique → mouvement directionnel imminent. Le gate utilise VPIN
+ Order Book Imbalance (OBI) pour décider :
  - allow              : trading normal
  - block_new_entries  : VPIN toxique, on n'ouvre plus
  - kill_and_block     : VPIN extrême + OBI déséquilibré → flat all + block

État machine
------------
Per cycle :
    if currently_blocked:
        if VPIN < reset_threshold     -> unblock (reset wins, exit normal)
        elif now > expires_at_ms      -> unblock (timer expiry, safety net)
        else                          -> stay blocked
    else:
        if VPIN > kill AND OBI < kill_obi  -> kill_and_block, set expires
        elif VPIN > block_threshold        -> block, set expires
        else                                -> allow

Block lift conditions (OR) :
  1. VPIN < reset_threshold   (normal exit, primary mechanism)
  2. now > expires_at_ms      (safety ceiling, failsafe)

Rationale : VPIN reflète l'état réel du flow d'order book. Si le flow
redevient sain (VPIN bas), bloquer plus longtemps n'apporte rien. Le
timer existe uniquement comme garde-fou contre une lecture VPIN
corrompue ou un bug qui maintiendrait artificiellement VPIN haut.

Décision design actée par Sky le 2026-04-28 (cf docs/PROTOCOL_REPORTS.md
section R5).

Référence
---------
Easley, D., López de Prado, M., O'Hara, M. (2012). "The Volume Clock:
Insights into the High-Frequency Paradigm." Journal of Portfolio
Management, 39(1), 19-29.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


# Actions retournées par le gate
ActionType = Literal["allow", "block_new_entries", "kill_and_block"]
ModeType = Literal["log_only", "gate"]


@dataclass
class VPINState:
    """État persistant du gate (à sauvegarder dans state.json par symbole)."""

    currently_blocked: bool = False
    block_expires_at_ms: Optional[int] = None
    last_action: str = "allow"
    last_reason: str = ""
    last_eval_ts_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VPINState":
        if not d:
            return cls()
        return cls(
            currently_blocked=bool(d.get("currently_blocked", False)),
            block_expires_at_ms=d.get("block_expires_at_ms"),
            last_action=str(d.get("last_action", "allow")),
            last_reason=str(d.get("last_reason", "")),
            last_eval_ts_ms=d.get("last_eval_ts_ms"),
        )


@dataclass
class VPINGateConfig:
    """Configuration immuable lue depuis bot_settings.yaml."""

    mode: ModeType = "log_only"
    block_threshold: float = 0.70
    kill_threshold: float = 0.85
    kill_obi_threshold: float = 0.30
    reset_threshold: float = 0.50
    block_duration_minutes: int = 15

    def __post_init__(self):
        # Validations soft : on clamp dans [0, 1] sans raise (tolérant aux confs)
        self.block_threshold = float(max(0.0, min(1.0, self.block_threshold)))
        self.kill_threshold = float(max(0.0, min(1.0, self.kill_threshold)))
        self.reset_threshold = float(max(0.0, min(1.0, self.reset_threshold)))
        self.kill_obi_threshold = float(max(0.0, min(1.0, self.kill_obi_threshold)))
        self.block_duration_minutes = int(max(1, self.block_duration_minutes))
        if self.mode not in ("log_only", "gate"):
            self.mode = "log_only"


def vpin_gate_check(
    current_vpin: float,
    current_obi: float,
    state: VPINState,
    config: VPINGateConfig,
    now_ms: int,
) -> tuple[ActionType, str, VPINState]:
    """Évalue un cycle du gate. Retourne (action, reason, new_state).

    new_state est un VPINState mis à jour ; le caller doit le persister
    (state.json) pour le prochain cycle.

    Args:
        current_vpin: VPIN actuel ∈ [0, 1] (cf src.vpin.compute_vpin).
        current_obi: Order Book Imbalance ∈ [0, 1] (1.0 = parfaitement
            balanced; 0.0 = un côté du book domine 100%).
        state: VPINState du cycle précédent (ou par défaut).
        config: VPINGateConfig depuis bot_settings.yaml.
        now_ms: timestamp courant en ms (UTC).

    Returns:
        (action, reason, new_state) :
        - action ∈ {"allow", "block_new_entries", "kill_and_block"}
        - reason: chaîne machine-readable (loggable)
        - new_state: VPINState mis à jour
    """
    # Mode log_only : on n'agit jamais, on retourne allow + tag mode
    if config.mode == "log_only":
        new_state = VPINState(
            currently_blocked=False,
            block_expires_at_ms=None,
            last_action="allow",
            last_reason="log_only_mode",
            last_eval_ts_ms=now_ms,
        )
        return ("allow", "log_only_mode", new_state)

    # Mode gate
    if state.currently_blocked:
        # 1. Reset normal : VPIN sain → unblock immédiat
        if current_vpin < config.reset_threshold:
            new_state = VPINState(
                currently_blocked=False,
                block_expires_at_ms=None,
                last_action="allow",
                last_reason="vpin_reset_below_threshold",
                last_eval_ts_ms=now_ms,
            )
            return ("allow", "vpin_reset_below_threshold", new_state)
        # 2. Timer expiry : safety ceiling
        if state.block_expires_at_ms is not None and now_ms > state.block_expires_at_ms:
            new_state = VPINState(
                currently_blocked=False,
                block_expires_at_ms=None,
                last_action="allow",
                last_reason="timer_expiry",
                last_eval_ts_ms=now_ms,
            )
            return ("allow", "timer_expiry", new_state)
        # Sinon on reste bloqué (VPIN encore au-dessus reset, timer non expiré)
        new_state = VPINState(
            currently_blocked=True,
            block_expires_at_ms=state.block_expires_at_ms,
            last_action="block_new_entries",
            last_reason="state_maintained",
            last_eval_ts_ms=now_ms,
        )
        return ("block_new_entries", "state_maintained", new_state)

    # Pas bloqué : on évalue les déclencheurs
    expires_at = now_ms + int(config.block_duration_minutes) * 60_000
    if current_vpin > config.kill_threshold and current_obi < config.kill_obi_threshold:
        new_state = VPINState(
            currently_blocked=True,
            block_expires_at_ms=expires_at,
            last_action="kill_and_block",
            last_reason="vpin_cascade_imminent",
            last_eval_ts_ms=now_ms,
        )
        return ("kill_and_block", "vpin_cascade_imminent", new_state)
    if current_vpin > config.block_threshold:
        new_state = VPINState(
            currently_blocked=True,
            block_expires_at_ms=expires_at,
            last_action="block_new_entries",
            last_reason="vpin_toxic_flow",
            last_eval_ts_ms=now_ms,
        )
        return ("block_new_entries", "vpin_toxic_flow", new_state)
    # Tout va bien
    new_state = VPINState(
        currently_blocked=False,
        block_expires_at_ms=None,
        last_action="allow",
        last_reason="vpin_below_threshold",
        last_eval_ts_ms=now_ms,
    )
    return ("allow", "vpin_below_threshold", new_state)


__all__ = [
    "ActionType",
    "ModeType",
    "VPINState",
    "VPINGateConfig",
    "vpin_gate_check",
]
