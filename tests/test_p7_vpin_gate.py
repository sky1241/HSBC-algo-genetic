"""P7 / R5 — Tests VPIN gate state machine (Option A reset wins)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vpin_gate import (
    VPINGateConfig,
    VPINState,
    vpin_gate_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate_config(mode="gate", **overrides) -> VPINGateConfig:
    base = dict(
        mode=mode,
        block_threshold=0.70,
        kill_threshold=0.85,
        kill_obi_threshold=0.30,
        reset_threshold=0.50,
        block_duration_minutes=15,
    )
    base.update(overrides)
    return VPINGateConfig(**base)


# ---------------------------------------------------------------------------
# Mode log_only — never blocks
# ---------------------------------------------------------------------------


def test_vpin_log_only_mode_never_blocks():
    """Mode log_only : même VPIN=0.95 → action='allow'."""
    cfg = _gate_config(mode="log_only")
    state = VPINState()
    action, reason, new_state = vpin_gate_check(
        current_vpin=0.95, current_obi=0.10,
        state=state, config=cfg, now_ms=0,
    )
    assert action == "allow"
    assert reason == "log_only_mode"
    assert new_state.currently_blocked is False


def test_vpin_log_only_does_not_persist_state_changes():
    """log_only ne propage jamais currently_blocked=True dans new_state."""
    cfg = _gate_config(mode="log_only")
    # Même si l'ancien state était blocked (transition gate→log_only), on reset
    state = VPINState(currently_blocked=True, block_expires_at_ms=999_999_999)
    _, _, new_state = vpin_gate_check(0.95, 0.1, state, cfg, now_ms=1000)
    assert new_state.currently_blocked is False
    assert new_state.block_expires_at_ms is None


# ---------------------------------------------------------------------------
# Mode gate — déclencheurs primaires
# ---------------------------------------------------------------------------


def test_vpin_below_block_threshold_allows():
    """VPIN=0.65 (<0.70) → allow."""
    cfg = _gate_config()
    action, reason, new_state = vpin_gate_check(
        current_vpin=0.65, current_obi=0.50,
        state=VPINState(), config=cfg, now_ms=1_700_000_000_000,
    )
    assert action == "allow"
    assert reason == "vpin_below_threshold"
    assert new_state.currently_blocked is False


def test_vpin_above_block_below_kill_blocks():
    """VPIN=0.75, OBI=0.50 → block_new_entries (pas kill car OBI haut)."""
    cfg = _gate_config()
    action, reason, new_state = vpin_gate_check(
        current_vpin=0.75, current_obi=0.50,
        state=VPINState(), config=cfg, now_ms=1_700_000_000_000,
    )
    assert action == "block_new_entries"
    assert reason == "vpin_toxic_flow"
    assert new_state.currently_blocked is True
    # expires_at = now + 15min
    assert new_state.block_expires_at_ms == 1_700_000_000_000 + 15 * 60_000


def test_vpin_above_kill_with_low_obi_kills():
    """VPIN=0.90, OBI=0.20 → kill_and_block (cascade imminente)."""
    cfg = _gate_config()
    action, reason, new_state = vpin_gate_check(
        current_vpin=0.90, current_obi=0.20,
        state=VPINState(), config=cfg, now_ms=0,
    )
    assert action == "kill_and_block"
    assert reason == "vpin_cascade_imminent"
    assert new_state.currently_blocked is True


def test_vpin_above_kill_with_high_obi_blocks_only():
    """VPIN=0.90 mais OBI=0.50 (>0.30) → block_new_entries (pas kill)."""
    cfg = _gate_config()
    action, _, _ = vpin_gate_check(
        current_vpin=0.90, current_obi=0.50,
        state=VPINState(), config=cfg, now_ms=0,
    )
    assert action == "block_new_entries"


# ---------------------------------------------------------------------------
# Hystérésis (option A reset wins)
# ---------------------------------------------------------------------------


def test_vpin_hysteresis_no_flapping():
    """Séquence VPIN [0.68, 0.72, 0.68, 0.72, 0.68] → block reste actif.

    Hystérésis : block_threshold=0.70 et reset_threshold=0.50. VPIN qui
    oscille entre 0.68 et 0.72 ne déclenche pas de flapping car :
      - 0.68 > reset_threshold (pas d'unblock)
      - 0.72 > block_threshold (block déclenché ou maintenu)
    Donc on RESTE bloqué (sauf si oscillation avant le 1er trigger).
    """
    cfg = _gate_config()
    sequence = [0.68, 0.72, 0.68, 0.72, 0.68]
    state = VPINState()
    actions = []
    for i, vpin in enumerate(sequence):
        action, _, state = vpin_gate_check(
            current_vpin=vpin, current_obi=0.5,
            state=state, config=cfg, now_ms=i * 1000,
        )
        actions.append(action)
    # Le 1er VPIN=0.68 < block_threshold → allow ; puis 0.72 → block ; après on reste blocked
    assert actions[0] == "allow"
    assert actions[1] == "block_new_entries"
    assert actions[2] == "block_new_entries"  # 0.68 > reset_threshold 0.50, pas d'unblock
    assert actions[3] == "block_new_entries"
    assert actions[4] == "block_new_entries"


def test_vpin_reset_unblocks_before_timer():
    """VPIN=0.80 à T=0, VPIN=0.40 à T=5min → unblock à T=5min (reset wins)."""
    cfg = _gate_config(block_duration_minutes=15)
    # T=0 : trigger
    a1, _, state = vpin_gate_check(
        current_vpin=0.80, current_obi=0.5,
        state=VPINState(), config=cfg, now_ms=0,
    )
    assert a1 == "block_new_entries"
    assert state.currently_blocked is True
    # T=5min : VPIN drop sous reset
    a2, reason2, new_state = vpin_gate_check(
        current_vpin=0.40, current_obi=0.5,
        state=state, config=cfg, now_ms=5 * 60_000,
    )
    assert a2 == "allow"
    assert reason2 == "vpin_reset_below_threshold"
    assert new_state.currently_blocked is False


def test_vpin_block_duration_acts_as_safety_ceiling():
    """VPIN=0.80 maintenu 20min sans descendre → block lift à T=15min (timer)."""
    cfg = _gate_config(block_duration_minutes=15)
    # T=0 : trigger
    _, _, state = vpin_gate_check(
        current_vpin=0.80, current_obi=0.5,
        state=VPINState(), config=cfg, now_ms=0,
    )
    # T=10min : VPIN encore haut, reset_threshold=0.50, on reste bloqué
    a10, reason10, state = vpin_gate_check(
        current_vpin=0.80, current_obi=0.5,
        state=state, config=cfg, now_ms=10 * 60_000,
    )
    assert a10 == "block_new_entries"
    assert reason10 == "state_maintained"
    # T=20min : VPIN encore 0.80 mais timer 15min expiré → unblock par safety
    a20, reason20, state = vpin_gate_check(
        current_vpin=0.80, current_obi=0.5,
        state=state, config=cfg, now_ms=20 * 60_000,
    )
    assert a20 == "allow"
    assert reason20 == "timer_expiry"
    assert state.currently_blocked is False


def test_vpin_stays_blocked_when_above_reset_within_duration():
    """VPIN=0.80 à T=0, VPIN=0.55 à T=10min (>reset) → reste blocked."""
    cfg = _gate_config(block_duration_minutes=15)
    _, _, state = vpin_gate_check(0.80, 0.5, VPINState(), cfg, now_ms=0)
    a, reason, _ = vpin_gate_check(0.55, 0.5, state, cfg, now_ms=10 * 60_000)
    assert a == "block_new_entries"
    assert reason == "state_maintained"


def test_vpin_stays_blocked_when_above_reset_after_duration_expired():
    """VPIN=0.80 à T=0, VPIN=0.55 à T=20min (timer expired mais > reset).

    Le timer wins ici : VPIN n'est jamais descendu sous reset pendant la
    fenêtre, donc unblock par expiry. Au cycle suivant, VPIN=0.55 est sous
    block_threshold → allow (pas de re-block).

    Edge case important : pas de "limbo state". L'état passe à
    currently_blocked=False, et l'évaluation suivante repart du `else:` du
    state machine.
    """
    cfg = _gate_config(block_duration_minutes=15)
    _, _, state = vpin_gate_check(0.80, 0.5, VPINState(), cfg, now_ms=0)
    a1, reason1, state = vpin_gate_check(0.55, 0.5, state, cfg, now_ms=20 * 60_000)
    assert a1 == "allow"
    assert reason1 == "timer_expiry"
    # Cycle suivant : état non-blocked, VPIN=0.55 < block_threshold 0.70 → allow
    a2, reason2, state = vpin_gate_check(0.55, 0.5, state, cfg, now_ms=21 * 60_000)
    assert a2 == "allow"
    assert reason2 == "vpin_below_threshold"


def test_vpin_post_timer_can_immediately_reblock():
    """Après expiry, si VPIN encore > block_threshold → re-block immédiat."""
    cfg = _gate_config(block_duration_minutes=15)
    _, _, state = vpin_gate_check(0.80, 0.5, VPINState(), cfg, now_ms=0)
    # T=20min : VPIN tjs élevé mais > reset → state_maintained (on n'expire pas
    # tant que le check des conditions blocked se fait dans l'ordre reset->timer)
    # Mais ici 0.55 > reset → ne déclenche pas reset, mais 20min > expires → timer
    a1, _, state = vpin_gate_check(0.55, 0.5, state, cfg, now_ms=20 * 60_000)
    assert a1 == "allow"  # timer_expiry
    # Cycle suivant : VPIN re-monte à 0.85 → re-block (kill car OBI=0.20)
    a2, reason2, state = vpin_gate_check(0.95, 0.20, state, cfg, now_ms=21 * 60_000)
    assert a2 == "kill_and_block"
    assert reason2 == "vpin_cascade_imminent"


def test_vpin_reset_threshold_only_when_blocked():
    """État initial not blocked. VPIN=0.45 (<reset) → reste 'allow' (pas reset)."""
    cfg = _gate_config()
    a, reason, _ = vpin_gate_check(0.45, 0.5, VPINState(), cfg, now_ms=0)
    assert a == "allow"
    assert reason == "vpin_below_threshold"  # pas vpin_reset_below_threshold


# ---------------------------------------------------------------------------
# VPINState dataclass
# ---------------------------------------------------------------------------


def test_vpin_state_to_dict_from_dict_roundtrip():
    s = VPINState(
        currently_blocked=True, block_expires_at_ms=12345,
        last_action="kill_and_block", last_reason="cascade",
        last_eval_ts_ms=10,
    )
    d = s.to_dict()
    s2 = VPINState.from_dict(d)
    assert s2 == s


def test_vpin_state_from_none_returns_default():
    assert VPINState.from_dict(None) == VPINState()


def test_vpin_state_from_partial_dict_uses_defaults():
    s = VPINState.from_dict({"currently_blocked": True})
    assert s.currently_blocked is True
    assert s.block_expires_at_ms is None
    assert s.last_action == "allow"


# ---------------------------------------------------------------------------
# VPINGateConfig validations
# ---------------------------------------------------------------------------


def test_config_clamps_thresholds_to_unit_interval():
    cfg = VPINGateConfig(block_threshold=1.5, reset_threshold=-0.2)
    assert cfg.block_threshold == 1.0
    assert cfg.reset_threshold == 0.0


def test_config_invalid_mode_falls_back_to_log_only():
    cfg = VPINGateConfig(mode="not_a_mode")  # type: ignore
    assert cfg.mode == "log_only"


def test_config_minimum_duration_is_one_minute():
    cfg = VPINGateConfig(block_duration_minutes=0)
    assert cfg.block_duration_minutes == 1
