"""Per-phase Ichimoku presets.

This small helper isolates the mapping between market phases and
preferred Ichimoku lengths so it can be tuned without touching the main
module. Parameters are expressed in bars.
"""
from __future__ import annotations

from typing import Dict, Tuple

PHASE_ICHIMOKU: Dict[str, Tuple[int, int, int]] = {
    "accumulation": (6, 26, 52),
    "expansion": (6, 43, 100),
    "euphoria": (6, 55, 120),
    "distribution": (9, 65, 120),
    "bear": (6, 26, 100),
}


def ichimoku_params_for_phase(
    phase: str, mapping: Dict[str, Tuple[int, int, int]] = PHASE_ICHIMOKU
) -> Tuple[int, int, int]:
    """Return (tenkan, kijun, senkou_b) for the given ``phase``.

    If the phase is unknown, fall back to the classic 9-26-52 tuple.
    """

    return mapping.get(phase, (9, 26, 52))
