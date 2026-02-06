#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 - Parameter Pools by Regime

Defines parameter search spaces conditioned by market regime.

Pools:
- TREND: For trending markets (LFP > 0.6) - wider stops, longer periods
- MIXED: For transitional markets - balanced parameters
- NOISE: For choppy markets (high flatness) - tight stops, faster periods

Usage:
    from src.spectral.param_pools import get_pool_for_regime, sample_from_pool
    from src.spectral.fourier_features import RegimeType

    pool = get_pool_for_regime(RegimeType.TREND)
    params = sample_from_pool(pool, trial)  # Optuna trial
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import random


@dataclass
class ParamPool:
    """Parameter search space for Optuna optimization.

    Attributes:
        name: Pool identifier
        tenkan_range: (min, max) for tenkan period
        kijun_range: (min, max) for kijun period
        senkou_b_range: (min, max) for senkou_b period
        shift_range: (min, max) for shift
        atr_mult_range: (min, max) for ATR multiplier
        tp_mult_range: (min, max) for take profit multiplier
        r_kijun_range: (min, max) ratio kijun/tenkan constraint
        entry_strictness: Signal filter strictness (0=loose, 1=strict)
        description: Human-readable description
    """
    name: str
    tenkan_range: Tuple[int, int]
    kijun_range: Tuple[int, int]
    senkou_b_range: Tuple[int, int]
    shift_range: Tuple[int, int]
    atr_mult_range: Tuple[float, float]
    tp_mult_range: Tuple[float, float]
    r_kijun_range: Tuple[int, int] = (2, 4)
    entry_strictness: float = 0.5
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tenkan": self.tenkan_range,
            "kijun": self.kijun_range,
            "senkou_b": self.senkou_b_range,
            "shift": self.shift_range,
            "atr_mult": self.atr_mult_range,
            "tp_mult": self.tp_mult_range,
            "r_kijun": self.r_kijun_range,
            "entry_strictness": self.entry_strictness,
        }


# ============================================================================
# Pre-defined pools based on doc/IDEES_OPTIMISATION_HALVING_FR.md
# ============================================================================

POOL_TREND = ParamPool(
    name="trend",
    tenkan_range=(15, 35),       # Slower tenkan for trend following
    kijun_range=(55, 100),       # Wide kijun for macro trend
    senkou_b_range=(80, 200),    # Very wide senkou_b
    shift_range=(40, 80),        # Longer anticipation
    atr_mult_range=(3.0, 5.0),   # Wider stops (let trends run)
    tp_mult_range=(8.0, 20.0),   # Higher TP for big moves
    r_kijun_range=(2, 4),
    entry_strictness=0.3,        # Loose entry (don't miss trends)
    description="Trending market: wide stops, slow periods, capture big moves",
)

POOL_MIXED = ParamPool(
    name="mixed",
    tenkan_range=(8, 25),        # Balanced tenkan
    kijun_range=(35, 70),        # Balanced kijun
    senkou_b_range=(50, 120),    # Balanced senkou_b
    shift_range=(20, 50),        # Moderate anticipation
    atr_mult_range=(2.5, 4.0),   # Medium stops
    tp_mult_range=(5.0, 12.0),   # Medium TP
    r_kijun_range=(2, 4),
    entry_strictness=0.5,        # Balanced entry
    description="Mixed/transitional market: balanced parameters",
)

POOL_NOISE = ParamPool(
    name="noise",
    tenkan_range=(5, 18),        # Fast tenkan (react to noise)
    kijun_range=(26, 55),        # Tight kijun
    senkou_b_range=(40, 90),     # Tighter senkou_b
    shift_range=(10, 30),        # Short anticipation
    atr_mult_range=(2.0, 3.0),   # Tight stops (cut losses fast)
    tp_mult_range=(3.0, 8.0),    # Lower TP (take profits quick)
    r_kijun_range=(2, 3),
    entry_strictness=0.8,        # Strict entry (filter false signals)
    description="Noisy/choppy market: tight stops, fast periods, strict entry",
)

# Alias pools by halving phase (as per doc recommendations)
POOL_BY_HALVING_PHASE = {
    "pre_halving": POOL_MIXED,      # Uncertainty before halving
    "discovery": POOL_MIXED,         # Post-halving adjustment
    "expansion": POOL_TREND,         # Bull run phase
    "maturation": POOL_MIXED,        # Topping pattern
    "late_cycle": POOL_NOISE,        # Distribution/bear preparation
}


def get_pool_for_regime(regime: "RegimeType") -> ParamPool:
    """Get the appropriate parameter pool for a market regime.

    Args:
        regime: RegimeType from fourier_features module

    Returns:
        ParamPool for that regime
    """
    from .fourier_features import RegimeType

    mapping = {
        RegimeType.TREND: POOL_TREND,
        RegimeType.MIXED: POOL_MIXED,
        RegimeType.NOISE: POOL_NOISE,
        RegimeType.UNKNOWN: POOL_MIXED,  # Default to mixed
    }
    return mapping.get(regime, POOL_MIXED)


def get_pool_for_halving_phase(phase: str) -> ParamPool:
    """Get parameter pool for a halving phase.

    Args:
        phase: Phase name (pre_halving, discovery, expansion, maturation, late_cycle)

    Returns:
        ParamPool for that phase
    """
    return POOL_BY_HALVING_PHASE.get(phase, POOL_MIXED)


def sample_from_pool(pool: ParamPool, trial: Optional[Any] = None) -> Dict[str, Any]:
    """Sample parameters from a pool, optionally using Optuna trial.

    Args:
        pool: ParamPool to sample from
        trial: Optuna trial object (if None, use random sampling)

    Returns:
        Dictionary of sampled parameters
    """
    if trial is not None:
        # Optuna sampling with constraints
        tenkan = trial.suggest_int("tenkan", pool.tenkan_range[0], pool.tenkan_range[1])
        r_kijun = trial.suggest_int("r_kijun", pool.r_kijun_range[0], pool.r_kijun_range[1])
        kijun = max(tenkan, r_kijun * tenkan)
        # Ensure kijun is in range
        kijun = min(max(kijun, pool.kijun_range[0]), pool.kijun_range[1])

        r_senkou = trial.suggest_int("r_senkou", 1, 6)
        senkou_b = max(kijun, r_senkou * tenkan)
        senkou_b = min(max(senkou_b, pool.senkou_b_range[0]), pool.senkou_b_range[1])

        shift = trial.suggest_int("shift", pool.shift_range[0], pool.shift_range[1])
        atr_mult = trial.suggest_float("atr_mult", pool.atr_mult_range[0], pool.atr_mult_range[1], step=0.1)
        tp_mult = trial.suggest_float("tp_mult", pool.tp_mult_range[0], pool.tp_mult_range[1], step=0.5)

        return {
            "tenkan": int(tenkan),
            "kijun": int(kijun),
            "senkou_b": int(senkou_b),
            "shift": int(shift),
            "atr_mult": float(atr_mult),
            "tp_mult": float(tp_mult),
            "pool": pool.name,
            "entry_strictness": pool.entry_strictness,
        }
    else:
        # Random sampling
        tenkan = random.randint(pool.tenkan_range[0], pool.tenkan_range[1])
        r_kijun = random.randint(pool.r_kijun_range[0], pool.r_kijun_range[1])
        kijun = max(tenkan, r_kijun * tenkan)
        kijun = min(max(kijun, pool.kijun_range[0]), pool.kijun_range[1])

        senkou_b = random.randint(pool.senkou_b_range[0], pool.senkou_b_range[1])
        senkou_b = max(senkou_b, kijun)

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_b": senkou_b,
            "shift": random.randint(pool.shift_range[0], pool.shift_range[1]),
            "atr_mult": round(random.uniform(pool.atr_mult_range[0], pool.atr_mult_range[1]), 1),
            "tp_mult": round(random.uniform(pool.tp_mult_range[0], pool.tp_mult_range[1]), 1),
            "pool": pool.name,
            "entry_strictness": pool.entry_strictness,
        }


def blend_pools(
    pool_a: ParamPool,
    pool_b: ParamPool,
    weight_a: float = 0.5,
) -> ParamPool:
    """Create a blended pool from two pools.

    Useful for regime transitions where you want parameters
    between two extremes.

    Args:
        pool_a: First pool
        pool_b: Second pool
        weight_a: Weight for pool_a (0-1)

    Returns:
        New ParamPool with blended ranges
    """
    w_a = weight_a
    w_b = 1.0 - weight_a

    def blend_range(a: Tuple, b: Tuple) -> Tuple:
        return (
            int(round(a[0] * w_a + b[0] * w_b)),
            int(round(a[1] * w_a + b[1] * w_b)),
        )

    def blend_float_range(a: Tuple, b: Tuple) -> Tuple:
        return (
            round(a[0] * w_a + b[0] * w_b, 1),
            round(a[1] * w_a + b[1] * w_b, 1),
        )

    return ParamPool(
        name=f"blend_{pool_a.name}_{pool_b.name}",
        tenkan_range=blend_range(pool_a.tenkan_range, pool_b.tenkan_range),
        kijun_range=blend_range(pool_a.kijun_range, pool_b.kijun_range),
        senkou_b_range=blend_range(pool_a.senkou_b_range, pool_b.senkou_b_range),
        shift_range=blend_range(pool_a.shift_range, pool_b.shift_range),
        atr_mult_range=blend_float_range(pool_a.atr_mult_range, pool_b.atr_mult_range),
        tp_mult_range=blend_float_range(pool_a.tp_mult_range, pool_b.tp_mult_range),
        r_kijun_range=blend_range(pool_a.r_kijun_range, pool_b.r_kijun_range),
        entry_strictness=w_a * pool_a.entry_strictness + w_b * pool_b.entry_strictness,
        description=f"Blended: {pool_a.name}({w_a:.0%}) + {pool_b.name}({w_b:.0%})",
    )
