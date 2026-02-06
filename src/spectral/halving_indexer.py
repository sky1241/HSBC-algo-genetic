#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.4 - Bitcoin Halving Phase Indexer

Aligns time axis to t=0 at each halving event for cycle-relative analysis.

Phases définies:
- PRE_HALVING: t ∈ [-180j, 0[ avant le halving
- DISCOVERY: t ∈ [0, +90j] post-halving phase I
- EXPANSION: t ∈ [+90j, +270j] post-halving phase II
- MATURATION: t ∈ [+270j, +540j] post-halving phase III
- LATE_CYCLE: t > +540j jusqu'au prochain pré-halving

Usage:
    from src.spectral.halving_indexer import HalvingIndexer, get_halving_phase

    indexer = HalvingIndexer()
    phase = indexer.get_phase(pd.Timestamp("2024-06-01"))
    # Returns: HalvingPhase.EXPANSION

    # Or use standalone function
    phase = get_halving_phase(pd.Timestamp("2024-06-01"))
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd


class HalvingPhase(Enum):
    """BTC halving cycle phases."""
    PRE_HALVING = "pre_halving"      # t ∈ [-180j, 0[
    DISCOVERY = "discovery"           # t ∈ [0, +90j]
    EXPANSION = "expansion"           # t ∈ [+90j, +270j]
    MATURATION = "maturation"         # t ∈ [+270j, +540j]
    LATE_CYCLE = "late_cycle"         # t > +540j
    UNKNOWN = "unknown"               # Before first halving or invalid


# Historical and projected BTC halving dates
HALVING_DATES: List[datetime] = [
    datetime(2012, 11, 28),   # Halving 1: Block 210,000
    datetime(2016, 7, 9),     # Halving 2: Block 420,000
    datetime(2020, 5, 11),    # Halving 3: Block 630,000
    datetime(2024, 4, 20),    # Halving 4: Block 840,000 (actual date)
    datetime(2028, 4, 1),     # Halving 5: Block 1,050,000 (projected)
    datetime(2032, 4, 1),     # Halving 6: Block 1,260,000 (projected)
]


@dataclass
class CycleInfo:
    """Information about position within halving cycle."""
    halving_date: datetime
    halving_number: int
    days_since_halving: int
    days_until_next: int
    phase: HalvingPhase
    cycle_progress: float  # 0-1 progress through current cycle


class HalvingIndexer:
    """Indexes timestamps relative to BTC halving events.

    Features:
    - Get phase (pre-halving, discovery, expansion, maturation, late-cycle)
    - Calculate days since/until halving
    - Align dataframes to t=0 at halving
    """

    # Phase boundaries in days relative to halving
    PHASE_BOUNDS = {
        "pre_start": -180,     # Pre-halving starts 180 days before
        "discovery_end": 90,   # Discovery: 0 to +90 days
        "expansion_end": 270,  # Expansion: +90 to +270 days
        "maturation_end": 540, # Maturation: +270 to +540 days
    }

    def __init__(self, halving_dates: Optional[List[datetime]] = None):
        self.halving_dates = sorted(halving_dates or HALVING_DATES)

    def _find_cycle(self, dt: datetime) -> Tuple[Optional[datetime], Optional[datetime], int]:
        """Find the current and next halving for a given date.

        Returns:
            (current_halving, next_halving, halving_number)
        """
        # Before first halving
        if dt < self.halving_dates[0]:
            return None, self.halving_dates[0], 0

        # Find the most recent halving
        for i, halving in enumerate(self.halving_dates):
            if i + 1 < len(self.halving_dates):
                next_halving = self.halving_dates[i + 1]
                if halving <= dt < next_halving:
                    return halving, next_halving, i + 1
            else:
                # After last known halving
                return halving, None, i + 1

        return self.halving_dates[-1], None, len(self.halving_dates)

    def get_phase(self, dt: datetime) -> HalvingPhase:
        """Get the halving phase for a given datetime."""
        current, next_h, _ = self._find_cycle(dt)

        if current is None:
            # Before first halving, check if in pre-halving phase
            if next_h is not None:
                days_until = (next_h - dt).days
                if days_until <= -self.PHASE_BOUNDS["pre_start"]:
                    return HalvingPhase.PRE_HALVING
            return HalvingPhase.UNKNOWN

        days_since = (dt - current).days

        # Check if in next cycle's pre-halving phase
        if next_h is not None:
            days_until_next = (next_h - dt).days
            if days_until_next <= -self.PHASE_BOUNDS["pre_start"]:
                return HalvingPhase.PRE_HALVING

        # Determine phase based on days since halving
        if days_since < 0:
            return HalvingPhase.PRE_HALVING
        elif days_since <= self.PHASE_BOUNDS["discovery_end"]:
            return HalvingPhase.DISCOVERY
        elif days_since <= self.PHASE_BOUNDS["expansion_end"]:
            return HalvingPhase.EXPANSION
        elif days_since <= self.PHASE_BOUNDS["maturation_end"]:
            return HalvingPhase.MATURATION
        else:
            return HalvingPhase.LATE_CYCLE

    def get_cycle_info(self, dt: datetime) -> CycleInfo:
        """Get detailed cycle information for a datetime."""
        current, next_h, halving_num = self._find_cycle(dt)
        phase = self.get_phase(dt)

        if current is None:
            days_since = 0
            halving_date = next_h or self.halving_dates[0]
        else:
            days_since = (dt - current).days
            halving_date = current

        if next_h is not None:
            days_until = (next_h - dt).days
            if current is not None:
                cycle_length = (next_h - current).days
                cycle_progress = days_since / cycle_length if cycle_length > 0 else 0.0
            else:
                cycle_progress = 0.0
        else:
            days_until = 0
            cycle_progress = 1.0

        return CycleInfo(
            halving_date=halving_date,
            halving_number=halving_num,
            days_since_halving=days_since,
            days_until_next=days_until,
            phase=phase,
            cycle_progress=min(1.0, max(0.0, cycle_progress)),
        )

    def align_to_halving(
        self,
        df: pd.DataFrame,
        halving_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Align DataFrame index to t=0 at halving.

        Args:
            df: DataFrame with datetime index
            halving_date: Specific halving to align to (uses nearest if None)

        Returns:
            DataFrame with 't_days' column (days relative to halving)
        """
        df = df.copy()

        if halving_date is None:
            # Use the halving that falls within the data range
            start = df.index.min()
            end = df.index.max()
            for h in self.halving_dates:
                if start <= pd.Timestamp(h) <= end:
                    halving_date = h
                    break
            if halving_date is None:
                # Use nearest halving
                mid = start + (end - start) / 2
                halving_date = min(self.halving_dates, key=lambda h: abs((h - mid.to_pydatetime()).days))

        halving_ts = pd.Timestamp(halving_date)
        df["t_days"] = (df.index - halving_ts).total_seconds() / 86400.0
        df["halving_phase"] = df.index.map(lambda x: self.get_phase(x.to_pydatetime()).value)

        return df

    def split_by_phase(self, df: pd.DataFrame) -> Dict[HalvingPhase, pd.DataFrame]:
        """Split DataFrame by halving phase.

        Returns:
            Dictionary mapping HalvingPhase to DataFrame slice
        """
        df = df.copy()
        df["_phase"] = df.index.map(lambda x: self.get_phase(x.to_pydatetime()))

        result = {}
        for phase in HalvingPhase:
            mask = df["_phase"] == phase
            if mask.any():
                result[phase] = df[mask].drop(columns=["_phase"])

        return result


# Singleton instance for convenience
_default_indexer = HalvingIndexer()


def get_halving_phase(dt: datetime) -> HalvingPhase:
    """Get halving phase for a datetime (convenience function)."""
    return _default_indexer.get_phase(dt)


def get_cycle_info(dt: datetime) -> CycleInfo:
    """Get cycle info for a datetime (convenience function)."""
    return _default_indexer.get_cycle_info(dt)


def add_halving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add halving-related features to a DataFrame.

    Adds columns:
    - halving_phase: str (phase name)
    - days_since_halving: int
    - cycle_progress: float [0,1]
    """
    df = df.copy()

    def _get_features(ts):
        info = _default_indexer.get_cycle_info(ts.to_pydatetime())
        return pd.Series({
            "halving_phase": info.phase.value,
            "days_since_halving": info.days_since_halving,
            "cycle_progress": info.cycle_progress,
        })

    features = df.index.to_series().apply(_get_features)
    return pd.concat([df, features], axis=1)
