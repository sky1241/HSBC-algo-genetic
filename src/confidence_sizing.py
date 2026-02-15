"""Confidence-based position sizing module.

Scales position size based on a continuous confidence signal (0-1).

- Base allocation: 1% of capital per trade
- With max confidence (1.0): up to 2% per trade
- Below threshold (0.5): stays at 1% (base only)
- Leverage remains FIXED at 10x (never modified)

Usage:
    from src.confidence_sizing import ConfidenceSizingConfig, compute_effective_size

    config = ConfidenceSizingConfig()
    # P(bull) = 0.8 -> 1.6% of capital per trade
    size_pct = compute_effective_size(0.8, config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConfidenceSizingConfig:
    """Configuration for confidence-based position sizing.

    Attributes:
        base_pct: Base position size as fraction of capital (0.01 = 1%)
        max_boost_pct: Maximum additional allocation at full confidence (0.01 = +1%)
        threshold: P(bull) below this -> base only, no boost
        leverage: Fixed leverage (for reference only, NEVER modified)
    """
    base_pct: float = 0.01        # 1% base
    max_boost_pct: float = 0.01   # +1% max boost -> total max 2%
    threshold: float = 0.50       # Below 0.5 = base only
    leverage: float = 10.0        # Fixed (reference, not modified)

    @property
    def max_pct(self) -> float:
        """Maximum total position size."""
        return self.base_pct + self.max_boost_pct


def compute_effective_size(
    confidence: float,
    config: Optional[ConfidenceSizingConfig] = None,
) -> float:
    """Compute effective position size based on confidence.

    Linear scaling between threshold and 1.0:
        confidence=0.50 -> 1.0% (base, at threshold)
        confidence=0.75 -> 1.5% (halfway)
        confidence=1.00 -> 2.0% (max boost)
        confidence=0.30 -> 1.0% (below threshold, base only)

    Args:
        confidence: Confidence score 0-1 (e.g., P(bull) from NHHM)
        config: Sizing configuration (uses defaults if None)

    Returns:
        Effective position size as fraction of capital
    """
    if config is None:
        config = ConfidenceSizingConfig()

    if np.isnan(confidence) or confidence <= config.threshold:
        return config.base_pct

    # Linear scale from threshold to 1.0
    range_above = 1.0 - config.threshold
    if range_above <= 0:
        return config.base_pct

    boost_ratio = (confidence - config.threshold) / range_above
    boost_ratio = max(0.0, min(1.0, boost_ratio))

    effective = config.base_pct + config.max_boost_pct * boost_ratio
    return min(effective, config.max_pct)


def build_sizing_series(
    confidence_series: pd.Series,
    config: Optional[ConfidenceSizingConfig] = None,
) -> pd.Series:
    """Build a position sizing series from a confidence series.

    Args:
        confidence_series: P(bull) values indexed by timestamp (0-1)
        config: Sizing configuration (uses defaults if None)

    Returns:
        Series of effective position sizes (same index)
    """
    if config is None:
        config = ConfidenceSizingConfig()
    return confidence_series.apply(lambda c: compute_effective_size(c, config))


def load_confidence_from_labels(
    labels_csv: str,
    p_bull_col: str = "p_bull",
) -> Optional[pd.Series]:
    """Load P(bull) confidence from a labels CSV if available.

    Args:
        labels_csv: Path to labels CSV
        p_bull_col: Column name for P(bull)

    Returns:
        Series indexed by timestamp with P(bull) values, or None if not available.
    """
    try:
        df = pd.read_csv(labels_csv, parse_dates=["timestamp"])
        if p_bull_col not in df.columns:
            return None
        series = df.set_index("timestamp")[p_bull_col].sort_index()
        return series.astype(float)
    except Exception:
        return None


__all__ = [
    "ConfidenceSizingConfig",
    "compute_effective_size",
    "build_sizing_series",
    "load_confidence_from_labels",
]
