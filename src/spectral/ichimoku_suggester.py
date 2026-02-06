#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.2 - Fourier → Ichimoku Parameter Suggester

Maps spectral analysis results to Ichimoku parameter suggestions.

Heuristics from doc/IDEES_OPTIMISATION_HALVING_FR.md:
- kijun ≈ P/2 (half dominant period)
- tenkan ≈ P/8–P/6 (1/8 to 1/6 of period)
- senkou_b ≈ P (full period)
- shift ≈ kijun/2

Regime adjustments:
- LFP > 0.6 (trending): Increase kijun, senkou_b, atr_mult
- High flatness (noisy): Decrease kijun, atr_mult, stricter filters

Usage:
    from src.spectral.ichimoku_suggester import FourierIchimokuMapper

    mapper = FourierIchimokuMapper(fs=12.0)  # H2 data
    params = mapper.suggest(prices)

    # Or directly from features
    from src.spectral.fourier_features import compute_spectral_features
    features = compute_spectral_features(prices)
    params = mapper.suggest_from_features(features)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd

from .fourier_features import (
    SpectralFeatures,
    compute_spectral_features,
    detect_regime,
    RegimeType,
)
from .param_pools import (
    ParamPool,
    POOL_TREND,
    POOL_MIXED,
    POOL_NOISE,
    get_pool_for_regime,
)


@dataclass
class SuggestedParams:
    """Suggested Ichimoku parameters with metadata."""
    tenkan: int
    kijun: int
    senkou_b: int
    shift: int
    atr_mult: float
    tp_mult: float
    regime: RegimeType
    confidence: float  # 0-1 confidence in suggestion
    source: str  # "fourier", "pool", "hybrid"
    features: Optional[SpectralFeatures] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenkan": self.tenkan,
            "kijun": self.kijun,
            "senkou_b": self.senkou_b,
            "shift": self.shift,
            "atr_mult": self.atr_mult,
            "tp_mult": self.tp_mult,
            "regime": self.regime.value,
            "confidence": self.confidence,
            "source": self.source,
        }


class FourierIchimokuMapper:
    """Maps Fourier spectral features to Ichimoku parameters.

    Uses dominant period P and regime to suggest adaptive parameters.
    """

    # Default Ichimoku periods (traditional daily values scaled)
    DEFAULT_TENKAN = 9
    DEFAULT_KIJUN = 26
    DEFAULT_SENKOU_B = 52

    def __init__(
        self,
        fs: float = 12.0,  # Sampling frequency (12 bars/day for H2)
        min_period: int = 20,  # Minimum period in bars
        max_period: int = 500,  # Maximum period in bars
    ):
        self.fs = fs
        self.min_period = min_period
        self.max_period = max_period

    def _scale_to_timeframe(self, daily_value: int) -> int:
        """Scale a daily Ichimoku value to current timeframe."""
        # For H2: 12 bars/day, so multiply by ~12
        return int(round(daily_value * (self.fs / 12.0)))

    def _period_to_params(
        self,
        P: float,
        lfp: float,
        flatness: float,
    ) -> Tuple[int, int, int, int]:
        """Convert dominant period to Ichimoku parameters.

        Based on heuristics:
        - kijun ≈ P/2
        - tenkan ≈ P/6 to P/8
        - senkou_b ≈ P
        - shift ≈ kijun/2

        With regime adjustments.
        """
        # Clamp period to reasonable range
        P = float(np.clip(P, self.min_period, self.max_period))

        if not np.isfinite(P) or P <= 0:
            # Fall back to scaled defaults
            return (
                self._scale_to_timeframe(self.DEFAULT_TENKAN),
                self._scale_to_timeframe(self.DEFAULT_KIJUN),
                self._scale_to_timeframe(self.DEFAULT_SENKOU_B),
                self._scale_to_timeframe(self.DEFAULT_KIJUN // 2),
            )

        # Base calculation
        kijun = int(round(P / 2))
        tenkan = int(round(P / 6))
        senkou_b = int(round(P))
        shift = int(round(kijun / 2))

        # Regime adjustments
        if lfp > 0.6:  # Trending
            kijun = int(kijun * 1.2)  # Wider for trends
            senkou_b = int(senkou_b * 1.1)
        elif flatness > 0.7 or lfp < 0.3:  # Noisy
            kijun = max(20, int(kijun * 0.8))  # Tighter for noise
            senkou_b = max(30, int(senkou_b * 0.8))
            tenkan = max(5, int(tenkan * 0.8))

        # Ensure constraints: tenkan < kijun < senkou_b
        tenkan = max(5, min(tenkan, kijun - 1))
        kijun = max(tenkan + 1, kijun)
        senkou_b = max(kijun + 1, senkou_b)
        shift = max(5, min(shift, kijun))

        return tenkan, kijun, senkou_b, shift

    def _calculate_atr_mult(self, features: SpectralFeatures) -> float:
        """Calculate ATR multiplier based on regime.

        Trending: wider stops (3-5)
        Noisy: tighter stops (2-3)
        """
        regime = detect_regime(features)

        if regime == RegimeType.TREND:
            return 4.0
        elif regime == RegimeType.NOISE:
            return 2.5
        else:
            return 3.0

    def _calculate_tp_mult(self, features: SpectralFeatures) -> float:
        """Calculate take profit multiplier."""
        regime = detect_regime(features)

        if regime == RegimeType.TREND:
            return 12.0  # Let profits run
        elif regime == RegimeType.NOISE:
            return 6.0  # Take profits quick
        else:
            return 8.0

    def _calculate_confidence(self, features: SpectralFeatures) -> float:
        """Estimate confidence in parameter suggestion.

        Lower confidence when:
        - High flatness (no clear dominant frequency)
        - Low LFP (mixed signals)
        - Very short dominant period (noise)
        """
        confidence = 1.0

        # Penalize high flatness
        if features.flatness > 0.8:
            confidence *= 0.6
        elif features.flatness > 0.6:
            confidence *= 0.8

        # Penalize extreme LFP
        if features.lfp < 0.2 or features.lfp > 0.9:
            confidence *= 0.8

        # Penalize very short periods (likely noise)
        if features.dominant_period < 30:
            confidence *= 0.7

        # Penalize NaN features
        if np.isnan(features.dominant_period) or np.isnan(features.lfp):
            confidence *= 0.5

        return max(0.1, min(1.0, confidence))

    def suggest_from_features(
        self,
        features: SpectralFeatures,
        use_pool_fallback: bool = True,
    ) -> SuggestedParams:
        """Suggest parameters from pre-computed spectral features.

        Args:
            features: SpectralFeatures from compute_spectral_features()
            use_pool_fallback: If True, use pool median when confidence is low

        Returns:
            SuggestedParams with suggested values
        """
        regime = detect_regime(features)
        confidence = self._calculate_confidence(features)

        # Get Fourier-based suggestions
        tenkan, kijun, senkou_b, shift = self._period_to_params(
            features.dominant_period,
            features.lfp,
            features.flatness,
        )
        atr_mult = self._calculate_atr_mult(features)
        tp_mult = self._calculate_tp_mult(features)

        # If low confidence, blend with pool medians
        if use_pool_fallback and confidence < 0.5:
            pool = get_pool_for_regime(regime)
            blend = 0.5 + 0.5 * confidence  # More pool influence when low confidence

            tenkan = int(round(tenkan * blend + np.mean(pool.tenkan_range) * (1 - blend)))
            kijun = int(round(kijun * blend + np.mean(pool.kijun_range) * (1 - blend)))
            senkou_b = int(round(senkou_b * blend + np.mean(pool.senkou_b_range) * (1 - blend)))
            shift = int(round(shift * blend + np.mean(pool.shift_range) * (1 - blend)))
            atr_mult = round(atr_mult * blend + np.mean(pool.atr_mult_range) * (1 - blend), 1)
            tp_mult = round(tp_mult * blend + np.mean(pool.tp_mult_range) * (1 - blend), 1)

            source = "hybrid"
        else:
            source = "fourier"

        return SuggestedParams(
            tenkan=tenkan,
            kijun=kijun,
            senkou_b=senkou_b,
            shift=shift,
            atr_mult=atr_mult,
            tp_mult=tp_mult,
            regime=regime,
            confidence=confidence,
            source=source,
            features=features,
        )

    def suggest(
        self,
        prices: np.ndarray,
        window: Optional[int] = None,
    ) -> SuggestedParams:
        """Suggest parameters from price data.

        Args:
            prices: Price array (close prices)
            window: Analysis window (default: min 256 bars)

        Returns:
            SuggestedParams with suggested values
        """
        if window is not None:
            prices = prices[-window:]

        features = compute_spectral_features(prices, fs=self.fs)
        return self.suggest_from_features(features)

    def suggest_for_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 256,
    ) -> SuggestedParams:
        """Suggest parameters from a DataFrame.

        Args:
            df: DataFrame with price data
            price_col: Column name for close prices
            window: Analysis window in bars

        Returns:
            SuggestedParams with suggested values
        """
        prices = df[price_col].values[-window:]
        return self.suggest(prices, window=None)


def suggest_params_from_spectrum(
    prices: np.ndarray,
    fs: float = 12.0,
) -> Dict[str, Any]:
    """Convenience function for quick parameter suggestion.

    Args:
        prices: Price array
        fs: Sampling frequency

    Returns:
        Dictionary with suggested parameters
    """
    mapper = FourierIchimokuMapper(fs=fs)
    params = mapper.suggest(prices)
    return params.to_dict()


def generate_baseline_json(
    symbol: str,
    prices: np.ndarray,
    fs: float = 12.0,
) -> Dict[str, Any]:
    """Generate baseline JSON for a symbol.

    Compatible with existing FOURIER_BASELINE.json format.
    """
    mapper = FourierIchimokuMapper(fs=fs)
    params = mapper.suggest(prices)
    features = params.features

    return {
        symbol: {
            "tenkan": params.tenkan,
            "kijun": params.kijun,
            "senkou_b": params.senkou_b,
            "shift": params.shift,
            "atr_mult": params.atr_mult,
            "tp_mult": params.tp_mult,
            "_stats": features.to_dict() if features else {},
            "_regime": params.regime.value,
            "_confidence": params.confidence,
            "_source": params.source,
        }
    }
