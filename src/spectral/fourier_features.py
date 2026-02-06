#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.1 - Enhanced Fourier features for regime detection and parameter suggestion.

Features extraites:
- PSD (Power Spectral Density) via Welch
- Dominant Period P
- LFP (Low Frequency Power ratio)
- Spectral Flatness (entropy proxy)
- Spectral Centroid
- Spectral Slope (1/f^alpha estimation)
- Band powers (low/mid/high)

Usage:
    from src.spectral.fourier_features import compute_spectral_features, detect_regime

    features = compute_spectral_features(close_prices, fs=12.0)  # H2 = 12 bars/day
    regime = detect_regime(features)  # Returns RegimeType.TREND, MIXED, or NOISE
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


class RegimeType(Enum):
    """Market regime types based on spectral analysis."""
    TREND = "trend"      # Low frequency dominant, trending market
    MIXED = "mixed"      # Mixed frequencies, transition regime
    NOISE = "noise"      # High frequency dominant, choppy/noisy market
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class SpectralFeatures:
    """Container for all spectral features."""
    dominant_period: float      # P = 1/f* in bars
    lfp: float                  # Low Frequency Power ratio [0,1]
    flatness: float             # Spectral flatness (GM/AM) [0,1]
    centroid: float             # Spectral centroid in Hz
    slope: float                # Spectral slope (alpha in 1/f^alpha)
    band_low: float             # Power in low freq band [0, f_low]
    band_mid: float             # Power in mid freq band [f_low, f_high]
    band_high: float            # Power in high freq band [f_high, fs/2]
    entropy: float              # Spectral entropy
    peak_freq: float            # Peak frequency (excluding DC)

    def to_dict(self) -> Dict[str, float]:
        return {
            "dominant_period": self.dominant_period,
            "lfp": self.lfp,
            "flatness": self.flatness,
            "centroid": self.centroid,
            "slope": self.slope,
            "band_low": self.band_low,
            "band_mid": self.band_mid,
            "band_high": self.band_high,
            "entropy": self.entropy,
            "peak_freq": self.peak_freq,
        }


def compute_welch_psd(
    data: np.ndarray,
    fs: float = 12.0,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        data: Time series data (close prices or returns)
        fs: Sampling frequency (bars per day, default 12 for H2)
        nperseg: Segment length for Welch (auto if None)

    Returns:
        freqs: Frequency bins
        psd: Power spectral density
    """
    try:
        from scipy.signal import welch
        n = len(data)
        if nperseg is None:
            nperseg = min(1024, max(64, n // 4))
        freqs, psd = welch(data, fs=fs, nperseg=nperseg)
        return freqs, psd
    except ImportError:
        # Fallback: windowed periodogram
        n = len(data)
        if n <= 1:
            return np.array([0.0]), np.array([0.0])
        window = np.hanning(n)
        data_centered = data - np.nanmean(data)
        fft = np.fft.rfft(window * data_centered)
        psd = (np.abs(fft) ** 2) / (np.sum(window ** 2) * fs)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, psd


def _dominant_period(freqs: np.ndarray, psd: np.ndarray, min_idx: int = 1) -> float:
    """Find dominant period P = 1/f* (ignoring DC)."""
    if len(freqs) <= min_idx or len(psd) <= min_idx:
        return float("nan")
    idx = int(np.nanargmax(psd[min_idx:]) + min_idx)
    f_star = max(1e-12, float(freqs[idx]))
    return 1.0 / f_star


def _lfp_ratio(freqs: np.ndarray, psd: np.ndarray, f0: float) -> float:
    """Low Frequency Power ratio: sum(PSD[f<f0]) / sum(PSD)."""
    total = float(np.nansum(psd))
    if total <= 0:
        return float("nan")
    mask = freqs < f0
    low = float(np.nansum(psd[mask]))
    return low / total


def _spectral_flatness(psd: np.ndarray) -> float:
    """Spectral flatness = GM/AM (0=peaky, 1=flat/white noise)."""
    psd = np.asarray(psd, dtype=float)
    if psd.size == 0:
        return float("nan")
    positives = psd[psd > 0]
    if positives.size == 0:
        return 0.0
    gmean = float(np.exp(np.mean(np.log(positives))))
    amean = float(np.mean(positives))
    return gmean / amean if amean > 0 else float("nan")


def _spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Spectral centroid = weighted mean frequency."""
    total = float(np.nansum(psd))
    if total <= 0:
        return float("nan")
    return float(np.nansum(freqs * psd) / total)


def _spectral_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Estimate spectral slope alpha in 1/f^alpha model via log-log regression."""
    # Skip DC and very low frequencies
    mask = freqs > 0.01
    if np.sum(mask) < 3:
        return float("nan")
    log_f = np.log(freqs[mask])
    log_psd = np.log(psd[mask] + 1e-12)
    try:
        slope, _ = np.polyfit(log_f, log_psd, 1)
        return -float(slope)  # Negate because PSD ~ 1/f^alpha means log(PSD) ~ -alpha*log(f)
    except Exception:
        return float("nan")


def _spectral_entropy(psd: np.ndarray) -> float:
    """Normalized spectral entropy."""
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    if len(psd_norm) == 0:
        return 0.0
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    max_entropy = np.log2(len(psd_norm))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float = 0.1,
    f_high: float = 1.0,
) -> Tuple[float, float, float]:
    """Compute power in low/mid/high frequency bands."""
    total = float(np.nansum(psd)) + 1e-12

    mask_low = freqs < f_low
    mask_mid = (freqs >= f_low) & (freqs < f_high)
    mask_high = freqs >= f_high

    band_low = float(np.nansum(psd[mask_low])) / total
    band_mid = float(np.nansum(psd[mask_mid])) / total
    band_high = float(np.nansum(psd[mask_high])) / total

    return band_low, band_mid, band_high


def compute_spectral_features(
    data: np.ndarray,
    fs: float = 12.0,
    f0_lfp: Optional[float] = None,
) -> SpectralFeatures:
    """Compute all spectral features from price/return data.

    Args:
        data: Price series or returns (min 64 points recommended)
        fs: Sampling frequency (12 for H2 = 12 bars/day)
        f0_lfp: Frequency threshold for LFP (default: 1/(5*fs/12) for 5-day cycles)

    Returns:
        SpectralFeatures dataclass with all computed features
    """
    if len(data) < 16:
        return SpectralFeatures(
            dominant_period=float("nan"),
            lfp=float("nan"),
            flatness=float("nan"),
            centroid=float("nan"),
            slope=float("nan"),
            band_low=float("nan"),
            band_mid=float("nan"),
            band_high=float("nan"),
            entropy=float("nan"),
            peak_freq=float("nan"),
        )

    # Use log returns for stationarity
    if np.all(data > 0):
        returns = np.diff(np.log(data))
    else:
        returns = np.diff(data)

    if len(returns) < 16:
        returns = data - np.mean(data)

    # Compute PSD
    freqs, psd = compute_welch_psd(returns, fs=fs)

    # Default f0 for LFP: cycles > 5 days
    if f0_lfp is None:
        f0_lfp = 1.0 / (5.0 * fs / 12.0)  # 5 days in bars

    # Band power thresholds (in cycles per day)
    f_low = 0.1   # < 10 days period
    f_high = 1.0  # > 1 day period

    return SpectralFeatures(
        dominant_period=_dominant_period(freqs, psd),
        lfp=_lfp_ratio(freqs, psd, f0_lfp),
        flatness=_spectral_flatness(psd),
        centroid=_spectral_centroid(freqs, psd),
        slope=_spectral_slope(freqs, psd),
        band_low=_band_powers(freqs, psd, f_low, f_high)[0],
        band_mid=_band_powers(freqs, psd, f_low, f_high)[1],
        band_high=_band_powers(freqs, psd, f_low, f_high)[2],
        entropy=_spectral_entropy(psd),
        peak_freq=freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else float("nan"),
    )


def detect_regime(
    features: SpectralFeatures,
    lfp_trend_threshold: float = 0.6,
    lfp_noise_threshold: float = 0.3,
    flatness_noise_threshold: float = 0.7,
) -> RegimeType:
    """Classify market regime based on spectral features.

    Rules:
    - TREND: LFP > 0.6 (low frequencies dominate = trending)
    - NOISE: LFP < 0.3 OR flatness > 0.7 (high freq or flat spectrum = noisy)
    - MIXED: otherwise (transition/mixed regime)

    Args:
        features: SpectralFeatures from compute_spectral_features()
        lfp_trend_threshold: LFP above this = TREND
        lfp_noise_threshold: LFP below this = NOISE
        flatness_noise_threshold: Flatness above this = NOISE

    Returns:
        RegimeType enum value
    """
    if np.isnan(features.lfp) or np.isnan(features.flatness):
        return RegimeType.UNKNOWN

    if features.lfp >= lfp_trend_threshold:
        return RegimeType.TREND

    if features.lfp <= lfp_noise_threshold or features.flatness >= flatness_noise_threshold:
        return RegimeType.NOISE

    return RegimeType.MIXED


def compute_rolling_spectral(
    prices: pd.Series,
    window: int = 256,
    step: int = 12,
    fs: float = 12.0,
) -> pd.DataFrame:
    """Compute rolling spectral features over time.

    Args:
        prices: Price series with datetime index
        window: Rolling window size in bars (default 256 = ~21 days for H2)
        step: Step size between windows (default 12 = 1 day for H2)
        fs: Sampling frequency

    Returns:
        DataFrame with spectral features at each step
    """
    results: List[Dict] = []

    for i in range(window, len(prices), step):
        chunk = prices.iloc[i-window:i].values
        features = compute_spectral_features(chunk, fs=fs)
        regime = detect_regime(features)

        results.append({
            "timestamp": prices.index[i-1],
            "regime": regime.value,
            **features.to_dict(),
        })

    return pd.DataFrame(results).set_index("timestamp")


# Aliases for backward compatibility with fourier_core
def dominant_period(freqs, psd, min_idx=1):
    return _dominant_period(freqs, psd, min_idx)

def low_freq_power_ratio(freqs, psd, f0):
    return _lfp_ratio(freqs, psd, f0)

def spectral_flatness(psd):
    return _spectral_flatness(psd)
