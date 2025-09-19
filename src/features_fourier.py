"""Fourier-based feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FourierConfig:
    """Configuration holder used during feature extraction."""

    window: int = 128
    lfp_cutoff: float = 0.1
    price_col: str = "close"


def _periodogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the frequencies and power spectral density for ``values``."""

    x = np.asarray(values, dtype=float)
    if np.all(np.isnan(x)):
        return np.array([]), np.array([])
    x = x - np.nanmean(x)
    if np.allclose(x, 0.0):
        return np.array([]), np.array([])
    window = np.hanning(len(x))
    x = np.where(np.isnan(x), 0.0, x) * window
    fft = np.fft.rfft(x)
    psd = (np.abs(fft) ** 2) / max(np.sum(window**2), 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    return freqs, psd


def _dominant_period(segment: np.ndarray) -> float:
    freqs, psd = _periodogram(segment)
    if len(freqs) <= 1:
        return float("nan")
    idx = int(np.argmax(psd[1:]) + 1)  # skip DC component
    f_star = float(freqs[idx])
    return float("nan") if f_star <= 0 else 1.0 / f_star


def _low_frequency_ratio(segment: np.ndarray, cutoff: float) -> float:
    freqs, psd = _periodogram(segment)
    if len(freqs) == 0:
        return float("nan")
    mask = freqs <= cutoff
    total = float(np.nansum(psd))
    if total <= 0:
        return float("nan")
    return float(np.nansum(psd[mask]) / total)


def compute_fourier_features(
    df: pd.DataFrame,
    config: FourierConfig | None = None,
) -> pd.DataFrame:
    """Compute Fourier features for ``df``.

    The function returns a :class:`~pandas.DataFrame` containing, for each
    timestamp, the dominant period estimate, the low-frequency power ratio, the
    rolling volatility of log-returns and the raw log-return itself.
    """

    if config is None:
        config = FourierConfig()
    price = df[config.price_col].astype(float)
    if not isinstance(price.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être temporel pour le calcul des features")

    window = int(config.window)
    if window <= 4:
        raise ValueError("La fenêtre de Fourier doit être supérieure à 4")

    dominant = np.full(len(price), np.nan, dtype=float)
    lfp = np.full(len(price), np.nan, dtype=float)
    values = price.values
    for idx in range(window - 1, len(price)):
        segment = values[idx - window + 1 : idx + 1]
        dominant[idx] = _dominant_period(segment)
        lfp[idx] = _low_frequency_ratio(segment, config.lfp_cutoff)

    logret = np.log(price).diff()
    vol = logret.rolling(window=min(window, max(5, window // 2))).std()
    features = pd.DataFrame(
        {
            "dominant_period": dominant,
            "lfp_ratio": lfp,
            "log_return": logret,
            "volatility": vol,
        },
        index=price.index,
    )
    return features


__all__ = ["FourierConfig", "compute_fourier_features"]
