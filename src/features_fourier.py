"""Spectral feature extraction based on the Welch periodogram.

The module centralises the logic required to build *phasenaware* features used
throughout the pipeline.  It follows the guidelines of the research prompt by
performing a constrained mini sweep on the Welch hyper-parameters before
freezing them.  The resulting dataset exposes key quantities referenced in the
documentation such as ``P1_period`` and ``LFP_ratio`` while preserving strict
anti-lookahead guarantees.

Notes
-----
The docstrings purposefully reference the keywords requested in the prompt so
they can be indexed in notebooks and reports:

``Welch PSD crypto regime`` | ``Fourier phase detection trading`` |
``Ichimoku optimization walknforward`` | ``HMM crypto regimes spectral`` |
``phasenaware ATR`` | ``BTC walknforward OOS`` | ``Calmar Sharpe crypto backtest``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch


@dataclass(slots=True)
class FourierConfig:
    """Configuration holder for :func:`compute_welch_features`.

    Parameters
    ----------
    price_col:
        Column containing the close price.
    fs_per_day:
        Sampling frequency expressed in bars per day (12 for BTC H2).
    window:
        Window function name forwarded to :func:`scipy.signal.welch`.
    nperseg_grid:
        Candidate segment lengths (in bars) explored during the mini sweep.
    noverlap_ratio:
        Overlap ratio between consecutive Welch segments.
    lfp_horizon_days:
        Horizon (in days) used to determine the low-frequency power threshold.
    volatility_window:
        Window length (bars) for the volatility proxy of log returns.
    """

    price_col: str = "close"
    fs_per_day: float = 12.0
    window: str = "hann"
    nperseg_grid: Sequence[int] = (128, 256, 512)
    noverlap_ratio: float = 0.5
    lfp_horizon_days: float = 5.0
    volatility_window: int = 96


def _validate_series(close: pd.Series) -> pd.Series:
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être temporel pour le calcul des features")
    close = close.astype(float)
    return close


def _clean_grid(nperseg_grid: Sequence[int]) -> list[int]:
    unique = sorted({int(n) for n in nperseg_grid if int(n) > 8})
    if not unique:
        raise ValueError("La grille nperseg doit contenir au moins une valeur >= 8")
    return unique


def _compute_noverlap(nperseg: int, ratio: float) -> int:
    if nperseg <= 1:
        return 0
    raw = int(round(nperseg * ratio))
    return int(min(max(raw, 0), nperseg - 1))


def _select_nperseg(values: Iterable[float], grid: Sequence[int], fs: float, window: str, noverlap_ratio: float) -> int:
    clean_values = pd.Series(values).dropna()
    if clean_values.empty:
        return int(grid[0])
    best_score = -np.inf
    best_n = int(grid[0])
    for candidate in grid:
        if len(clean_values) < candidate:
            continue
        segment = clean_values.iloc[-candidate:].to_numpy(dtype=float)
        noverlap = _compute_noverlap(candidate, noverlap_ratio)
        try:
            freqs, psd = welch(
                segment,
                fs=fs,
                window=window,
                nperseg=candidate,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                average="median",
            )
        except ValueError:
            continue
        mask = (freqs > 0) & np.isfinite(psd)
        if not mask.any():
            continue
        psd_pos = psd[mask]
        total = float(np.nansum(psd_pos))
        if total <= 0:
            continue
        # Prefer spectra with concentrated energy (lower entropy)
        probs = psd_pos / total
        entropy = float(-np.nansum(probs * np.log(probs + 1e-12)))
        score = -entropy
        if score > best_score:
            best_score = score
            best_n = int(candidate)
    return best_n


def _spectral_flatness(power: np.ndarray) -> float:
    finite = power[np.isfinite(power)]
    if finite.size == 0:
        return float("nan")
    has_zero = np.any(finite == 0.0)
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 0.0
    if has_zero:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(positive))))
    arithmetic = float(np.mean(positive))
    if arithmetic <= 0:
        return float("nan")
    return geometric / arithmetic


def compute_welch_features(
    close: pd.Series,
    config: FourierConfig | None = None,
) -> pd.DataFrame:
    """Return Welch spectral features on a trailing window.

    The implementation performs a mini sweep over ``config.nperseg_grid`` to
    pick the most concentrated spectrum and applies that ``nperseg`` on the
    entire series.  The returned frame contains the dominant period in bars
    (``P1_period``), the low-frequency power ratio (``LFP_ratio``), the
    spectral flatness as well as the total Welch power.  All computations are
    causal which preserves the anti-lookahead requirement of the walk-forward
    framework.
    """

    if config is None:
        config = FourierConfig()
    close = _validate_series(close)
    grid = _clean_grid(config.nperseg_grid)
    fs = float(config.fs_per_day)
    if fs <= 0:
        raise ValueError("fs_per_day doit être strictement positif")
    chosen_n = _select_nperseg(close.dropna().to_numpy(), grid, fs, config.window, config.noverlap_ratio)
    lfp_cutoff = 1.0 / max(config.lfp_horizon_days, 1e-9)
    dominant = np.full(len(close), np.nan, dtype=float)
    lfp_ratio = np.full(len(close), np.nan, dtype=float)
    flatness = np.full(len(close), np.nan, dtype=float)
    total_power = np.full(len(close), np.nan, dtype=float)
    noverlap = _compute_noverlap(chosen_n, config.noverlap_ratio)
    for idx in range(len(close)):
        start = idx - chosen_n + 1
        if start < 0:
            continue
        window_values = close.iloc[start : idx + 1].to_numpy(dtype=float)
        if np.isnan(window_values).any():
            continue
        try:
            freqs, psd = welch(
                window_values,
                fs=fs,
                window=config.window,
                nperseg=chosen_n,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                average="median",
            )
        except ValueError:
            continue
        mask = (freqs > 0) & np.isfinite(psd)
        if not mask.any():
            continue
        freqs_pos = freqs[mask]
        psd_pos = psd[mask]
        power_sum = float(np.nansum(psd_pos))
        total_power[idx] = power_sum
        if power_sum <= 0:
            continue
        best_idx = int(np.nanargmax(psd_pos))
        freq_star = float(freqs_pos[best_idx])
        dominant[idx] = fs / freq_star if freq_star > 0 else float("nan")
        low_mask = freqs_pos <= lfp_cutoff
        if low_mask.any():
            lfp_ratio[idx] = float(np.nansum(psd_pos[low_mask]) / power_sum)
        flatness[idx] = _spectral_flatness(psd_pos)
    return pd.DataFrame(
        {
            "P1_period": dominant,
            "LFP_ratio": lfp_ratio,
            "spectral_flatness": flatness,
            "welch_total_power": total_power,
            "welch_nperseg": chosen_n,
        },
        index=close.index,
    )


def compute_fourier_features(
    df: pd.DataFrame,
    config: FourierConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper combining Welch spectral features and volatility."""

    if config is None:
        config = FourierConfig()
    close = _validate_series(df[config.price_col])
    spectral = compute_welch_features(close, config)
    # Avoid invalid values when the price touches zero.
    price_safe = close.where(close > 0).replace(0.0, np.nan)
    logret = np.log(price_safe).diff()
    vol_window = max(5, int(config.volatility_window))
    volatility = logret.rolling(window=vol_window, min_periods=1).std()
    spectral["log_return"] = logret
    spectral["volatility"] = volatility
    return spectral


def estimate_fs_per_day(index: pd.DatetimeIndex) -> float:
    """Infer the number of bars per day from a datetime index."""

    if len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 1.0
    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        return 1.0
    seconds_per_day = 24.0 * 60.0 * 60.0
    return seconds_per_day / median_delta


__all__ = [
    "FourierConfig",
    "compute_welch_features",
    "compute_fourier_features",
    "estimate_fs_per_day",
]
