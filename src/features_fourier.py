"""Welch-based spectral feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _WelchConfig:
    nperseg: int
    noverlap: int


def _normalize_ratios(noverlap_ratio: float | Sequence[float]) -> list[float]:
    if isinstance(noverlap_ratio, (float, int)):
        ratios = [float(noverlap_ratio)]
    else:
        ratios = [float(r) for r in noverlap_ratio]
    cleaned = [r for r in ratios if 0.0 <= r < 1.0]
    if not cleaned:
        return [0.5]
    return sorted(dict.fromkeys(cleaned))


def _fallback_window(name: str, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=float)
    key = name.lower()
    if key in {"hann", "hanning"}:
        return np.hanning(length)
    if key in {"boxcar", "rect", "rectangular", "ones"}:
        return np.ones(length, dtype=float)
    if key in {"bartlett", "triang"}:
        return np.bartlett(length)
    return np.hanning(length)


def _periodogram_psd(values: np.ndarray, *, fs: float, window: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return np.asarray([]), np.asarray([])
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.asarray([]), np.asarray([])
    centered = values.copy()
    mean = np.nanmean(centered[valid])
    centered[valid] = centered[valid] - mean
    centered[~valid] = 0.0
    win = _fallback_window(window, centered.size)
    if win.size != centered.size:
        win = np.resize(win, centered.size)
    scale = np.sum(win ** 2)
    if scale <= 0:
        return np.asarray([]), np.asarray([])
    fft = np.fft.rfft(centered * win)
    psd = (np.abs(fft) ** 2) / (fs * scale)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    return freqs, psd


def _compute_psd(
    values: Iterable[float],
    *,
    fs: float,
    window: str,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return np.asarray([]), np.asarray([])
    valid = np.isfinite(array)
    if not np.any(valid) or array.size < max(8, nperseg):
        return np.asarray([]), np.asarray([])
    array = array.copy()
    mean = float(np.nanmean(array[valid])) if np.any(valid) else 0.0
    array[valid] = array[valid] - mean
    array[~valid] = 0.0
    try:
        from scipy.signal import welch  # type: ignore

        nperseg = min(int(nperseg), array.size)
        noverlap = max(0, min(int(noverlap), nperseg - 1))
        freqs, psd = welch(
            array,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )
        return freqs, psd
    except Exception:
        return _periodogram_psd(array, fs=fs, window=window)


def _extract_metrics(freqs: np.ndarray, psd: np.ndarray, *, fs_per_day: float) -> dict[str, float]:
    if freqs.size == 0 or psd.size == 0:
        return {
            "P1_period": float("nan"),
            "P1_period_days": float("nan"),
            "P1_frequency": float("nan"),
            "LFP_ratio": float("nan"),
            "spectral_flatness": float("nan"),
            "peak_power": float("nan"),
            "total_power": float("nan"),
            "low_freq_power": float("nan"),
            "high_freq_power": float("nan"),
        }
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(psd)
    if not np.any(mask):
        return {
            "P1_period": float("nan"),
            "P1_period_days": float("nan"),
            "P1_frequency": float("nan"),
            "LFP_ratio": float("nan"),
            "spectral_flatness": float("nan"),
            "peak_power": float("nan"),
            "total_power": float("nan"),
            "low_freq_power": float("nan"),
            "high_freq_power": float("nan"),
        }
    freqs = freqs[mask]
    psd = psd[mask]
    positive = freqs > 0
    if not np.any(positive):
        return {
            "P1_period": float("nan"),
            "P1_period_days": float("nan"),
            "P1_frequency": float("nan"),
            "LFP_ratio": float("nan"),
            "spectral_flatness": float("nan"),
            "peak_power": float("nan"),
            "total_power": float(np.nanmean(psd) * len(psd)),
            "low_freq_power": float("nan"),
            "high_freq_power": float("nan"),
        }
    freqs = freqs[positive]
    psd = psd[positive]
    idx = int(np.nanargmax(psd))
    peak_freq = float(freqs[idx]) if idx < freqs.size else float("nan")
    peak_power = float(psd[idx]) if idx < psd.size else float("nan")
    period_days = (1.0 / peak_freq) if peak_freq > 0 else float("nan")
    period_bars = period_days * float(fs_per_day) if np.isfinite(period_days) else float("nan")
    total_power = float(np.nansum(psd))
    low_cut = 1.0 / 5.0  # cycles per day (period >= 5 days)
    low_mask = freqs <= low_cut
    low_power = float(np.nansum(psd[low_mask])) if np.any(low_mask) else float("nan")
    high_power = total_power - low_power if np.isfinite(total_power) and np.isfinite(low_power) else float("nan")
    psd_clipped = np.clip(psd, 1e-12, None)
    spectral_flatness = float(np.exp(np.nanmean(np.log(psd_clipped))) / np.nanmean(psd_clipped)) if np.nanmean(psd_clipped) > 0 else float("nan")
    lfp_ratio = (low_power / total_power) if total_power > 0 and np.isfinite(low_power) else float("nan")
    return {
        "P1_period": period_bars,
        "P1_period_days": period_days,
        "P1_frequency": peak_freq,
        "LFP_ratio": lfp_ratio,
        "spectral_flatness": spectral_flatness,
        "peak_power": peak_power,
        "total_power": total_power,
        "low_freq_power": low_power,
        "high_freq_power": high_power,
    }


def _relative_gap(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return float("inf")
    return float(abs(np.log(a / b)))


def _score_config(
    window_values: np.ndarray,
    *,
    fs: float,
    window: str,
    nperseg: int,
    noverlap: int,
) -> float:
    if window_values.size < nperseg * 2:
        return float("inf")
    train = window_values[:-nperseg]
    test = window_values[-nperseg:]
    freqs_train, psd_train = _compute_psd(train, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    freqs_test, psd_test = _compute_psd(test, fs=fs, window=window, nperseg=min(nperseg, test.size), noverlap=min(noverlap, max(0, test.size - 1)))
    metrics_train = _extract_metrics(freqs_train, psd_train, fs_per_day=fs)
    metrics_test = _extract_metrics(freqs_test, psd_test, fs_per_day=fs)
    if not np.isfinite(metrics_train["P1_period"]) or not np.isfinite(metrics_test["P1_period"]):
        return float("inf")
    if not np.isfinite(metrics_train["LFP_ratio"]) or not np.isfinite(metrics_test["LFP_ratio"]):
        return float("inf")
    gap_period = _relative_gap(metrics_train["P1_period"], metrics_test["P1_period"])
    if not np.isfinite(gap_period):
        return float("inf")
    gap_lfp = abs(metrics_train["LFP_ratio"] - metrics_test["LFP_ratio"])
    gap_flat = abs(metrics_train["spectral_flatness"] - metrics_test["spectral_flatness"]) if (
        np.isfinite(metrics_train["spectral_flatness"]) and np.isfinite(metrics_test["spectral_flatness"])
    ) else 0.0
    stability = gap_period + gap_lfp + 0.5 * gap_flat
    penalty = 1.0 / float(nperseg)
    return stability + penalty


def _select_config(
    window_values: np.ndarray,
    *,
    fs: float,
    window: str,
    nperseg_grid: Sequence[int],
    noverlap_ratios: Sequence[float],
) -> _WelchConfig | None:
    best_score = float("inf")
    best_config: _WelchConfig | None = None
    for nperseg in sorted({int(v) for v in nperseg_grid if int(v) > 8}):
        if window_values.size < nperseg * 2:
            continue
        for ratio in noverlap_ratios:
            noverlap = max(0, min(int(round(ratio * nperseg)), nperseg - 1))
            score = _score_config(window_values, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_config = _WelchConfig(nperseg=nperseg, noverlap=noverlap)
    return best_config


def compute_welch_features(
    close: pd.Series,
    *,
    fs_per_day: float = 12.0,
    window: str = "hann",
    nperseg_grid: Sequence[int] | None = None,
    noverlap_ratio: float | Sequence[float] = (0.5, 0.75),
) -> pd.DataFrame:
    """Compute Welch-based rolling features (P1_period, LFP_ratio, and extras).

    The function scans trailing windows (no lookahead; final windows are never
    centred) and performs a mini sweep OOS across the provided ``nperseg`` grid
    and ``noverlap`` ratios. Once the most stable configuration is identified it
    is locked-in for the remainder of the series. ``scipy.signal.welch`` is used
    when available; otherwise a Hann periodogram fallback maintains continuity.

    Parameters
    ----------
    close : pd.Series
        Price series ordered chronologically.
    fs_per_day : float, default 12.0
        Sampling frequency expressed in bars per day (2h bars → 12).
    window : str, default "hann"
        Window applied inside Welch / the fallback periodogram.
    nperseg_grid : sequence of int, optional
        Candidate segment lengths for the mini sweep OOS. Defaults to
        ``[128, 256, 512]``.
    noverlap_ratio : float or sequence of float, default (0.5, 0.75)
        Ratio(s) of overlap relative to ``nperseg`` to test during the sweep.

    Returns
    -------
    pd.DataFrame
        Rolling features aligned to ``close.index`` with the following columns:
        ``P1_period`` (bars), ``P1_period_days``, ``P1_frequency`` (cycles/day),
        ``LFP_ratio``, ``spectral_flatness``, ``peak_power``, ``total_power``,
        ``low_freq_power``, ``high_freq_power``, ``selected_nperseg`` and
        ``selected_noverlap``.
    """

    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.empty:
        return pd.DataFrame(
            columns=[
                "P1_period",
                "P1_period_days",
                "P1_frequency",
                "LFP_ratio",
                "spectral_flatness",
                "peak_power",
                "total_power",
                "low_freq_power",
                "high_freq_power",
                "selected_nperseg",
                "selected_noverlap",
            ],
            index=close.index,
        )
    grid = list(nperseg_grid) if nperseg_grid is not None else [128, 256, 512]
    if not grid:
        raise ValueError("nperseg_grid must contain at least one candidate")
    ratios = _normalize_ratios(noverlap_ratio)
    fs = float(fs_per_day)
    max_nperseg = max(int(v) for v in grid if int(v) > 0)
    window_multiple = 4
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)

    columns = {
        "P1_period": np.full(close.shape, np.nan, dtype=float),
        "P1_period_days": np.full(close.shape, np.nan, dtype=float),
        "P1_frequency": np.full(close.shape, np.nan, dtype=float),
        "LFP_ratio": np.full(close.shape, np.nan, dtype=float),
        "spectral_flatness": np.full(close.shape, np.nan, dtype=float),
        "peak_power": np.full(close.shape, np.nan, dtype=float),
        "total_power": np.full(close.shape, np.nan, dtype=float),
        "low_freq_power": np.full(close.shape, np.nan, dtype=float),
        "high_freq_power": np.full(close.shape, np.nan, dtype=float),
        "selected_nperseg": np.full(close.shape, np.nan, dtype=float),
        "selected_noverlap": np.full(close.shape, np.nan, dtype=float),
    }

    selected: _WelchConfig | None = None
    max_window = min(max_nperseg * window_multiple, values.size)
    for idx in range(values.size):
        end = idx + 1
        window_len = min(max_window, end)
        if end >= max_nperseg:
            window_len = max(window_len, max_nperseg)
        start = max(0, end - window_len)
        segment = values[start:end]
        if not np.any(np.isfinite(segment)):
            continue
        if selected is None:
            config = _select_config(segment, fs=fs, window=window, nperseg_grid=grid, noverlap_ratios=ratios)
            if config is None:
                continue
            selected = config
        if selected.nperseg > segment.size:
            continue
        freqs, psd = _compute_psd(segment, fs=fs, window=window, nperseg=selected.nperseg, noverlap=selected.noverlap)
        metrics = _extract_metrics(freqs, psd, fs_per_day=fs)
        for key in ("P1_period", "P1_period_days", "P1_frequency", "LFP_ratio", "spectral_flatness", "peak_power", "total_power", "low_freq_power", "high_freq_power"):
            columns[key][idx] = metrics[key]
        if selected is not None:
            columns["selected_nperseg"][idx] = float(selected.nperseg)
            columns["selected_noverlap"][idx] = float(selected.noverlap)

    return pd.DataFrame(columns, index=close.index)


__all__ = ["compute_welch_features"]

