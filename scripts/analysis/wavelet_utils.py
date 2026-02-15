#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import scipy.signal as spsig  # type: ignore
except Exception:  # pragma: no cover
    spsig = None

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None


@dataclass
class STFTResult:
    times: np.ndarray
    freqs: np.ndarray
    power: np.ndarray  # shape: (freqs, times)


@dataclass
class CWTResult:
    times: np.ndarray
    scales: np.ndarray
    power: np.ndarray  # shape: (scales, times)


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int, window: str = 'hann') -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if frame_length <= 0:
        raise ValueError('frame_length must be > 0')
    if hop_length <= 0:
        raise ValueError('hop_length must be > 0')
    num_frames = 1 + max(0, (len(x) - frame_length) // hop_length)
    shape = (num_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    if window == 'hann':
        w = np.hanning(frame_length)
    elif window == 'hamming':
        w = np.hamming(frame_length)
    else:
        w = np.ones(frame_length)
    return frames * w


def compute_stft(x: np.ndarray, fs: float, nperseg: int = 256, noverlap: Optional[int] = None) -> STFTResult:
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = int(0.5 * nperseg)
    hop = max(1, nperseg - noverlap)
    if spsig is not None:
        freqs, times, Zxx = spsig.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann', boundary=None)
        power = np.abs(Zxx) ** 2
        return STFTResult(times=times, freqs=freqs, power=power)
    # Fallback numpy STFT
    frames = _frame_signal(x, frame_length=nperseg, hop_length=hop, window='hann')
    Z = np.fft.rfft(frames, axis=1)
    power = (np.abs(Z) ** 2).T  # (freqs, times)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    times = (np.arange(frames.shape[0]) * hop + nperseg / 2) / fs
    return STFTResult(times=times, freqs=freqs, power=power)


def compute_cwt(x: np.ndarray, fs: float, wavelet: str = 'morl', num_scales: int = 64, min_period_s: float = 60.0, max_period_s: float = 60.0 * 24.0 * 90.0) -> Optional[CWTResult]:
    x = np.asarray(x, dtype=float)
    if pywt is None:
        return None
    dt = 1.0 / fs
    # Map periods to scales for Morlet: scale ~ period / (dt * f_c)
    # Use center frequency of Morlet ~ 0.8125 (approx for complex Morlet in pywt)
    f_c = 0.8125
    min_scale = (min_period_s / dt) / (2.0 * np.pi)  # rough mapping
    max_scale = (max_period_s / dt) / (2.0 * np.pi)
    scales = np.geomspace(min_scale, max_scale, num_scales)
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    power = (np.abs(coeffs) ** 2)
    times = np.arange(len(x)) * dt
    return CWTResult(times=times, scales=scales, power=power)


def dominant_periods_from_tfr(power: np.ndarray, axis_freq: int, top_k: int = 3) -> np.ndarray:
    # Sum energy over time to get marginal spectrum
    if axis_freq == 0:
        spec = power.sum(axis=1)
    else:
        spec = power.sum(axis=0)
    # Top-k indices by energy
    idx = np.argsort(spec)[::-1][:top_k]
    return idx


