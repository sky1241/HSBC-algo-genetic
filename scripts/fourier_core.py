#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fourier_core: fonctions FFT/PSD et utilitaires de resampling anti‑alias

- compute_welch_psd: PSD via scipy.signal.welch (fallback numpy)
- dominant_period: période dominante 1/f*
- low_freq_power_ratio: ratio puissance < f0
- spectral_flatness: platitude spectrale (GM/AM)
- fir_lowpass_subsample: sous‑échantillonnage avec filtre FIR passe‑bas zéro‑phase

Ces primitives sont factorisées pour garantir une implémentation unique et
cohérente à travers les scripts.
"""
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd


def compute_welch_psd(close: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule la PSD avec Welch. fs en échantillons par unité de temps (bar^-1).

    Utilise scipy si disponible; sinon, repli sur un périodogramme fenêtré.
    Retourne (freqs, psd).
    """
    try:
        from scipy.signal import welch  # type: ignore
        n = len(close)
        nperseg = min(1024, max(64, n // 4))
        freqs, psd = welch(close, fs=fs, nperseg=nperseg)
        return freqs, psd
    except Exception:
        # Fallback: periodogram naïf fenêtré Hann
        n = len(close)
        if n <= 1:
            return np.asarray([0.0]), np.asarray([0.0])
        window = np.hanning(n)
        close_d = close - float(np.nanmean(close))
        fft = np.fft.rfft(window * close_d)
        psd = (np.abs(fft) ** 2) / (np.sum(window ** 2) * fs)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, psd


def dominant_period(freqs: np.ndarray, psd: np.ndarray, *, min_idx: int = 1) -> float:
    """Retourne la période dominante P = 1/f* (ignore DC par défaut)."""
    if freqs is None or psd is None or len(freqs) <= min_idx or len(psd) <= min_idx:
        return float("nan")
    idx = int(np.nanargmax(psd[min_idx:]) + min_idx)
    f_star = max(1e-12, float(freqs[idx]))
    return 1.0 / f_star


def low_freq_power_ratio(freqs: np.ndarray, psd: np.ndarray, *, f0: float) -> float:
    """Part de puissance cumulée sous f0."""
    if freqs is None or psd is None or len(freqs) == 0 or len(psd) == 0:
        return float("nan")
    total = float(np.nansum(psd))
    if total <= 0:
        return float("nan")
    mask = freqs < float(f0)
    low = float(np.nansum(psd[mask]))
    return low / total


def spectral_flatness(psd: np.ndarray) -> float:
    """GM/AM de la densité spectrale (proche de 0: picquée; proche de 1: plate)."""
    psd = np.asarray(psd, dtype=float)
    psd = psd + 1e-12
    gmean = np.exp(float(np.nanmean(np.log(psd))))
    amean = float(np.nanmean(psd))
    return float(gmean / amean) if amean > 0 else float("nan")


def fir_lowpass_subsample(df: pd.DataFrame, q: int, *, fs: float, cutoff: float) -> pd.DataFrame:
    """Filtre passe‑bas FIR zéro‑phase puis sous‑échantillonne 1/q les lignes.

    Paramètres
    - df: DataFrame avec index temporel et colonnes numériques
    - q: facteur de sous‑échantillonnage (garder 1 ligne sur q)
    - fs: fréquence d’échantillonnage (échantillons par jour, ex 12 pour H2)
    - cutoff: coupure en cycles/jour (doit être < fs/2 et compatible Nyquist cible)
    """
    from scipy import signal  # lazy import

    if q <= 1:
        return df.copy()
    nyq = fs / 2.0
    wc = float(cutoff) / float(nyq)
    # Longueur 101 par défaut; peut être ajustée si besoin
    b = signal.firwin(101, wc)
    filt_cols: Dict[str, np.ndarray] = {}
    for c in df.columns:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy()
        try:
            filt_cols[c] = signal.filtfilt(b, [1.0], arr)
        except Exception:
            # Fallback: pas de filtrage si problème numérique
            filt_cols[c] = arr
    df_filt = pd.DataFrame(filt_cols, index=df.index)
    return df_filt.iloc[::q]


def anti_aliased_daily(df_2h: pd.DataFrame) -> pd.DataFrame:
    """Convertit des features H2 en D1 via un filtrage passe‑bas puis échantillonnage quotidien.

    - Applique fir_lowpass_subsample(q=12, fs=12/jour, cutoff=0.4 cycles/jour)
    - Puis resample en '1D' en prenant la dernière valeur disponible de la journée
    """
    df_f = fir_lowpass_subsample(df_2h, q=12, fs=12.0, cutoff=0.4)
    daily = df_f.resample("1D").last().dropna(how="all")
    return daily


__all__ = [
    "compute_welch_psd",
    "dominant_period",
    "low_freq_power_ratio",
    "spectral_flatness",
    "fir_lowpass_subsample",
    "anti_aliased_daily",
]


