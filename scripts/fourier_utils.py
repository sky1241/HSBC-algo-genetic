import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_welch_psd(close: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple PSD using Welch via numpy (fallback) or scipy if available.

    Returns (freqs, psd). fs is sampling frequency in bars^-1 (e.g., 1 per bar).
    """
    try:
        from scipy.signal import welch  # type: ignore
        freqs, psd = welch(close, fs=fs, nperseg=min(1024, max(64, len(close)//4)))
        return freqs, psd
    except Exception:
        # Fallback: naive periodogram
        n = len(close)
        window = np.hanning(n)
        close_d = close - np.mean(close)
        fft = np.fft.rfft(window * close_d)
        psd = (np.abs(fft) ** 2) / (np.sum(window**2) * fs)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        return freqs, psd


def dominant_period(freqs: np.ndarray, psd: np.ndarray, min_idx: int = 1) -> float:
    """Return dominant period P = 1/f* (skip DC at index 0)."""
    if len(freqs) <= min_idx:
        return float('nan')
    idx = np.argmax(psd[min_idx:]) + min_idx
    f_star = max(1e-12, freqs[idx])
    return 1.0 / f_star


def low_freq_power_ratio(freqs: np.ndarray, psd: np.ndarray, f0: float) -> float:
    total = float(np.sum(psd))
    if total <= 0:
        return float('nan')
    mask = freqs < f0
    low = float(np.sum(psd[mask]))
    return low / total


def spectral_flatness(psd: np.ndarray) -> float:
    # Geometric mean / arithmetic mean
    psd = np.asarray(psd) + 1e-12
    gmean = np.exp(np.mean(np.log(psd)))
    amean = float(np.mean(psd))
    return float(gmean / amean)


def suggest_ichimoku_params(P: float, lfp: float) -> Dict[str, float]:
    """Heuristic mapping from P (bars) and LFP to Ichimoku ranges/values."""
    if not np.isfinite(P) or P <= 0:
        # Fallback defaults
        return {"tenkan": 12, "kijun": 34, "senkou_b": 72, "shift": 26, "atr_mult": 3.0}
    kijun = max(10, int(round(P / 2)))
    tenkan = max(5, int(round(P / 6)))
    senkou_b = max(20, int(round(P)))
    shift = max(10, int(round(kijun / 2)))
    if lfp >= 0.6:
        atr_mult = 4.0
    else:
        atr_mult = 2.6
    return {"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b, "shift": shift, "atr_mult": atr_mult}


def analyze_csv(path: Path, timeframe_hours: float = 2.0, window_days: int = 180) -> Dict[str, float]:
    df = pd.read_csv(path)
    # Expect columns: timestamp, open, high, low, close, volume (flexible)
    close = df[df.columns[-2]].to_numpy(dtype=float) if 'close' not in df.columns else df['close'].to_numpy(dtype=float)
    # Use last N days
    bars_per_day = int(round(24.0 / timeframe_hours))
    n = max(256, window_days * bars_per_day)
    close_win = close[-n:]
    fs = 1.0  # 1 unit per bar
    freqs, psd = compute_welch_psd(close_win, fs=fs)
    P = dominant_period(freqs, psd)
    # f0 = cycles > 5 days in H2 → 5*bars_per_day bars period → freq threshold
    f0 = 1.0 / float(5 * bars_per_day)
    lfp = low_freq_power_ratio(freqs, psd, f0=f0)
    flat = spectral_flatness(psd)
    return {"P_bars": float(P), "LFP": float(lfp), "flatness": float(flat)}


def analyze_and_suggest(symbol: str, csv_path: Path) -> Dict[str, Dict[str, float]]:
    stats = analyze_csv(csv_path)
    params = suggest_ichimoku_params(stats.get("P_bars", float('nan')), stats.get("LFP", float('nan')))
    return {symbol: params | {"_stats": stats}}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fourier-based suggester for Ichimoku params (Welch PSD, LFP)")
    p.add_argument("symbol")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs")/"FOURIER_BASELINE.json")
    args = p.parse_args()
    out = analyze_and_suggest(args.symbol, args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


