import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fourier_utils import (
    compute_welch_psd,
    dominant_period,
    low_freq_power_ratio,
)

HALVING_DATE = "2024-04-20"
BARS_PER_MONTH = 30 * 12  # H2 bars in ~30 days


def load_returns(csv: Path, start: str = HALVING_DATE) -> pd.Series:
    """Load log-returns for BTC from a CSV file starting at a given date."""
    df = (
        pd.read_csv(csv, parse_dates=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )
    df = df[df.index >= start]
    df["log_close"] = np.log(df["close"])
    return df["log_close"].diff().dropna()


def monthly_spectra(ret: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute Welch PSD for each monthly window of returns."""
    starts = range(0, len(ret) - BARS_PER_MONTH + 1, BARS_PER_MONTH)
    spectra: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in starts:
        seg = ret.iloc[s : s + BARS_PER_MONTH].values
        freqs, psd = compute_welch_psd(seg, fs=1.0)
        spectra.append((freqs[1:], psd[1:]))  # drop DC component
    return spectra


def analyze_last_month(spectra: List[Tuple[np.ndarray, np.ndarray]]):
    """Return metrics for the last monthly spectrum."""
    freqs, psd = spectra[-1]
    P = dominant_period(freqs, psd)
    lfp = low_freq_power_ratio(freqs, psd, f0=1 / (5 * 12))
    log_f, log_psd = np.log(freqs), np.log(psd)
    alpha, _ = np.polyfit(log_f, log_psd, 1)
    return {"P_bars": float(P), "LFP": float(lfp), "alpha": float(-alpha)}, (freqs, psd)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Analyze Fourier transform of BTC price since the 2024 halving"
    )
    p.add_argument("csv", type=Path, help="CSV file with BTC data")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Display log-log PSD plot for the most recent month",
    )
    args = p.parse_args()

    ret = load_returns(args.csv)
    spectra = monthly_spectra(ret)
    metrics, (freqs, psd) = analyze_last_month(spectra)
    print(json.dumps(metrics, indent=2))

    if args.plot:
        plt.figure(figsize=(6, 4))
        plt.loglog(freqs, psd, label="PSD ret H2 (last month)")
        plt.xlabel("Frequency (cycles/bar)")
        plt.ylabel("Power")
        plt.title("Welch PSD – BTC returns since halving")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
