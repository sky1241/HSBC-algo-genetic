import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Core FFT/PSD utilities (factorized)
from scripts.fourier_core import (
    compute_welch_psd,
    dominant_period,
    low_freq_power_ratio,
    spectral_flatness,
)


def scale_ichimoku(base: Tuple[int, int, int] = (9, 26, 52), *, days_per_week: int = 7, bars_per_day: int = 12) -> Tuple[int, int, int]:
    """Scale Ichimoku periods from daily defaults to target timeframe.

    Parameters
    ----------
    base: tuple of ints
        Baseline Ichimoku windows in days for a 7-day week.
    days_per_week: int
        Trading days per week for the asset (e.g. 5 for stocks, 7 for crypto).
    bars_per_day: int
        Number of bars per day for the timeframe (e.g. 12 for 2h data).

    Returns
    -------
    Tuple[int, int, int]
        Scaled (tenkan, kijun, senkou_b) expressed in bars.
    """

    scale = bars_per_day * (days_per_week / 7.0)
    return tuple(int(round(x * scale)) for x in base)


def suggest_ichimoku_params(
    P: float,
    lfp: float,
    *,
    days_per_week: int = 7,
    bars_per_day: int = 12,
) -> Dict[str, float]:
    """Heuristic mapping from dominant period and LFP to Ichimoku values."""
    base_t, base_k, base_sb = scale_ichimoku(days_per_week=days_per_week, bars_per_day=bars_per_day)
    if not np.isfinite(P) or P <= 0:
        return {"tenkan": base_t, "kijun": base_k, "senkou_b": base_sb, "shift": base_k, "atr_mult": 3.0}
    kijun = max(base_k, int(round(P / 2)))
    tenkan = max(base_t, int(round(P / 6)))
    senkou_b = max(base_sb, int(round(P)))
    shift = int(kijun)
    atr_mult = 4.0 if lfp >= 0.6 else 2.6
    return {"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b, "shift": shift, "atr_mult": atr_mult}


def analyze_csv(path: Path, timeframe_hours: float = 2.0, window_days: int = 180) -> Dict[str, float]:
    df = pd.read_csv(path)
    # Expect a column containing close prices (case-insensitive search).
    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    if close_col is None:
        raise ValueError(f"No close column found in {path}")
    close = df[close_col].to_numpy(dtype=float)
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


def analyze_and_suggest(
    symbol: str,
    csv_path: Path,
    *,
    timeframe_hours: float = 2.0,
    days_per_week: int = 7,
) -> Dict[str, Dict[str, float]]:
    stats = analyze_csv(csv_path, timeframe_hours=timeframe_hours)
    bars_per_day = int(round(24.0 / timeframe_hours))
    params = suggest_ichimoku_params(
        stats.get("P_bars", float("nan")),
        stats.get("LFP", float("nan")),
        days_per_week=days_per_week,
        bars_per_day=bars_per_day,
    )
    return {symbol: params | {"_stats": stats}}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fourier-based suggester for Ichimoku params (Welch PSD, LFP)")
    p.add_argument("symbol")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs")/"FOURIER_BASELINE.json")
    p.add_argument("--timeframe-hours", type=float, default=2.0)
    p.add_argument("--days-per-week", type=int, default=7)
    args = p.parse_args()
    out = analyze_and_suggest(
        args.symbol,
        args.csv,
        timeframe_hours=args.timeframe_hours,
        days_per_week=args.days_per_week,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


