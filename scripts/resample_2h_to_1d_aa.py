#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample 2h OHLCV CSV to 1d CSV with anti-aliasing on close.

- Applies a zero-phase FIR low-pass to the 2h close series (fs=12/day),
  with default cutoff ≈ 0.4 cycles/day (< 0.5/day Nyquist of 1d),
  then samples once per day (end-of-day) to avoid aliasing.
- Volume is aggregated by daily sum (no filtering, additive quantity).

Outputs a minimal CSV for Fourier: timestamp, close, volume

Usage:
  py -3 scripts/resample_2h_to_1d_aa.py --in data/BTC_USDT_2h.csv --out data/BTC_USDT_1d_from_2h_aa.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from scipy import signal


def _fir_lowpass_close(close: pd.Series, cutoff_per_day: float = 0.4) -> pd.Series:
    """Zero-phase FIR low-pass on close sampled at 12 bars/day (2h bars).

    cutoff_per_day is absolute cutoff in cycles/day, default 0.4/day.
    """
    fs = 12.0  # samples per day for 2h bars
    nyq = fs / 2.0
    b = signal.firwin(101, cutoff_per_day / nyq)
    y = signal.filtfilt(b, [1.0], close.to_numpy(dtype=float))
    return pd.Series(y, index=close.index, name=close.name)


def main() -> int:
    p = argparse.ArgumentParser(description="Anti-aliased resample 2h→1d (close filtered, volume summed)")
    p.add_argument('--in', dest='inp', required=True, help='Path to 2h OHLCV CSV (timestamp, open, high, low, close, volume)')
    p.add_argument('--out', dest='outp', required=True, help='Path to write 1d CSV (timestamp, close, volume)')
    p.add_argument('--cutoff', type=float, default=0.4, help='Cutoff in cycles/day for FIR low-pass on close (default: 0.4)')
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    if 'timestamp' not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    if 'close' not in df.columns:
        raise ValueError("Input CSV must contain a 'close' column")
    if 'volume' not in df.columns:
        # Create a zero volume if missing to keep schema stable
        df['volume'] = 0.0

    # Low-pass filter close before daily sampling
    close_filt = _fir_lowpass_close(df['close'], cutoff_per_day=args.cutoff)

    # Aggregate to daily
    close_d = close_filt.resample('1D').last()
    volume_d = df['volume'].resample('1D').sum()

    out = pd.DataFrame({'timestamp': close_d.index, 'close': close_d.values, 'volume': volume_d.reindex(close_d.index).values})
    out = out.dropna(subset=['close'])
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outp, index=False)
    print(f"Wrote {args.outp} rows={len(out)} range={out['timestamp'].min()} -> {out['timestamp'].max()}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


