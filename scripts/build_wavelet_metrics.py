#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse
import time
import traceback
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.wavelet_utils import compute_stft  # type: ignore
from tqdm import tqdm


OHLCV_PATHS = {
    ("BTC_USDT", "2h"): Path("data") / "BTC_USDT_2h.csv",
    ("BTC_USDT", "1d"): Path("data") / "BTC_USDT_1d.csv",
    ("BTC_USD", "2h"): Path("data") / "BTC_USD_2h.csv",
    ("BTC_USD", "1d"): Path("data") / "BTC_USD_1d.csv",
}


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df[["open","high","low","close","volume"]]


def month_index(idx: pd.DatetimeIndex) -> pd.Series:
    return idx.to_period('M').to_timestamp()


def build_time_index_mapping(n_samples: int, n_cols: int) -> np.ndarray:
    # Map STFT/CWT time columns back to integer indices into the close series
    return np.clip(np.linspace(0, n_samples - 1, num=n_cols).astype(int), 0, n_samples - 1)


def stft_monthly_metrics(close: pd.Series, lf_min_bars: int = 64, lf_max_bars: int = 4096) -> pd.DataFrame:
    # One sample per bar
    st = compute_stft(close.values, fs=1.0, nperseg=128, noverlap=96)
    freqs = st.freqs
    power = st.power  # (freqs, times)
    # Dominant frequency per time slice
    dom_idx = np.nanargmax(power, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_bars = np.where(freqs > 0, 1.0 / freqs, np.nan)
    dom_periods = periods_bars[dom_idx]
    # Low-frequency band: periods in [lf_min_bars, lf_max_bars]
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_bars = np.where(freqs > 0, 1.0 / freqs, np.nan)
    mask_lf = (periods_bars >= lf_min_bars) & (periods_bars <= lf_max_bars)
    total_power_t = np.nansum(power, axis=0)
    lf_power_t = np.nansum(power[mask_lf, :], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lfp_like_t = np.where(total_power_t > 0, lf_power_t / total_power_t, np.nan)

    # Map columns to timestamps / months
    col_idx = build_time_index_mapping(len(close), power.shape[1])
    ts = close.index[col_idx]
    mon = month_index(ts)

    df = pd.DataFrame({
        'timestamp': ts,
        'month': mon,
        'STFT_domP_bars': dom_periods,
        'STFT_LFP_like': lfp_like_t,
    })
    agg = df.groupby('month').agg({
        'STFT_domP_bars': 'median',
        'STFT_LFP_like': 'mean',
    }).reset_index()
    agg['month'] = pd.to_datetime(agg['month']).dt.strftime('%Y-%m')
    return agg


def cwt_monthly_metrics(close: pd.Series, lf_min_bars: int = 64, lf_max_bars: int = 4096) -> pd.DataFrame:
    import pywt
    # Group by calendar month to keep CWT computations small and fast
    month_keys = month_index(close.index)
    groups = {}
    for ts, val in zip(close.index, close.values):
        m = pd.Timestamp(year=ts.year, month=ts.month, day=1)
        groups.setdefault(m, []).append(val)

    rows = []
    f_c = pywt.central_frequency('morl') if hasattr(pywt, 'central_frequency') else 0.8125
    for m, vals in groups.items():
        x = np.asarray(vals, dtype=float)
        n = len(x)
        if n < 32:
            continue
        # Bound periods to feasible range for this month
        pmin = max(2, min(lf_min_bars, n // 2))
        pmax = max(pmin + 1, min(lf_max_bars, n // 2))
        # Build compact scale grid focused on [pmin, pmax]
        scale_min = pmin * f_c
        scale_max = pmax * f_c
        scales = np.geomspace(scale_min, scale_max, num=32)
        coeffs, freqs = pywt.cwt(x, scales, 'morl', sampling_period=1.0)
        power = (np.abs(coeffs) ** 2)
        # Dominant scale for the month from time-averaged spectrum
        spec = np.nanmean(power, axis=1)
        dom_idx = int(np.nanargmax(spec))
        dom_scale = scales[dom_idx]
        # Convert dominant scale to period (bars)
        dom_freq = pywt.scale2frequency('morl', dom_scale)
        dom_period = (1.0 / dom_freq) if dom_freq > 0 else np.nan
        # LF ratio within this scale grid: top 25% of periods (largest scales)
        k = max(1, int(0.75 * len(scales)))
        e_total = float(np.nansum(power))
        e_lf = float(np.nansum(power[k:, :]))
        lfp_like = (e_lf / e_total) if e_total > 0 else np.nan
        rows.append({'month': m, 'CWT_domScale_bars': dom_period, 'CWT_LFP_like': lfp_like})

    if not rows:
        return pd.DataFrame(columns=['month','CWT_domScale_bars','CWT_LFP_like'])
    df = pd.DataFrame(rows)
    df['month'] = pd.to_datetime(df['month']).dt.strftime('%Y-%m')
    return df


def compute_for_pair(symbol: str, timeframe: str, verbose: bool = False) -> pd.DataFrame:
    path = OHLCV_PATHS.get((symbol, timeframe))
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    df = read_ohlcv(path)
    t0 = time.time()
    if verbose:
        print(f"[metrics] {symbol} {timeframe}: n={len(df)} bars", flush=True)
    st = stft_monthly_metrics(df['close'])
    if verbose:
        print(f"[metrics] {symbol} {timeframe}: STFT done in {time.time()-t0:.2f}s, rows={len(st)}", flush=True)
    t1 = time.time()
    cw = cwt_monthly_metrics(df['close'])
    if verbose:
        print(f"[metrics] {symbol} {timeframe}: CWT done in {time.time()-t1:.2f}s, rows={len(cw)}", flush=True)
    out = pd.merge(st, cw, on='month', how='outer')
    out['symbol'] = symbol
    out['timeframe'] = timeframe
    cols_front = ['month','symbol','timeframe']
    return out[cols_front + [c for c in out.columns if c not in cols_front]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    tasks = [("BTC_USDT","2h"),("BTC_USDT","1d"),("BTC_USD","2h"),("BTC_USD","1d")]
    frames: List[pd.DataFrame] = []
    for sym, tf in tqdm(tasks, desc='Wavelet/STFT metrics', leave=False):
        try:
            frames.append(compute_for_pair(sym, tf, verbose=args.verbose))
        except FileNotFoundError:
            if args.verbose:
                print(f"[metrics] Missing OHLCV for {sym} {tf}", flush=True)
            continue
        except Exception:
            print(f"[metrics] ERROR {sym} {tf}:", flush=True)
            traceback.print_exc()
            continue
    if not frames:
        print('No OHLCV found')
        return 0
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(['month','symbol','timeframe'])
    out_dir = Path('outputs') / 'fourier' / 'wavelets'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'WAVELET_STFT_MONTHLY_METRICS.csv'
    full.to_csv(out_csv, index=False)
    print('Wrote:', out_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


