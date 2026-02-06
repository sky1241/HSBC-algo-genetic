#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.4 - FFT Acceleration for Rolling Calculations

Provides O(N log N) implementations for:
- Rolling means (moving averages)
- Rolling ATR
- Ichimoku indicator batch computation
- Multi-parameter backtest optimization

Usage:
    from src.spectral.fft_acceleration import (
        fft_rolling_mean,
        fft_batch_ichimoku,
        BatchBacktester,
    )

    # Fast rolling mean
    ma = fft_rolling_mean(prices, window=26)

    # Batch Ichimoku for multiple parameter sets
    results = fft_batch_ichimoku(df, param_sets)
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from numpy.fft import rfft, irfft


def fft_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean using FFT convolution.

    Complexity: O(N log N) vs O(N*W) for naive implementation.

    Args:
        data: Input array
        window: Window size

    Returns:
        Rolling mean array (same length as input, NaN for first window-1)
    """
    n = len(data)
    if n < window:
        return np.full(n, np.nan)

    # Pad data
    padded = np.pad(data, (window - 1, 0), mode='constant', constant_values=0)

    # Create kernel
    kernel = np.ones(window) / window

    # FFT convolution
    n_fft = len(padded) + len(kernel) - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))  # Power of 2 for efficiency

    fft_data = rfft(padded, n_fft)
    fft_kernel = rfft(kernel, n_fft)
    result = irfft(fft_data * fft_kernel, n_fft)

    # Extract valid portion
    output = result[window - 1:window - 1 + n]

    # Set NaN for warming up period
    output[:window - 1] = np.nan

    return output


def fft_rolling_max(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling max.

    Note: Rolling max cannot be accelerated with FFT in general,
    but we use a deque-based O(N) algorithm.
    """
    from collections import deque

    n = len(data)
    result = np.full(n, np.nan)
    dq = deque()  # Store (value, index)

    for i in range(n):
        # Remove elements outside window
        while dq and dq[0][1] <= i - window:
            dq.popleft()

        # Remove smaller elements
        while dq and dq[-1][0] <= data[i]:
            dq.pop()

        dq.append((data[i], i))

        if i >= window - 1:
            result[i] = dq[0][0]

    return result


def fft_rolling_min(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling min using deque-based O(N) algorithm."""
    from collections import deque

    n = len(data)
    result = np.full(n, np.nan)
    dq = deque()

    for i in range(n):
        while dq and dq[0][1] <= i - window:
            dq.popleft()

        while dq and dq[-1][0] >= data[i]:
            dq.pop()

        dq.append((data[i], i))

        if i >= window - 1:
            result[i] = dq[0][0]

    return result


def fast_ichimoku_lines(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan: int,
    kijun: int,
    senkou_b: int,
) -> Dict[str, np.ndarray]:
    """Compute Ichimoku lines using optimized rolling functions.

    Returns dict with: tenkan_sen, kijun_sen, senkou_a, senkou_b
    """
    # Tenkan-sen
    tenkan_high = fft_rolling_max(high, tenkan)
    tenkan_low = fft_rolling_min(low, tenkan)
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Kijun-sen
    kijun_high = fft_rolling_max(high, kijun)
    kijun_low = fft_rolling_min(low, kijun)
    kijun_sen = (kijun_high + kijun_low) / 2

    # Senkou Span A
    senkou_a = (tenkan_sen + kijun_sen) / 2

    # Senkou Span B
    sb_high = fft_rolling_max(high, senkou_b)
    sb_low = fft_rolling_min(low, senkou_b)
    senkou_b_line = (sb_high + sb_low) / 2

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
    }


def fast_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute ATR using optimized rolling mean.

    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = rolling_mean(TR, period)
    """
    n = len(close)

    # True Range components
    h_l = high - low
    h_c = np.abs(high - np.roll(close, 1))
    l_c = np.abs(low - np.roll(close, 1))

    # First value has no previous close
    h_c[0] = h_l[0]
    l_c[0] = h_l[0]

    tr = np.maximum(np.maximum(h_l, h_c), l_c)

    # ATR via FFT rolling mean
    atr = fft_rolling_mean(tr, period)

    return atr


class BatchBacktester:
    """Batch backtester with shared indicator computation.

    When testing multiple parameter sets, common calculations
    are reused to reduce redundant computation.
    """

    def __init__(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
    ):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.n = len(close)

        # Cache for computed indicators
        self._cache: Dict[str, np.ndarray] = {}

    def _cache_key(self, indicator: str, params: Tuple) -> str:
        return f"{indicator}_{params}"

    def get_rolling_max(self, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get rolling max for high and low."""
        key_h = self._cache_key("rolling_max_h", (window,))
        key_l = self._cache_key("rolling_min_l", (window,))

        if key_h not in self._cache:
            self._cache[key_h] = fft_rolling_max(self.high, window)
        if key_l not in self._cache:
            self._cache[key_l] = fft_rolling_min(self.low, window)

        return self._cache[key_h], self._cache[key_l]

    def get_ichimoku(
        self,
        tenkan: int,
        kijun: int,
        senkou_b: int,
    ) -> Dict[str, np.ndarray]:
        """Get Ichimoku lines, using cache when possible."""
        key = self._cache_key("ichimoku", (tenkan, kijun, senkou_b))

        if key not in self._cache:
            result = fast_ichimoku_lines(
                self.high, self.low, self.close,
                tenkan, kijun, senkou_b
            )
            self._cache[key] = result

        return self._cache[key]

    def get_atr(self, period: int = 14) -> np.ndarray:
        """Get ATR, cached."""
        key = self._cache_key("atr", (period,))

        if key not in self._cache:
            self._cache[key] = fast_atr(self.high, self.low, self.close, period)

        return self._cache[key]

    def run_batch(
        self,
        param_sets: List[Dict[str, Any]],
        backtest_fn,
    ) -> List[Dict[str, Any]]:
        """Run backtest for multiple parameter sets.

        Args:
            param_sets: List of parameter dictionaries
            backtest_fn: Function(BatchBacktester, params) -> metrics dict

        Returns:
            List of result dictionaries
        """
        results = []

        for params in param_sets:
            try:
                metrics = backtest_fn(self, params)
                results.append({
                    "params": params,
                    "metrics": metrics,
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "params": params,
                    "error": str(e),
                    "success": False,
                })

        return results

    def clear_cache(self):
        """Clear the indicator cache."""
        self._cache.clear()


def benchmark_fft_vs_naive(n_samples: int = 50000, window: int = 50) -> Dict[str, float]:
    """Benchmark FFT vs naive rolling mean.

    Returns timing comparison.
    """
    import time

    data = np.random.randn(n_samples)

    # Naive implementation
    start = time.perf_counter()
    naive_result = np.convolve(data, np.ones(window)/window, mode='valid')
    naive_time = time.perf_counter() - start

    # FFT implementation
    start = time.perf_counter()
    fft_result = fft_rolling_mean(data, window)
    fft_time = time.perf_counter() - start

    # Verify correctness (within floating point tolerance)
    valid_start = window - 1
    max_diff = np.max(np.abs(naive_result - fft_result[valid_start:valid_start + len(naive_result)]))

    return {
        "n_samples": n_samples,
        "window": window,
        "naive_time_ms": naive_time * 1000,
        "fft_time_ms": fft_time * 1000,
        "speedup": naive_time / fft_time if fft_time > 0 else 0,
        "max_difference": max_diff,
        "correct": max_diff < 1e-10,
    }


# Vectorized signal generation for batch backtesting
def vectorized_ichimoku_signals(
    close: np.ndarray,
    tenkan_sen: np.ndarray,
    kijun_sen: np.ndarray,
    senkou_a: np.ndarray,
    senkou_b: np.ndarray,
    shift: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Ichimoku signals in vectorized form.

    Returns:
        (long_signals, short_signals) as boolean arrays
    """
    n = len(close)

    # Shifted cloud
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)

    # Lead lines (shifted back for signal generation)
    lead1_sig = np.roll(senkou_a, -(shift - 1))
    lead2_sig = np.roll(senkou_b, -(shift - 1))

    # Cloud direction
    cloud_bullish = lead1_sig > lead2_sig
    cloud_bearish = lead2_sig > lead1_sig

    # Cross detection
    prev_close = np.roll(close, 1)
    prev_lead2 = np.roll(lead2_sig, 1)

    crossup = (close > lead2_sig) & (prev_close <= prev_lead2)
    crossdn = (close < lead2_sig) & (prev_close >= prev_lead2)

    # Candle direction
    # For this we'd need open prices; approximate with previous close
    green_candle = close > prev_close
    red_candle = close < prev_close

    # Signals
    long_signals = crossup & cloud_bearish & green_candle
    short_signals = crossdn & cloud_bullish & red_candle

    # Clean up edges
    long_signals[:max(shift, 52)] = False
    short_signals[:max(shift, 52)] = False
    long_signals[-shift:] = False
    short_signals[-shift:] = False

    return long_signals, short_signals
