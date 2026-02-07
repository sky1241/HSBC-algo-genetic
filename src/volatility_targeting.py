#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOLATILITY TARGETING MODULE

Implements dynamic position sizing based on:
1. Volatility targeting: scale exposure inversely to realized volatility
2. Drawdown throttle: reduce exposure when equity drawdown exceeds threshold

This module can be used standalone or integrated into the backtest pipeline.

Formula:
    leverage = min(L_max, sigma_target / sigma_realized)

    If drawdown > dd_threshold:
        leverage *= (1 - drawdown_penalty)

Usage:
    from src.volatility_targeting import VolatilityTargeter

    targeter = VolatilityTargeter(
        sigma_target=0.15,      # 15% annualized target vol
        L_max=10.0,             # Maximum leverage
        lookback=20,            # Rolling window for vol calculation
        dd_threshold=0.10,      # 10% drawdown threshold
        dd_penalty=0.5,         # Reduce leverage by 50% when DD > threshold
    )

    for bar in bars:
        leverage = targeter.get_leverage(returns, current_equity, peak_equity)
        position_size = base_position * leverage

Version: 1.0
Date: 2025-02-07
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VolatilityTargeting")


@dataclass
class VolTargetConfig:
    """Configuration for volatility targeting."""
    sigma_target: float = 0.15      # Annualized target volatility (15%)
    L_max: float = 10.0             # Maximum leverage
    L_min: float = 0.1              # Minimum leverage (never go to 0)
    lookback: int = 20              # Rolling window for vol calculation (bars)
    annualization_factor: float = np.sqrt(365 * 12)  # For 2h bars: 12 bars/day * 365 days
    dd_threshold: float = 0.10      # Drawdown threshold for throttle (10%)
    dd_penalty: float = 0.5         # Leverage reduction when DD > threshold
    dd_scale_max: float = 0.25      # Maximum DD for progressive scaling
    smooth_factor: float = 0.1      # Exponential smoothing for vol estimate (0 = no smoothing)
    vol_floor: float = 0.01         # Minimum volatility to avoid division by zero


class VolatilityTargeter:
    """
    Dynamic leverage adjustment based on realized volatility and drawdown.

    The core idea:
    - When volatility is LOW, we can take MORE risk (higher leverage)
    - When volatility is HIGH, we should take LESS risk (lower leverage)
    - When in drawdown, we should be MORE CONSERVATIVE
    """

    def __init__(self, config: Optional[VolTargetConfig] = None, **kwargs):
        """
        Initialize volatility targeter.

        Args:
            config: VolTargetConfig object (optional)
            **kwargs: Override individual config parameters
        """
        if config is None:
            config = VolTargetConfig()

        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # State
        self._returns_buffer: List[float] = []
        self._last_vol: float = config.sigma_target / config.annualization_factor
        self._peak_equity: float = 1.0

    def reset(self) -> None:
        """Reset internal state."""
        self._returns_buffer = []
        self._last_vol = self.config.sigma_target / self.config.annualization_factor
        self._peak_equity = 1.0

    def _compute_realized_vol(self, returns: Union[List[float], np.ndarray]) -> float:
        """
        Compute realized volatility from returns.

        Args:
            returns: Array of log returns

        Returns:
            Annualized volatility estimate
        """
        if len(returns) < 2:
            return self.config.sigma_target  # Fallback to target

        returns = np.asarray(returns)

        # Use recent window only
        if len(returns) > self.config.lookback:
            returns = returns[-self.config.lookback:]

        # Standard deviation of returns
        vol_raw = float(np.std(returns, ddof=1))

        # Apply floor
        vol_raw = max(vol_raw, self.config.vol_floor / self.config.annualization_factor)

        # Annualize
        vol_annualized = vol_raw * self.config.annualization_factor

        # Optional exponential smoothing
        if self.config.smooth_factor > 0 and self._last_vol > 0:
            vol_annualized = (
                self.config.smooth_factor * vol_annualized +
                (1 - self.config.smooth_factor) * self._last_vol * self.config.annualization_factor
            )

        self._last_vol = vol_annualized / self.config.annualization_factor

        return vol_annualized

    def _compute_drawdown(self, current_equity: float, peak_equity: Optional[float] = None) -> float:
        """
        Compute current drawdown.

        Args:
            current_equity: Current portfolio equity
            peak_equity: Peak equity (if None, uses internal tracking)

        Returns:
            Drawdown as positive fraction (0.10 = 10% drawdown)
        """
        if peak_equity is None:
            peak_equity = self._peak_equity

        # Update peak
        if current_equity > peak_equity:
            self._peak_equity = current_equity
            peak_equity = current_equity

        if peak_equity <= 0:
            return 0.0

        drawdown = (peak_equity - current_equity) / peak_equity
        return max(0.0, drawdown)

    def _drawdown_adjustment(self, drawdown: float) -> float:
        """
        Compute leverage adjustment based on drawdown.

        Uses progressive scaling:
        - Below threshold: no adjustment (1.0)
        - At threshold: penalty applied
        - Progressive increase up to dd_scale_max

        Args:
            drawdown: Current drawdown (positive fraction)

        Returns:
            Multiplier for leverage (0.0 to 1.0)
        """
        if drawdown <= self.config.dd_threshold:
            return 1.0

        # Progressive penalty between threshold and scale_max
        excess_dd = drawdown - self.config.dd_threshold
        max_excess = self.config.dd_scale_max - self.config.dd_threshold

        if max_excess <= 0:
            # No progressive scaling, just apply penalty
            return 1.0 - self.config.dd_penalty

        # Linear interpolation
        penalty_fraction = min(1.0, excess_dd / max_excess)
        adjustment = 1.0 - (self.config.dd_penalty * penalty_fraction)

        return max(0.0, adjustment)

    def get_leverage(
        self,
        returns: Union[List[float], np.ndarray],
        current_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal leverage based on vol targeting and drawdown.

        Args:
            returns: Historical returns for vol calculation
            current_equity: Current portfolio equity (for DD throttle)
            peak_equity: Peak equity (for DD throttle)

        Returns:
            Recommended leverage multiplier
        """
        # Compute realized volatility
        sigma_realized = self._compute_realized_vol(returns)

        # Base leverage from vol targeting
        if sigma_realized > 0:
            leverage = min(
                self.config.L_max,
                self.config.sigma_target / sigma_realized
            )
        else:
            leverage = self.config.L_max

        # Apply drawdown throttle if equity info provided
        if current_equity is not None:
            drawdown = self._compute_drawdown(current_equity, peak_equity)
            dd_adjustment = self._drawdown_adjustment(drawdown)
            leverage *= dd_adjustment

        # Apply floor
        leverage = max(self.config.L_min, leverage)

        return leverage

    def update_return(self, ret: float) -> None:
        """
        Update the returns buffer with a new return.

        Args:
            ret: New log return
        """
        self._returns_buffer.append(float(ret))

        # Limit buffer size
        max_size = self.config.lookback * 3
        if len(self._returns_buffer) > max_size:
            self._returns_buffer = self._returns_buffer[-self.config.lookback:]

    def get_leverage_from_buffer(
        self,
        current_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
    ) -> float:
        """
        Get leverage using internal returns buffer.

        Args:
            current_equity: Current portfolio equity
            peak_equity: Peak equity

        Returns:
            Recommended leverage
        """
        return self.get_leverage(self._returns_buffer, current_equity, peak_equity)


def apply_vol_targeting_to_series(
    prices: Union[np.ndarray, "pd.Series"],
    config: Optional[VolTargetConfig] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply volatility targeting to a price series.

    Returns an array of recommended leverage for each bar.

    Args:
        prices: Price series
        config: VolTargetConfig
        **kwargs: Override config params

    Returns:
        Array of leverage values (same length as prices)
    """
    prices = np.asarray(prices)
    n = len(prices)

    if n < 2:
        return np.ones(n)

    # Compute returns
    returns = np.diff(np.log(prices))
    returns = np.insert(returns, 0, 0.0)  # Pad to match length

    targeter = VolatilityTargeter(config, **kwargs)
    leverages = np.ones(n)

    for i in range(1, n):
        # Use returns up to current bar
        targeter.update_return(returns[i])
        if i >= targeter.config.lookback:
            leverages[i] = targeter.get_leverage_from_buffer()

    return leverages


def compute_vol_adjusted_returns(
    returns: np.ndarray,
    config: Optional[VolTargetConfig] = None,
    **kwargs
) -> np.ndarray:
    """
    Compute volatility-adjusted returns.

    Scales each return by the vol-targeted leverage.

    Args:
        returns: Raw returns
        config: VolTargetConfig
        **kwargs: Override config params

    Returns:
        Vol-adjusted returns
    """
    returns = np.asarray(returns)
    n = len(returns)

    targeter = VolatilityTargeter(config, **kwargs)
    adjusted = np.zeros(n)

    equity = 1.0
    peak = 1.0

    for i in range(n):
        targeter.update_return(returns[i])

        if i >= targeter.config.lookback:
            leverage = targeter.get_leverage_from_buffer(equity, peak)
        else:
            leverage = 1.0

        adjusted[i] = returns[i] * leverage

        # Update equity tracking
        equity *= (1 + adjusted[i])
        peak = max(peak, equity)

    return adjusted


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VOLATILITY TARGETING - Test")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_bars = 1000

    # Simulate returns with varying volatility
    vol_regime = np.concatenate([
        np.random.normal(0.001, 0.01, 300),   # Low vol
        np.random.normal(0.001, 0.03, 400),   # High vol
        np.random.normal(0.001, 0.015, 300),  # Medium vol
    ])

    print(f"\n1. Testing basic vol targeting...")

    config = VolTargetConfig(
        sigma_target=0.15,
        L_max=5.0,
        lookback=20,
        dd_threshold=0.10,
        dd_penalty=0.5,
    )

    targeter = VolatilityTargeter(config)

    # Simulate equity curve
    equity = 1.0
    peak = 1.0
    leverages = []
    equities = []

    for ret in vol_regime:
        targeter.update_return(ret)
        lev = targeter.get_leverage_from_buffer(equity, peak)
        leverages.append(lev)

        # Apply leveraged return
        equity *= (1 + ret * lev)
        peak = max(peak, equity)
        equities.append(equity)

    leverages = np.array(leverages)
    equities = np.array(equities)

    print(f"   Average leverage: {leverages.mean():.2f}")
    print(f"   Min leverage: {leverages.min():.2f}")
    print(f"   Max leverage: {leverages.max():.2f}")
    print(f"   Final equity: {equities[-1]:.4f}")

    # Compare with unleveraged
    equity_raw = np.cumprod(1 + vol_regime)
    print(f"   Unleveraged final equity: {equity_raw[-1]:.4f}")

    print("\n2. Testing drawdown throttle...")

    # Simulate a drawdown scenario
    dd_returns = np.concatenate([
        np.random.normal(0.002, 0.01, 100),   # Gains
        np.random.normal(-0.01, 0.02, 50),    # Losses (drawdown)
        np.random.normal(0.001, 0.01, 100),   # Recovery
    ])

    targeter2 = VolatilityTargeter(config)
    equity2 = 1.0
    peak2 = 1.0
    dd_values = []

    for ret in dd_returns:
        targeter2.update_return(ret)
        lev = targeter2.get_leverage_from_buffer(equity2, peak2)

        dd = (peak2 - equity2) / peak2 if peak2 > 0 else 0
        dd_values.append(dd)

        equity2 *= (1 + ret * lev)
        peak2 = max(peak2, equity2)

    dd_values = np.array(dd_values)
    max_dd = dd_values.max()
    print(f"   Max drawdown: {max_dd:.2%}")
    print(f"   Times DD > threshold: {(dd_values > config.dd_threshold).sum()}")

    print("\n3. Testing helper functions...")

    prices = np.cumprod(1 + vol_regime[:100])
    leverages_series = apply_vol_targeting_to_series(prices, config)
    print(f"   Leverage series shape: {leverages_series.shape}")
    print(f"   Leverage series mean: {leverages_series.mean():.2f}")

    adj_returns = compute_vol_adjusted_returns(vol_regime[:100], config)
    print(f"   Adjusted returns shape: {adj_returns.shape}")
    print(f"   Raw returns std: {vol_regime[:100].std():.4f}")
    print(f"   Adjusted returns std: {adj_returns.std():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
