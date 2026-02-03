#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.3 - ATR-Based Dynamic Position Sizing

Provides volatility-adjusted position sizing:
- ATR-based risk per trade
- Kelly criterion estimation
- Regime-adaptive sizing
- Risk budget management

Usage:
    from src.spectral.position_sizing import ATRPositionSizer

    sizer = ATRPositionSizer(
        risk_per_trade=0.01,  # 1% risk per trade
        max_position=0.10,    # 10% max position
    )

    size = sizer.calculate(
        capital=10000,
        entry_price=50000,
        atr=1000,
        atr_mult=3.0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    size_units: float      # Position size in base units
    size_value: float      # Position value in quote currency
    size_pct: float        # Position as % of capital
    risk_amount: float     # Dollar amount at risk
    stop_distance: float   # Distance to stop loss
    leverage_used: float   # Effective leverage


class ATRPositionSizer:
    """ATR-based dynamic position sizing.

    Uses the formula:
        position_size = (capital * risk_pct) / (atr * atr_mult)

    This ensures each trade risks a fixed % of capital regardless
    of current volatility.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,    # 1% risk per trade
        max_position: float = 0.10,       # 10% max position
        max_leverage: float = 10.0,       # Maximum leverage
        min_position_value: float = 10.0, # Minimum position (Binance limit)
    ):
        """
        Args:
            risk_per_trade: Risk as fraction of capital (0.01 = 1%)
            max_position: Max position as fraction of capital
            max_leverage: Maximum allowed leverage
            min_position_value: Minimum position value in quote currency
        """
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.min_position_value = min_position_value

    def calculate(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        atr_mult: float,
        direction: int = 1,  # 1 for long, -1 for short
    ) -> PositionSize:
        """Calculate position size based on ATR.

        Args:
            capital: Available capital
            entry_price: Entry price
            atr: Current ATR value
            atr_mult: ATR multiplier for stop loss
            direction: Trade direction (1=long, -1=short)

        Returns:
            PositionSize with all sizing information
        """
        # Stop distance in price terms
        stop_distance = atr * atr_mult

        if stop_distance <= 0 or entry_price <= 0:
            return PositionSize(
                size_units=0,
                size_value=0,
                size_pct=0,
                risk_amount=0,
                stop_distance=0,
                leverage_used=0,
            )

        # Risk amount
        risk_amount = capital * self.risk_per_trade

        # Position size in units
        # If we risk $100 with a $50 stop distance, we can hold 2 units
        size_units = risk_amount / stop_distance

        # Position value
        size_value = size_units * entry_price

        # Apply max position constraint
        max_value = capital * self.max_position
        if size_value > max_value:
            size_value = max_value
            size_units = size_value / entry_price
            risk_amount = size_units * stop_distance

        # Apply max leverage constraint
        leverage = size_value / capital
        if leverage > self.max_leverage:
            size_value = capital * self.max_leverage
            size_units = size_value / entry_price
            leverage = self.max_leverage
            risk_amount = size_units * stop_distance

        # Apply minimum position
        if size_value < self.min_position_value:
            size_value = self.min_position_value
            size_units = size_value / entry_price
            risk_amount = size_units * stop_distance

        return PositionSize(
            size_units=round(size_units, 8),
            size_value=round(size_value, 2),
            size_pct=round(size_value / capital, 4),
            risk_amount=round(risk_amount, 2),
            stop_distance=round(stop_distance, 2),
            leverage_used=round(size_value / capital, 2),
        )

    def calculate_with_regime(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        atr_mult: float,
        regime: str,
        direction: int = 1,
    ) -> PositionSize:
        """Calculate position size with regime adjustment.

        Regime adjustments:
        - TREND: Normal sizing (capture moves)
        - MIXED: Reduced sizing (uncertainty)
        - NOISE: Further reduced (high chop)
        """
        regime_multipliers = {
            "trend": 1.0,
            "mixed": 0.7,
            "noise": 0.5,
            "unknown": 0.5,
        }

        mult = regime_multipliers.get(regime.lower(), 0.7)

        # Temporarily adjust risk
        original_risk = self.risk_per_trade
        self.risk_per_trade = original_risk * mult

        result = self.calculate(capital, entry_price, atr, atr_mult, direction)

        # Restore
        self.risk_per_trade = original_risk

        return result


class KellyPositionSizer:
    """Kelly Criterion position sizing.

    Uses the formula:
        f* = (p * b - q) / b

    Where:
    - p = win probability
    - q = loss probability (1-p)
    - b = win/loss ratio
    """

    def __init__(
        self,
        fraction: float = 0.25,  # Use 1/4 Kelly for safety
        max_position: float = 0.20,
        min_trades: int = 30,
    ):
        """
        Args:
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position: Maximum position as fraction of capital
            min_trades: Minimum trades needed for estimation
        """
        self.fraction = fraction
        self.max_position = max_position
        self.min_trades = min_trades

    def estimate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Estimate Kelly fraction.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (as fraction)
            avg_loss: Average losing trade (as positive fraction)

        Returns:
            Kelly fraction (can be negative if unprofitable)
        """
        if avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss)

        kelly = (p * b - q) / b

        return kelly

    def calculate_from_history(
        self,
        returns: np.ndarray,
        capital: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate position size from trade history.

        Args:
            returns: Array of trade returns (as fractions)
            capital: Current capital

        Returns:
            (position_fraction, stats_dict)
        """
        returns = np.asarray(returns)
        returns = returns[np.isfinite(returns)]

        if len(returns) < self.min_trades:
            return 0.01, {"error": "insufficient_trades"}

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.01, {"error": "no_wins_or_losses"}

        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        kelly = self.estimate_kelly(win_rate, avg_win, avg_loss)

        # Apply fraction and limits
        position_frac = kelly * self.fraction
        position_frac = max(0, min(self.max_position, position_frac))

        stats = {
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "raw_kelly": float(kelly),
            "adjusted_kelly": float(position_frac),
            "n_trades": len(returns),
        }

        return position_frac, stats


class RiskBudgetManager:
    """Manages risk budget across multiple positions.

    Ensures total portfolio risk stays within limits.
    """

    def __init__(
        self,
        max_total_risk: float = 0.05,  # 5% max portfolio risk
        max_correlated_risk: float = 0.03,  # 3% for correlated assets
        max_positions: int = 5,
    ):
        self.max_total_risk = max_total_risk
        self.max_correlated_risk = max_correlated_risk
        self.max_positions = max_positions
        self.current_positions: Dict[str, float] = {}  # symbol -> risk amount

    def available_risk(self) -> float:
        """Calculate available risk budget."""
        used = sum(self.current_positions.values())
        return max(0, self.max_total_risk - used)

    def can_open_position(
        self,
        symbol: str,
        risk_amount: float,
        correlated_symbols: Optional[list] = None,
    ) -> Tuple[bool, str]:
        """Check if a new position can be opened.

        Returns:
            (can_open, reason)
        """
        if len(self.current_positions) >= self.max_positions:
            return False, f"max_positions_reached ({self.max_positions})"

        available = self.available_risk()
        if risk_amount > available:
            return False, f"insufficient_risk_budget (need {risk_amount:.2%}, have {available:.2%})"

        # Check correlated risk
        if correlated_symbols:
            correlated_risk = sum(
                self.current_positions.get(s, 0)
                for s in correlated_symbols
            )
            if correlated_risk + risk_amount > self.max_correlated_risk:
                return False, f"correlated_risk_exceeded ({correlated_risk + risk_amount:.2%} > {self.max_correlated_risk:.2%})"

        return True, "ok"

    def add_position(self, symbol: str, risk_amount: float):
        """Record a new position."""
        self.current_positions[symbol] = risk_amount

    def remove_position(self, symbol: str):
        """Remove a closed position."""
        self.current_positions.pop(symbol, None)

    def get_status(self) -> Dict[str, float]:
        """Get current risk status."""
        used = sum(self.current_positions.values())
        return {
            "total_risk_used": used,
            "total_risk_available": self.max_total_risk - used,
            "utilization": used / self.max_total_risk if self.max_total_risk > 0 else 0,
            "n_positions": len(self.current_positions),
        }


def calculate_dynamic_atr_mult(
    realized_vol: float,
    lfp: float,
    base_mult: float = 3.0,
) -> float:
    """Calculate dynamic ATR multiplier based on market conditions.

    Args:
        realized_vol: Current realized volatility (annualized)
        lfp: Low Frequency Power ratio (0-1)
        base_mult: Base ATR multiplier

    Returns:
        Adjusted ATR multiplier
    """
    # Vol adjustment: higher vol -> wider stops
    if realized_vol > 0.8:  # Very high vol
        vol_factor = 1.3
    elif realized_vol > 0.5:  # High vol
        vol_factor = 1.15
    elif realized_vol < 0.2:  # Low vol
        vol_factor = 0.9
    else:
        vol_factor = 1.0

    # LFP adjustment: trending -> wider stops, noise -> tighter
    if lfp > 0.6:  # Trending
        lfp_factor = 1.2
    elif lfp < 0.3:  # Noisy
        lfp_factor = 0.85
    else:
        lfp_factor = 1.0

    return round(base_mult * vol_factor * lfp_factor, 1)
