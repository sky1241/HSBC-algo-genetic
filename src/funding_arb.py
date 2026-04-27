"""Funding rate arbitrage delta-neutral strategy on Binance perpetuals.

Idea (long-spot/short-perp leg only — conservative):
- When funding > +threshold, longs pay shorts every 8h. We open
  SHORT perp + LONG spot (delta-neutral) and collect the funding.
- When funding < -threshold, the inverse trade requires shorting spot
  (margin spot, expensive borrow). We OPTIONALLY take the +1 signal,
  but the simulator does NOT execute it by default — set
  ``allow_short_spot=True`` to include it (with explicit borrow cost).
- Otherwise we stay flat.

Pivot post-ALPHA-2 (cf RAPPORT_ALPHA_FINAL_2026-04-26.md): Ichimoku K3 nu
Sharpe -1.91, on cherche un edge crypto-spécifique. Funding carry est
Sharpe 3-6 dans la littérature mais l'edge mean-revert vite (95% des
opportunités). Cette implémentation modélise honnêtement les frais taker
× 4 (open spot + open perp + close spot + close perp) et le borrow APR
si on accepte le côté short-spot.

Références:
- He & Manela (2022) "Fundamentals of Perpetual Futures"
- BIS WP 1087 (2023) "Crypto Carry"
- Binance funding history empirique (cf data/funding_rate_BTCUSDT.csv)

Note: ce module est PUR (pas de side-effects, pas d'I/O), il peut être
utilisé en backtest WFA ou plug & play dans une CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Binance pays funding at 00, 08, 16 UTC. On H2 bars these are the bar
# CLOSING at exactly 00:00 / 08:00 / 16:00 (since H2 closes are even hours).
FUNDING_HOURS_UTC = (0, 8, 16)

# Fees: 4 open legs (spot taker + perp taker) + 4 close legs.
# 4 fills × taker_bps each.
N_LEGS_ROUND_TRIP = 4

# 8h periods per year for annualization of funding.
FUNDING_PERIODS_PER_YEAR = 365 * 3  # 1095

# H2 periods per year for Sharpe annualization.
H2_PERIODS_PER_YEAR = 365 * 12  # 4380


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


def funding_arb_signal(
    funding_h8: pd.Series,
    threshold_bps: float = 5.0,
) -> pd.Series:
    """Return signal aligned to funding_h8 index.

    -1 → SHORT perp + LONG spot (collect positive funding).
    +1 → LONG perp + SHORT spot (collect negative funding; expensive).
     0 → flat.

    Args:
        funding_h8: funding rate series (as a fraction, e.g. 0.0001 = 1bp/8h).
        threshold_bps: absolute threshold in bps per 8h. 5 bps = 0.05%/8h ~= 55%/an.

    Returns:
        Series int8 in {-1, 0, +1}.
    """
    if not isinstance(funding_h8, pd.Series):
        raise TypeError("funding_h8 must be a pandas Series")
    threshold = float(threshold_bps) * 1e-4
    sig = pd.Series(0, index=funding_h8.index, dtype="int8")
    sig[funding_h8 > threshold] = -1
    sig[funding_h8 < -threshold] = 1
    return sig


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FundingArbResult:
    """Output of simulate_funding_arb."""

    equity_curve: pd.Series                # cumulative pnl multiplier (start=1.0)
    returns: pd.Series                     # per-bar net return
    funding_pnl: pd.Series                 # funding cashflow per bar (fraction)
    fees_pnl: pd.Series                    # fee cashflow per bar (fraction, <=0)
    borrow_pnl: pd.Series                  # spot-borrow cost per bar (fraction, <=0)
    position: pd.Series                    # signal held per bar (-1/0/+1)
    n_trades: int                          # number of round-trips executed
    total_funding_received: float          # sum of funding_pnl
    total_fees: float                      # sum of fees_pnl (<=0)
    total_borrow: float                    # sum of borrow_pnl (<=0)
    sharpe: float                          # annualized Sharpe of returns
    metadata: dict = field(default_factory=dict)


def _is_funding_bar(ts: pd.Timestamp) -> bool:
    return ts.hour in FUNDING_HOURS_UTC and ts.minute == 0


def simulate_funding_arb(
    df: pd.DataFrame,
    funding_8h: pd.Series,
    fee_taker_bps: float = 4.0,
    spot_short_borrow_apr: float = 0.05,
    threshold_bps: float = 5.0,
    holding_min_periods: int = 1,
    allow_short_spot: bool = False,
) -> FundingArbResult:
    """Simulate funding-arb delta-neutral strategy bar-by-bar.

    Convention de signe: signal -1 = short perp + long spot. Quand funding > 0
    et qu'on est short perp, on REÇOIT funding (pnl positif). Inversely si
    signal +1 et funding < 0.

    À chaque H2 bar:
    1. Si on a un signal change vs précédent, on FERME l'ancienne position
       (close spot + close perp = 2 taker fees) et on OUVRE la nouvelle
       (open spot + open perp = 2 taker fees). Sinon, hold.
    2. Si la barre tombe sur un funding event (00/08/16 UTC), on encaisse
       (signal == -1 et funding > 0) ou paye (signal == -1 et funding < 0)
       le funding rate × notional. Pour signal == +1 c'est le miroir.
    3. Si signal == +1 (short spot), on paye spot_short_borrow_apr prorata
       sur la durée de la barre (2h).

    Le delta-neutral suppose que le spot et le perp tracent. En pratique
    il existe un basis qui fluctue — ignoré ici pour simplifier (premier-ordre).

    Args:
        df: DataFrame OHLCV avec DatetimeIndex H2.
        funding_8h: funding aligné sur l'index de df (méthode ffill).
            Convention: funding fractionnel (1e-4 = 1bp).
        fee_taker_bps: fee taker par leg (Binance VIP 0 = 4 bps = 0.04%).
        spot_short_borrow_apr: APR de borrow spot pour côté short-spot.
        threshold_bps: seuil d'entrée en bps par 8h.
        holding_min_periods: nombre min de barres à hold avant de fermer
            (pour amortir 16 bps round-trip frais). 1 = pas de filtre.
        allow_short_spot: si False, ignore signal +1 (trop coûteux pour retail).

    Returns:
        FundingArbResult.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if not df.index.equals(funding_8h.index):
        raise ValueError("df and funding_8h must share the same index")
    if fee_taker_bps < 0:
        raise ValueError("fee_taker_bps must be >= 0")
    if threshold_bps < 0:
        raise ValueError("threshold_bps must be >= 0")
    if holding_min_periods < 1:
        raise ValueError("holding_min_periods must be >= 1")

    n = len(df)
    fee_per_leg = float(fee_taker_bps) * 1e-4
    fee_round_trip = N_LEGS_ROUND_TRIP * fee_per_leg

    # Pre-compute signal
    raw_signal = funding_arb_signal(funding_8h, threshold_bps=threshold_bps)
    if not allow_short_spot:
        raw_signal = raw_signal.where(raw_signal != 1, 0)

    # Per-bar borrow accrual: APR over 2h ≈ APR * (2/8760)
    borrow_per_bar = float(spot_short_borrow_apr) * (2.0 / (365.0 * 24.0))

    position = np.zeros(n, dtype=np.int8)
    funding_pnl = np.zeros(n, dtype=np.float64)
    fees_pnl = np.zeros(n, dtype=np.float64)
    borrow_pnl = np.zeros(n, dtype=np.float64)

    cur_pos = 0          # position en début de barre
    bars_held = 0        # barres tenues consécutives
    sig_arr = raw_signal.to_numpy(dtype=np.int8, copy=False)
    fund_arr = funding_8h.to_numpy(dtype=np.float64, copy=False)
    idx = df.index
    n_trades = 0

    for i in range(n):
        target = sig_arr[i]
        if cur_pos != 0 and bars_held < holding_min_periods:
            # On force le hold même si signal change.
            target = cur_pos

        if target != cur_pos:
            # Switch de position. Frais payés instantanément.
            # Si on était à 0, on n'a que des frais d'OUVERTURE (2 legs = open spot + open perp).
            # Si on était != 0 et target != 0, on a CLOSE+OPEN = 4 legs.
            # Si target == 0, on a CLOSE = 2 legs.
            if cur_pos == 0:
                fees_pnl[i] -= 2 * fee_per_leg
                n_trades += 1  # on compte une nouvelle entrée
            elif target == 0:
                fees_pnl[i] -= 2 * fee_per_leg  # close
            else:
                fees_pnl[i] -= 4 * fee_per_leg  # close + open (round-trip + new open)
                n_trades += 1
            cur_pos = int(target)
            bars_held = 0

        # Funding cashflow: à 00/08/16 UTC.
        ts = idx[i]
        if cur_pos != 0 and _is_funding_bar(ts):
            f = fund_arr[i]
            if not np.isnan(f):
                # Si cur_pos == -1 (short perp), longs payent shorts ⇒ on reçoit f.
                # Si cur_pos == +1 (long perp), shorts payent longs ⇒ on reçoit -f.
                funding_pnl[i] += -cur_pos * f

        # Borrow cost (uniquement si short spot, càd cur_pos == +1).
        if cur_pos == 1:
            borrow_pnl[i] -= borrow_per_bar

        position[i] = cur_pos
        if cur_pos != 0:
            bars_held += 1

    returns_arr = funding_pnl + fees_pnl + borrow_pnl
    returns_series = pd.Series(returns_arr, index=df.index, name="ret")
    equity = (1.0 + returns_series).cumprod()
    equity.name = "equity"

    # Sharpe annualisé H2.
    std = returns_series.std(ddof=0)
    if std > 0:
        sharpe = float(np.sqrt(H2_PERIODS_PER_YEAR) * returns_series.mean() / std)
    else:
        sharpe = float("nan")

    return FundingArbResult(
        equity_curve=equity,
        returns=returns_series,
        funding_pnl=pd.Series(funding_pnl, index=df.index, name="funding"),
        fees_pnl=pd.Series(fees_pnl, index=df.index, name="fees"),
        borrow_pnl=pd.Series(borrow_pnl, index=df.index, name="borrow"),
        position=pd.Series(position, index=df.index, name="position"),
        n_trades=int(n_trades),
        total_funding_received=float(funding_pnl.sum()),
        total_fees=float(fees_pnl.sum()),
        total_borrow=float(borrow_pnl.sum()),
        sharpe=sharpe,
        metadata={
            "fee_taker_bps": fee_taker_bps,
            "threshold_bps": threshold_bps,
            "holding_min_periods": holding_min_periods,
            "allow_short_spot": allow_short_spot,
            "spot_short_borrow_apr": spot_short_borrow_apr,
            "n_bars": n,
            "n_bars_long_spot": int((position == -1).sum()),
            "n_bars_short_spot": int((position == 1).sum()),
            "n_bars_flat": int((position == 0).sum()),
        },
    )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def align_funding_to_h2(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.Series:
    """Reindex funding (8h cadence) to H2 bars via forward-fill.

    No lookahead: a funding rate published at t is only known at t+, so we
    ffill forward (each H2 bar uses the latest funding ANNOUNCED ≤ bar).
    """
    if "funding_rate" not in funding_df.columns:
        raise ValueError("funding_df must have a 'funding_rate' column")
    if "timestamp" in funding_df.columns:
        funding = funding_df.set_index("timestamp")["funding_rate"]
    else:
        funding = funding_df["funding_rate"]
    funding.index = pd.to_datetime(funding.index)
    funding = funding.sort_index()
    aligned = funding.reindex(ohlcv_df.index, method="ffill")
    aligned.name = "funding_rate"
    return aligned


__all__ = [
    "FUNDING_HOURS_UTC",
    "FUNDING_PERIODS_PER_YEAR",
    "H2_PERIODS_PER_YEAR",
    "FundingArbResult",
    "funding_arb_signal",
    "simulate_funding_arb",
    "align_funding_to_h2",
]
