"""Baselines passifs — HODL spot, HODL perp leveraged, DCA hebdo (B24).

Pour répondre à la question "un bot bat-il un HODL passif net de frais sur la
période ?". STRAT-A a montré que HODL spot fait Sharpe 0.798 / CAGR 31.8% sur
2020-2025, dominant largement les algos testés. Ces baselines doivent être
le PREMIER benchmark de toute nouvelle stratégie.

API:
    sim = simulate_hodl_spot(close, fee_bps=4)
    sim_perp = simulate_hodl_perp(close, funding_8h, leverage=1, fee_bps=4)
    sim_dca = simulate_dca(close, freq_bars=84)  # ~1 semaine en H2
    metrics = compute_metrics_for_sim(sim, periods_per_year=4380)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SimulationResult:
    equity: pd.Series       # equity normalisée, commence à 1.0
    returns: pd.Series      # returns par bar
    cash_at_end: float
    n_buys: int
    total_fees_bps: float


def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def simulate_hodl_spot(close: pd.Series, fee_bps: float = 4.0) -> SimulationResult:
    """Buy-and-hold spot: achète à l'open, vend jamais.

    Equity = close / close[0] − 1 (en pct, ajusté de la fee taker à l'entrée).
    """
    close = _ensure_series(close).dropna()
    if len(close) == 0:
        return SimulationResult(pd.Series(dtype=float), pd.Series(dtype=float), 1.0, 0, 0.0)
    fee_pct = fee_bps / 10000.0
    initial_equity = 1.0 - fee_pct  # acheté avec frais
    equity = initial_equity * (close / close.iloc[0])
    returns = equity.pct_change().fillna(equity.iloc[0] - 1.0 if equity.iloc[0] != 0 else 0)
    return SimulationResult(
        equity=equity,
        returns=returns,
        cash_at_end=float(equity.iloc[-1]),
        n_buys=1,
        total_fees_bps=fee_bps,
    )


def simulate_hodl_perp(
    close: pd.Series,
    funding_8h: pd.Series,
    leverage: float = 1.0,
    fee_bps: float = 4.0,
    liquidation_threshold_pct: float = 0.99,
) -> SimulationResult:
    """Long perp leveraged et hold. Inclut funding **uniquement aux bars 00/08/16 UTC**
    et **liquidation** si equity tombe sous threshold (default 1% restant).

    Args:
        funding_8h: rate par 8h. Indexé sur DatetimeIndex UTC. Le funding n'est
            appliqué QU'AUX BARS dont l'heure est 00, 08 ou 16 UTC. Sur les autres
            bars, on ignore (pas de double-comptage).
        leverage: notional ratio.
        liquidation_threshold_pct: si equity descend en dessous de
            (1 - 1/leverage * liquidation_threshold_pct), liquidé → equity = 0
            stable jusqu'à la fin (impossible de re-trade).

    Avec leverage L: returns_position = L * returns_close.
    Funding paid each 8h: cost = L * notional * funding_rate.
    """
    close = _ensure_series(close).dropna()
    f = _ensure_series(funding_8h).reindex(close.index).fillna(0.0)
    if len(close) == 0:
        return SimulationResult(pd.Series(dtype=float), pd.Series(dtype=float), 1.0, 0, 0.0)

    fee_pct = fee_bps / 10000.0
    base_ret = close.pct_change().fillna(0.0)

    # Funding appliqué uniquement aux 3 bars/jour (00/08/16 UTC).
    if isinstance(close.index, pd.DatetimeIndex):
        is_funding_bar = close.index.hour.isin([0, 8, 16])
        funding_per_bar = pd.Series(
            np.where(is_funding_bar, f.values, 0.0),
            index=close.index,
        )
    else:
        # Fallback: pas de tz info, applique à 1 bar sur 4 (proxy H2)
        funding_per_bar = f.copy()
        mask = pd.Series(False, index=close.index)
        mask.iloc[::4] = True
        funding_per_bar = funding_per_bar.where(mask, 0.0)

    # Net return par bar = leverage * (price_return - funding)
    net_ret = leverage * (base_ret - funding_per_bar)

    # Equity avec gestion liquidation
    equity_vals = []
    e = 1.0 - fee_pct
    liq_e = (1.0 / leverage) * (1.0 - liquidation_threshold_pct)
    liquidated = False
    rets_actual = []
    for r in net_ret.values:
        if liquidated:
            equity_vals.append(0.0)
            rets_actual.append(0.0)
            continue
        e_new = e * (1.0 + r)
        if e_new <= liq_e:
            equity_vals.append(0.0)
            rets_actual.append(-1.0)
            liquidated = True
            e = 0.0
        else:
            rets_actual.append(r)
            e = e_new
            equity_vals.append(e)

    equity = pd.Series(equity_vals, index=close.index)
    returns = pd.Series(rets_actual, index=close.index)
    return SimulationResult(
        equity=equity,
        returns=returns,
        cash_at_end=float(equity.iloc[-1]),
        n_buys=1,
        total_fees_bps=fee_bps,
    )


def simulate_dca(
    close: pd.Series,
    freq_bars: int = 84,        # ~1 semaine en H2 (12 bars/jour * 7)
    fee_bps: float = 4.0,
    investment_per_buy: float = 1.0,  # unitaire — l'equity est cumulée
) -> SimulationResult:
    """Dollar-Cost Averaging: achète `investment_per_buy` USD chaque `freq_bars` bars.

    Modèle:
    - À chaque bar de freq_bars: dépense investment_per_buy USD pour acheter
      qty = investment_per_buy * (1 - fee_pct) / close[t]
    - À la fin: total qty held × close[-1] = équity.
    - Cash investi = n_buys × investment_per_buy.
    - Equity normalisée = portfolio_value / cash_invested.
    """
    close = _ensure_series(close).dropna()
    if len(close) == 0:
        return SimulationResult(pd.Series(dtype=float), pd.Series(dtype=float), 1.0, 0, 0.0)
    fee_pct = fee_bps / 10000.0
    qty_held = 0.0
    cash_invested = 0.0
    portfolio_values = []
    n_buys = 0
    for i, price in enumerate(close.values):
        if i % freq_bars == 0:
            qty_held += investment_per_buy * (1.0 - fee_pct) / price
            cash_invested += investment_per_buy
            n_buys += 1
        pv = qty_held * price
        portfolio_values.append(pv / cash_invested if cash_invested > 0 else 1.0)
    equity = pd.Series(portfolio_values, index=close.index)
    returns = equity.pct_change().fillna(0.0)
    return SimulationResult(
        equity=equity,
        returns=returns,
        cash_at_end=float(equity.iloc[-1]),
        n_buys=n_buys,
        total_fees_bps=fee_bps * n_buys,
    )


def compute_metrics_for_sim(sim: SimulationResult, periods_per_year: int = 4380) -> dict:
    """Sharpe, CAGR, MDD pour une SimulationResult."""
    rs = sim.returns.dropna()
    if len(rs) < 2:
        return {"sharpe": float("nan"), "cagr": float("nan"), "mdd": float("nan"),
                "total_return": float("nan")}
    mean = rs.mean()
    std = rs.std(ddof=0)
    sharpe = float(np.sqrt(periods_per_year) * mean / std) if std > 0 else float("nan")
    eq = sim.equity
    final = float(eq.iloc[-1])
    initial = float(eq.iloc[0])
    n_periods = len(rs)
    years = n_periods / periods_per_year if periods_per_year > 0 else float("nan")
    cagr = float((final / initial) ** (1.0 / years) - 1.0) if (initial > 0 and years > 0 and final > 0) else float("nan")
    drawdown = (eq / eq.cummax() - 1.0)
    mdd = float(drawdown.min()) if not drawdown.empty else float("nan")
    total_ret = float(final / initial - 1.0)
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": mdd,
        "total_return": total_ret,
    }


__all__ = [
    "SimulationResult",
    "simulate_hodl_spot",
    "simulate_hodl_perp",
    "simulate_dca",
    "compute_metrics_for_sim",
]
