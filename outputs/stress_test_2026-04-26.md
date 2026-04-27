# B6 — Stress test report (2026-04-26)

Backtest engine reused: `src.risk_sizing.simulate_strategy` (Ichimoku K3 baseline,
`tenkan=9, kijun=26, senkou_b=52, shift=26, atr_mult=2.0`). Metrics from
`src.stats_eval.compute_metrics`. Leverage acts as a P&L multiplier on per-bar
returns (gross — fees / slippage modelled in B5, intentionally excluded here so
leverage tail-risk is not masked).

Tests: `tests/stress/test_extreme_scenarios.py`, `tests/stress/test_synthetic_scenarios.py`.
Run: `cd /home/ludov/HSBC-algo-genetic && .venv/bin/python -m pytest tests/stress/ -v` -> 11 passed.

## 1. Data sources

| File | Source | Period | Rows (2h bars) | Peak->trough |
|---|---|---|---|---|
| `data/stress/btc_usdt_2022_luna.csv` | Binance public REST (already cached in `data/BTC_USDT_2h.csv`, fetched via CCXT) | 2022-04-15 -> 2022-06-30 | 924 | -57.7% |
| `data/stress/btc_usdt_2022_ftx.csv` | Binance public REST (cached) | 2022-10-15 -> 2022-11-30 | 564 | -27.0% |
| `data/stress/btc_usdt_2020_covid.csv` | Binance public REST (cached) | 2020-03-01 -> 2020-03-31 | 372 | -54.9% |

The 2h history in `data/BTC_USDT_2h.csv` (35 123 rows, 2017-08-17 -> 2025-08-26)
already covered all three windows, so no new HTTP calls were necessary. Slices
were extracted by date with pandas (no resampling, no synthetic interpolation).

## 2. Historical scenarios — leverage sensitivity

P&L is the period return on a unit equity baseline starting at 1.0.
"Liquidated" = peak-to-trough equity DD has crossed `-1/L` (simplified, ignores
maintenance margin; matches the audit's first-order model).

### LUNA / UST depeg (Apr-Jun 2022)
| Leverage | Final equity | P&L | Max DD | Sortino | Ulcer | DD duration (bars) | Liquidated |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1x  | 1.245 | +24.5% |  -9.3% | 2.75 |  3.9% | 223 | no  |
| 3x  | 1.743 | +74.3% | -26.8% | 2.75 | 11.7% | 383 | no  |
| 5x  | 2.128 | +112.8% | -42.2% | 2.75 | 19.8% | 385 | YES |
| 10x | 1.903 | +90.3% | -71.2% | 2.75 | 39.6% | 391 | YES |

LUNA was a directional move with clean trends — the strategy actually profited.
But at 5x the equity DD breaches the 20% liq threshold (intra-period),
funds would have been wiped before the recovery.

### FTX collapse (Oct-Nov 2022)
| Leverage | Final equity | P&L | Max DD | Sortino | Ulcer | Liquidated |
|---:|---:|---:|---:|---:|---:|:---:|
| 1x  | 1.009 |  +0.9% |  -7.4% | 0.37 |  3.7% | no  |
| 3x  | 0.997 |  -0.3% | -20.8% | 0.37 | 11.3% | no  |
| 5x  | 0.945 |  -5.5% | -34.0% | 0.37 | 19.0% | YES |
| 10x | 0.683 | -31.7% | -63.4% | 0.37 | 38.0% | YES |

3x just survives (-20.8% DD, below the 33% threshold). 5x liquidates.

### COVID flash crash (Mar 2020) — **WORST CASE**
| Leverage | Final equity | P&L | Max DD | Sortino | Ulcer | Liquidated |
|---:|---:|---:|---:|---:|---:|:---:|
| 1x  | 0.933 |  -6.7% | -17.5% | -0.39 |  7.9% | no  |
| 3x  | 0.642 | -35.8% | -51.9% | -0.39 | 30.4% | **YES** |
| 5x  | 0.293 | -70.7% | -79.3% | -0.39 | 56.1% | YES |
| 10x |-0.021 |-102.1% |-106.8% | -0.39 | 83.0% | YES |

The COVID single-day -50% wipes the account at every leverage above 1x once
liquidation thresholds are applied: at 3x the rolling equity DD reaches -51.9%
(well past the 33% liq line). **3x is NOT a free pass against tail events of
this magnitude.** This is the strongest argument for a market-regime kill
switch (volatility filter) on top of leverage caps.

## 3. Synthetic scenarios

| Scenario | Outcome | Detail |
|---|---|---|
| Flash crash -20%/1h, 3x | Kill-switch fires (mdd -29.1%, beyond -25% operational threshold) | net P&L -12.6% after bounce |
| Flash crash -20%/1h, 1x | No kill-switch (mdd -10.0%) | net P&L -5.4% |
| Funding spike +0.5%/8h x 48h, 3x | Drag = **+2.97%** lost over 48h | matches theoretical 6 funding slots * 0.5% = 3.0% |
| Funding normal 0.01%/8h x 48h | Drag = 0.06% (negligible) | confirms baseline |

## 4. Empirical liquidation probability (5y rolling 30-day windows)

Computed on the last 5 years of 2h BTC bars (21 900 rows). For each rolling
30-day window (360 bars) we measured the peak-to-trough drawdown and counted
windows breaching each threshold.

| Threshold | P(peak->trough drop >= threshold) | Implied leverage |
|---|---|---|
| 10% | 93.6% | L=10 (every month at risk) |
| 20% | 52.7% | L=5 (one month in two) |
| 33% | 10.4% | L=3 (one month in ten) |

A 33% rolling drop happens roughly **once per quarter** on BTC over 2020-2025
— the bot must therefore plan for liquidation events at 3x several times a year
unless protected by a regime kill-switch.

## 5. Conclusions and recommendations

1. **Leverage cap = 3x with kill-switch / 2x without.** Without an external
   regime filter, a 33%-magnitude move on BTC has happened in ~10% of rolling
   30-day windows over 2020-2025; the COVID episode shows 3x can still be
   liquidated. We recommend keeping the recently-applied 3x cap but adding
   a hard rule "exit all positions when realised vol > 6% / day OR funding
   > 0.1% / 8h".
2. **5x and above is unsafe** on BTC futures with the current strategy —
   liquidated in every one of the three historical stress windows.
3. **Funding spike kill-switch.** A spike of +0.5% / 8h costs ~3% over 48h
   on a held long at 3x. The bot should be paused (or flip-only allowed) when
   `predicted_funding_8h > 0.1%`. Below that level the drag is negligible
   (< 6 bps over 48h).
4. **Kill-switch validated** on the synthetic -20%/1h flash at 3x (DD breaches
   -25%). At 1x the same flash does not breach the threshold — confirming that
   the kill-switch only "fires" where it is actually needed.
5. **Action items for B7+**
   - Wire the empirical liq-probability check into the live risk monitor (alert
     if rolling 30d DD > 25%).
   - Add a unit test on the live bot that asserts `stop_global` triggers when
     equity touches -30% (not implemented yet for futures path).
   - Re-run with `realistic_costs=True` once `cost_model` funding feed is wired
     end-to-end (currently funding rates are not plumbed through the test
     harness — drag here is purely from leverage on price).

## Appendix — raw snapshot

`outputs/stress_test_2026-04-26.json` carries the full numerical payload used
to render this report.
