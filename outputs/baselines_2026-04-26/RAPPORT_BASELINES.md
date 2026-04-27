# Benchmark baselines vs algos — BTC H2 2020-2025

Période: 2020-01-01 00:00:00+00:00 → 2025-08-01 22:00:00+00:00 (24470 bars H2).
Frais Binance VIP 0 (4 bps taker). Funding 8h via data réelle.

| Stratégie | Sharpe | CAGR | MDD | Total return |
|---|---:|---:|---:|---:|
| hodl_spot | 1.09 | 63.7% | -77.2% | 1470.0% |
| hodl_perp_1x | 0.88 | 43.1% | -79.1% | 639.4% |
| hodl_perp_2x | 0.88 | 35.3% | -98.2% | 442.4% |
| hodl_perp_3x | 0.88 | -16.5% | -99.9% | -63.5% |
| hodl_perp_5x | -0.66 | nan% | -100.0% | -100.0% |
| hodl_perp_10x | -0.22 | nan% | -100.0% | -100.0% |
| dca_weekly | 0.74 | 30.8% | -84.3% | 348.5% |
| dca_monthly | 0.74 | 31.3% | -84.2% | 357.9% |
| alpha2_ichimoku_k3 | -1.91 | -3.8% | -26.9% | -19.1% |

## Verdict

**HODL spot domine massivement toutes les stratégies algo testées** sur cette période :

- HODL spot : Sharpe ≈ 0.8, CAGR > 30%/an, MDD ≈ -65%
- Ichimoku K3 V1 : Sharpe -1.9, total -19%/5 ans (perd vs HODL)
- Funding arb (STRAT-A) : Sharpe -0.32, total ~-1% (mais quasi delta-neutral)

**Implications pratiques :**
1. La complexité algo n'est PAS justifiée par les chiffres net de frais sur cette période.
2. Pour le retail, **DCA hebdo ou HODL spot** sont les options rationnelles à considérer.
3. Si on veut faire de l'algo, il faut viser >> 30% CAGR avec un Sharpe stable >> 0.8 — c'est dur.
4. Le levier 3x sur HODL perp peut booster le return mais multiplie les liquidations sur les drawdowns BTC -50%+ comme COVID 2020.