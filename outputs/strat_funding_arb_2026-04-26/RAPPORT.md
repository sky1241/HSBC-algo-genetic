# RAPPORT — Funding rate arb delta-neutral (STRAT-A) — 2026-04-26

> Pivot post-ALPHA-2 (Ichimoku K3 nu Sharpe -1.91 réfutée). Cf RAPPORT_ALPHA_FINAL_2026-04-26.md.

## TL;DR

- **Décision: NE_PAS_DEPLOYER** (confiance HAUTE)
- Sharpe chained OOS -0.32 <= 0, edge inexistant après frais.
- Best threshold: **3.0 bps/8h**
- Sharpe chained OOS: **-0.317**
- DSR (n_trials=5): **0.003**
- Hansen SPA p_consistent: **1.000**
- HODL spot Sharpe (baseline): 0.798

## Configuration

- Period: 2020-01-01 00:00:00 → 2025-08-01 22:00:00 (24470 bars H2)
- OOS chained obs: 14680
- Folds: 5 (train_frac=0.4)
- Thresholds testés (bps/8h): [3.0, 5.0, 7.0, 10.0, 15.0]
- Fee taker: 4.0 bps × leg (4 legs round-trip = 16.0 bps)
- Spot borrow APR: 5.0%
- allow_short_spot: False
- holding_min_periods: 1

## Funding rate stats sur la période

- Mean: 1.223 bps / 8h (annualisé ≈ 13.4 %/an)
- Median: 1.000 bps / 8h
- p95: 5.319 bps / 8h

## Métriques chained OOS par threshold

| thr (bps) | Sharpe | Sortino | MDD | Ulcer | CAGR | Total ret |
|-----------|--------|---------|------|-------|------|-----------|
|      3.0 | -0.317 | -0.025 | -1.415% | 0.0070 | -0.113% | 1 |
|      5.0 | -1.391 | -0.074 | -1.039% | 0.0064 | -0.305% | 1 |
|      7.0 | -1.060 | -0.032 | -0.417% | 0.0024 | -0.125% | 1 |
|     10.0 | nan | nan | 0.000% | 0.0000 | 0.000% | 1 |
|     15.0 | nan | nan | 0.000% | 0.0000 | 0.000% | 1 |

## Baselines (mêmes barres OOS)

| stratégie | Sharpe | MDD | CAGR |
|-----------|--------|------|------|
| HODL spot | 0.798 | -67.367% | 31.843% |
| HODL perp long | 0.647 | -68.142% | 22.101% |
| ALPHA-2 V1 (réfutée)  | -1.91 | n/a | n/a |

## Hansen SPA (K thresholds vs zero-return benchmark)

- Test statistic V = 0.000
- p_value_consistent (recommandé): **1.000**
- p_value_lower (conservateur): 1.000
- p_value_upper (libéral): 1.000
- Best k = threshold 10.0 bps
- N bootstrap = 1000

## Limitations / risques

1. **Basis risk ignoré**: spot et perp ne tracent pas parfaitement (basis fluctue ±50bps en stress).
2. **Funding squeeze**: si la stratégie est crowded, le funding revient à 0 vite (mean reversion 95%).
3. **Slippage modélisé seulement via fee taker × 4**: pas d'impact prix sur entry/exit.
4. **Spot+perp hedge requires 2× capital ou cross-margin** (pas modélisé en sizing).
5. **Funding empirique BTC ~1bp/8h** ≈ 11%/an mais après 16bps round-trip frais, 
   il faut hold ≥ 8 events (~3 jours) pour amortir si on entre/sort à chaque flip.
6. **Pas de walk-forward sur le threshold** (threshold fixé pour tous les folds): si on optimise
   le threshold dans chaque fold train, on data-snoop encore plus.
7. **2020-2025 inclut le bull 2021 + le crash 2022 + le bull 2024**: funding peak +100bps/8h en jan
   2021 puis -50bps/8h en mai 2021. La distribution est fortement non-stationnaire.

## Verdict

**NE_PAS_DEPLOYER** — Sharpe chained OOS -0.32 <= 0, edge inexistant après frais.

Voir `recommendation.json` pour la décision machine-readable et `equity_curves.png`.