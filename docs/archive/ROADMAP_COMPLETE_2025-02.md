# ROADMAP COMPLETE - Février 2025

**Date**: 2025-02-03
**Version**: v5.0 (P1 + P2 + P3 Complet)

---

## Vue d'ensemble

Ce document résume l'ensemble des améliorations implémentées pour le système de trading HSBC-algo-genetic.

### Historique
- **v4.8**: Système original avec Ichimoku + Optuna + WFA annuel
- **v4.9** (P1): Quick wins - ATR optimisé, Calmar ratio, trials augmentés
- **v5.0** (P2+P3): Module spectral complet avec HMM, Monte Carlo, FFT

---

## P1 - Quick Wins ✅

### P1.1 - Réduction ATR_mult
- **Fichier**: `ichimoku_pipeline_web_v4_8_fixed.py`
- **Changement**: `[5.0, 25.0]` → `[2.0, 8.0]`
- **Impact**: +200-300% trades/an

### P1.2 - Contrainte ratio Kijun
- **Changement**: `r_kijun ∈ [1, 5]` → `[2, 4]`
- **Impact**: Cohérence Ichimoku respectée

### P1.3 - Loss Calmar Ratio
- **Fichier**: `scripts/run_scheduler_wfa_phase.py`
- **Changement**: Score = 0.5*Calmar + 0.3*Sharpe + 0.2*CAGR
- **Impact**: Optimisation orientée performance/risque

### P1.4 - Trials augmentés
- **Changement**: 300 → 500 par défaut

### P1.5 - Script de test
- **Fichier**: `scripts/test_p1_quick_validation.py`

---

## P2 - Phase Improvements ✅

### P2.1 - Utilitaires Fourier
- **Fichier**: `src/spectral/fourier_features.py`
- **Features**: PSD, LFP, flatness, entropy, centroid, slope, band powers
- **Usage**:
```python
from src.spectral import compute_spectral_features, detect_regime
features = compute_spectral_features(prices, fs=12.0)
regime = detect_regime(features)  # TREND, MIXED, NOISE
```

### P2.2 - Suggesteur Fourier → Ichimoku
- **Fichier**: `src/spectral/ichimoku_suggester.py`
- **Heuristiques**:
  - kijun ≈ P/2 (demi-période dominante)
  - tenkan ≈ P/6
  - senkou_b ≈ P
  - ATR ajusté selon LFP

### P2.3 - Pools par Régime
- **Fichier**: `src/spectral/param_pools.py`
- **Pools**:
  - `POOL_TREND`: kijun [55,100], ATR [3,5]
  - `POOL_MIXED`: kijun [35,70], ATR [2.5,4]
  - `POOL_NOISE`: kijun [26,55], ATR [2,3]

### P2.4 - Indexation Halving
- **Fichier**: `src/spectral/halving_indexer.py`
- **Phases**:
  - PRE_HALVING: t ∈ [-180j, 0[
  - DISCOVERY: t ∈ [0, +90j]
  - EXPANSION: t ∈ [+90j, +270j]
  - MATURATION: t ∈ [+270j, +540j]
  - LATE_CYCLE: t > +540j

### P2.5 - Monte Carlo Robuste
- **Fichier**: `src/spectral/monte_carlo.py`
- **Métriques**: p5, p25, p50, p75, p95 pour equity, MDD, Calmar
- **Usage**:
```python
from src.spectral import MonteCarloValidator
validator = MonteCarloValidator(n_simulations=1000)
results = validator.run_from_returns(returns)
```

### P2.6 - Walk-Forward Rolling
- **Fichier**: `src/spectral/walk_forward.py`
- **Config**: 12 mois train, 6 mois test, 3 mois step

---

## P3 - Modern Extensions ✅

### P3.1 & P3.2 - HMM Features & Sélection K
- **Fichier**: `src/spectral/hmm_features.py`
- **Features**:
  - Spectrales: LFP, flatness, centroid, entropy
  - Prix: log returns, momentum, RSI
  - Volatilité: ATR, realized vol, skew, kurtosis
  - Ichimoku: cloud distance, TK cross
- **Sélection K**: AIC/BIC automatique

### P3.3 - Position Sizing Dynamique
- **Fichier**: `src/spectral/position_sizing.py`
- **Classes**:
  - `ATRPositionSizer`: Sizing basé sur ATR
  - `KellyPositionSizer`: Kelly criterion
  - `RiskBudgetManager`: Gestion du budget risque

### P3.4 - Accélération FFT
- **Fichier**: `src/spectral/fft_acceleration.py`
- **Fonctions**:
  - `fft_rolling_mean`: O(N log N) vs O(N*W)
  - `fast_ichimoku_lines`: Calcul batch optimisé
  - `BatchBacktester`: Cache des indicateurs

---

## Architecture du Module Spectral

```
src/spectral/
├── __init__.py              # Exports complets
├── fourier_features.py      # P2.1 - PSD, LFP, régimes
├── halving_indexer.py       # P2.4 - Phases halving
├── param_pools.py           # P2.3 - Pools conditionnés
├── ichimoku_suggester.py    # P2.2 - Mapping Fourier→Ichimoku
├── monte_carlo.py           # P2.5 - Validation MC
├── walk_forward.py          # P2.6 - WF rolling
├── hmm_features.py          # P3.1/P3.2 - Features HMM
├── position_sizing.py       # P3.3 - Sizing dynamique
└── fft_acceleration.py      # P3.4 - Optimisation FFT
```

---

## Usage Recommandé

### 1. Analyse Spectrale Rapide
```python
from src.spectral import (
    compute_spectral_features,
    detect_regime,
    get_pool_for_regime,
)

features = compute_spectral_features(prices)
regime = detect_regime(features)
pool = get_pool_for_regime(regime)
params = sample_from_pool(pool)
```

### 2. Walk-Forward Complet
```python
from src.spectral import RollingWalkForward, MonteCarloValidator

wf = RollingWalkForward(train_months=12, test_months=6)
results = wf.run(df, optimize_fn, backtest_fn)

mc = MonteCarloValidator(n_simulations=1000)
mc_results = mc.run_from_returns(returns)
```

### 3. HMM Regime Detection
```python
from src.spectral import HMMFeatureBuilder, select_optimal_k

builder = HMMFeatureBuilder()
feature_set = builder.build(ohlcv_df)
best_k, scores = select_optimal_k(feature_set.df, feature_set.feature_names)
```

---

## Scripts Disponibles

| Script | Description | Usage |
|--------|-------------|-------|
| `test_p1_quick_validation.py` | Test rapide P1 | `python scripts/test_p1_quick_validation.py --trials 100 --use-fused` |
| `run_scheduler_wfa_phase.py` | WFA phase-aware | `python scripts/run_scheduler_wfa_phase.py --labels-csv ... --use-fused` |

---

## Métriques Cibles

| Métrique | Avant (K3) | Objectif P1 | Objectif Final |
|----------|------------|-------------|----------------|
| Monthly return | 0.30% | ≥1.0% | ≥2.0% |
| Max Drawdown | 12.2% | ≤15% | ≤20% |
| Trades/an | 32 | ≥100 | ≥150 |
| Calmar ratio | ~0.25 | ≥0.5 | ≥1.0 |
| Survie | 100% | 100% | 100% |

---

## Prochaines Étapes (P4)

1. **Deep RL**: PPO/A3C pour optimisation continue
2. **Transformer**: Attention mechanism pour séries temporelles
3. **Multi-asset**: Risk parity cross-crypto
4. **Sentiment**: Intégration données on-chain

---

## Références

- `docs/P1_OPTIMISATIONS_2025-02.md` - Détails P1
- `docs/METHODOLOGIE_COMPLETE.md` - Méthodologie WFA
- `docs/IDEES_OPTIMISATION_HALVING_FR.md` - Roadmap Fourier
- `docs/HMM_SPECTRAL_ALGO_FR.md` - Pipeline HMM
- `docs/ANALYSE_COMPLETE_K3_20251022.md` - Baseline K3
