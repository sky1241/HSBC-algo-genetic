"""
Spectral Analysis Module for Trading Strategy Optimization.

P2/P3 Implementation - Complete Fourier/HMM enhancements for phase-aware trading.

Modules:
- fourier_features: PSD, LFP, flatness, entropy, regime detection
- halving_indexer: BTC halving cycle phase alignment
- param_pools: Regime-conditioned parameter search spaces
- ichimoku_suggester: Fourier → Ichimoku parameter mapping
- monte_carlo: Bootstrap validation with percentile statistics
- walk_forward: Rolling window walk-forward analysis
- hmm_features: HMM feature engineering and K selection
- position_sizing: ATR-based dynamic position sizing
- fft_acceleration: O(N log N) rolling calculations

Version: 2.0 (P2+P3 Complete)
Date: 2025-02-03
"""

# P2.1 - Fourier features
from .fourier_features import (
    compute_spectral_features,
    compute_rolling_spectral,
    detect_regime,
    RegimeType,
    SpectralFeatures,
)

# P2.4 - Halving indexer
from .halving_indexer import (
    HalvingIndexer,
    HalvingPhase,
    HALVING_DATES,
    get_halving_phase,
    get_cycle_info,
    add_halving_features,
)

# P2.3 - Parameter pools
from .param_pools import (
    ParamPool,
    POOL_TREND,
    POOL_MIXED,
    POOL_NOISE,
    get_pool_for_regime,
    get_pool_for_halving_phase,
    sample_from_pool,
    blend_pools,
)

# P2.2 - Ichimoku suggester
from .ichimoku_suggester import (
    suggest_params_from_spectrum,
    FourierIchimokuMapper,
    SuggestedParams,
    generate_baseline_json,
)

# P2.5 - Monte Carlo validation
from .monte_carlo import (
    MonteCarloValidator,
    MCResults,
    aggregate_seed_results,
    worst_case_decision,
)

# P2.6 - Walk-forward analysis
from .walk_forward import (
    RollingWalkForward,
    WFWindow,
    WFResult,
    WFSummary,
    make_monthly_folds,
    make_annual_folds,
)

# P3.1 & P3.2 - HMM features and K selection
from .hmm_features import (
    HMMFeatureBuilder,
    HMMFeatureSet,
    fit_hmm,
    select_optimal_k,
    predict_states,
    interpret_states,
)

# P3.3 - Position sizing
from .position_sizing import (
    ATRPositionSizer,
    KellyPositionSizer,
    RiskBudgetManager,
    PositionSize,
    calculate_dynamic_atr_mult,
)

# P3.4 - FFT acceleration
from .fft_acceleration import (
    fft_rolling_mean,
    fft_rolling_max,
    fft_rolling_min,
    fast_ichimoku_lines,
    fast_atr,
    BatchBacktester,
    vectorized_ichimoku_signals,
)

__all__ = [
    # P2.1 - Fourier features
    "compute_spectral_features",
    "compute_rolling_spectral",
    "detect_regime",
    "RegimeType",
    "SpectralFeatures",

    # P2.4 - Halving indexer
    "HalvingIndexer",
    "HalvingPhase",
    "HALVING_DATES",
    "get_halving_phase",
    "get_cycle_info",
    "add_halving_features",

    # P2.3 - Parameter pools
    "ParamPool",
    "POOL_TREND",
    "POOL_MIXED",
    "POOL_NOISE",
    "get_pool_for_regime",
    "get_pool_for_halving_phase",
    "sample_from_pool",
    "blend_pools",

    # P2.2 - Ichimoku suggester
    "suggest_params_from_spectrum",
    "FourierIchimokuMapper",
    "SuggestedParams",
    "generate_baseline_json",

    # P2.5 - Monte Carlo
    "MonteCarloValidator",
    "MCResults",
    "aggregate_seed_results",
    "worst_case_decision",

    # P2.6 - Walk-forward
    "RollingWalkForward",
    "WFWindow",
    "WFResult",
    "WFSummary",
    "make_monthly_folds",
    "make_annual_folds",

    # P3.1 & P3.2 - HMM
    "HMMFeatureBuilder",
    "HMMFeatureSet",
    "fit_hmm",
    "select_optimal_k",
    "predict_states",
    "interpret_states",

    # P3.3 - Position sizing
    "ATRPositionSizer",
    "KellyPositionSizer",
    "RiskBudgetManager",
    "PositionSize",
    "calculate_dynamic_atr_mult",

    # P3.4 - FFT acceleration
    "fft_rolling_mean",
    "fft_rolling_max",
    "fft_rolling_min",
    "fast_ichimoku_lines",
    "fast_atr",
    "BatchBacktester",
    "vectorized_ichimoku_signals",
]
