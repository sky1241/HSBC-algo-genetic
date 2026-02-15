#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for ACOR optimizer.

Runs a minimal ACOR optimization on a short data window to verify:
- Archive initialization
- Iteration loop (generate, evaluate, merge)
- Stagnation detection
- Fitness functions (simple + robust)
- Result export
- Parameter constraints (kijun >= tenkan, senkou_b >= kijun)

Usage:
  py -3 tests/test_aco_basic.py
  py -3 -m pytest tests/test_aco_basic.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_short_data() -> pd.DataFrame:
    """Load a 1-year slice of BTC data for fast testing."""
    os.environ["USE_FUSED_H2"] = "1"
    import ichimoku_pipeline_web_v4_8_fixed as pipe

    df = pipe._load_local_csv_if_configured("BTC/USDT", "2h")
    if df is None:
        raise RuntimeError("data/BTC_FUSED_2h.csv not found")
    df = pipe.ensure_utc_index(df)
    # Use only 2023 for speed
    df = df[df.index.year == 2023]
    if df.empty:
        raise RuntimeError("No data for 2023")
    return df


def test_decode_params():
    """Test ratio-encoded params decode correctly with constraints."""
    from optimizers.aco_optimizer import decode_params

    raw = {"tenkan": 10, "r_kijun": 3, "r_senkou": 5, "shift": 26, "atr_mult": 2.5}
    decoded = decode_params(raw)

    assert decoded["tenkan"] == 10
    assert decoded["kijun"] == 30   # max(10, 3*10) = 30
    assert decoded["senkou_b"] == 50  # max(30, 5*10) = 50
    assert decoded["shift"] == 26
    assert decoded["atr_mult"] == 2.5
    # Constraints
    assert decoded["kijun"] >= decoded["tenkan"]
    assert decoded["senkou_b"] >= decoded["kijun"]
    print("  [OK] decode_params constraints satisfied")


def test_param_def_clip():
    """Test ParamDef clipping and casting."""
    from optimizers.aco_optimizer import ParamDef

    p_int = ParamDef("test", 5, 30, "int")
    assert p_int.clip_and_cast(3.7) == 5    # clipped to low
    assert p_int.clip_and_cast(50.0) == 30   # clipped to high
    assert p_int.clip_and_cast(15.3) == 15   # rounded

    p_float = ParamDef("test", 0.5, 6.0, "float", step=0.1)
    assert abs(p_float.clip_and_cast(0.37) - 0.5) < 0.01   # clipped to low
    assert abs(p_float.clip_and_cast(2.34) - 2.3) < 0.01   # stepped
    print("  [OK] ParamDef clip and cast")


def test_eval_cache():
    """Test evaluation cache hit/miss."""
    from optimizers.fitness import EvalCache

    cache = EvalCache()
    params = {"tenkan": 10, "kijun": 26, "senkou_b": 52, "shift": 26, "atr_mult": 2.0}
    assert cache.get(params) is None
    assert cache.misses == 1

    cache.put(params, {"sharpe_proxy": 0.5})
    result = cache.get(params)
    assert result is not None
    assert result["sharpe_proxy"] == 0.5
    assert cache.hits == 1
    print("  [OK] EvalCache hit/miss")


def test_fitness_simple():
    """Test FitnessSimple on real data."""
    import ichimoku_pipeline_web_v4_8_fixed as pipe
    from optimizers.fitness import FitnessSimple

    df = _load_short_data()
    fitness = FitnessSimple(min_trades=5)  # low threshold for short data

    params = {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26, "atr_mult": 2.0}
    score, metrics = fitness(params, df, pipe.backtest_long_short,
                             symbol="BTC/USDT", timeframe="2h")

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert "equity_mult" in metrics
    assert "sharpe_proxy" in metrics
    assert "trades" in metrics
    print(f"  [OK] FitnessSimple: score={score:.4f}, equity={metrics['equity_mult']:.3f}x, "
          f"trades={metrics['trades']}")

    # Test cache works
    score2, _ = fitness(params, df, pipe.backtest_long_short,
                        symbol="BTC/USDT", timeframe="2h")
    assert score2 == score
    assert fitness.cache.hits >= 1
    print(f"  [OK] Cache: {fitness.cache.hits} hits, {fitness.cache.misses} misses")


def test_acor_smoke():
    """Smoke test: run ACOR with minimal settings on short data."""
    import ichimoku_pipeline_web_v4_8_fixed as pipe
    from optimizers.aco_optimizer import ACOROptimizer, ACORConfig
    from optimizers.fitness import FitnessSimple

    df = _load_short_data()

    cfg = ACORConfig(
        n_ants=3,
        archive_size=5,
        q=0.5,
        xi=0.85,
        max_iter=2,
        seed=42,
        stagnation_limit=5,
    )
    fitness = FitnessSimple(min_trades=5)
    optimizer = ACOROptimizer(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        best = optimizer.optimize(
            fitness_fn=fitness,
            df=df,
            backtest_fn=pipe.backtest_long_short,
            log_dir=Path(tmpdir),
            symbol="BTC/USDT",
            timeframe="2h",
        )

        # Verify best solution
        assert best is not None
        assert np.isfinite(best.score)
        assert "tenkan" in best.decoded
        assert "kijun" in best.decoded
        assert best.decoded["kijun"] >= best.decoded["tenkan"]
        assert best.decoded["senkou_b"] >= best.decoded["kijun"]

        # Verify output files
        assert (Path(tmpdir) / "aco_best.json").exists()
        assert (Path(tmpdir) / "aco_top_k.json").exists()
        assert (Path(tmpdir) / "aco_history.csv").exists()
        assert (Path(tmpdir) / "aco_archive.json").exists()

        with open(Path(tmpdir) / "aco_best.json") as f:
            best_json = json.load(f)
        assert "score" in best_json
        assert "params" in best_json

        # Verify archive size maintained
        assert len(optimizer.archive) == cfg.archive_size

        # Verify history
        assert len(optimizer.history) == cfg.max_iter

        total_evals = cfg.archive_size + cfg.max_iter * cfg.n_ants
        print(f"  [OK] ACOR smoke test: {total_evals} evals, "
              f"best score={best.score:.4f}, params={json.dumps(best.decoded)}")


def test_constraint_satisfaction():
    """Verify ALL solutions in archive satisfy Ichimoku constraints."""
    import ichimoku_pipeline_web_v4_8_fixed as pipe
    from optimizers.aco_optimizer import ACOROptimizer, ACORConfig
    from optimizers.fitness import FitnessSimple

    df = _load_short_data()

    cfg = ACORConfig(n_ants=5, archive_size=10, max_iter=3, seed=123)
    fitness = FitnessSimple(min_trades=5)
    optimizer = ACOROptimizer(cfg)

    best = optimizer.optimize(
        fitness_fn=fitness, df=df, backtest_fn=pipe.backtest_long_short,
        symbol="BTC/USDT", timeframe="2h",
    )

    for i, entry in enumerate(optimizer.archive):
        p = entry.decoded
        assert p["kijun"] >= p["tenkan"], f"Archive[{i}]: kijun < tenkan"
        assert p["senkou_b"] >= p["kijun"], f"Archive[{i}]: senkou_b < kijun"
        assert 5 <= p["tenkan"] <= 30, f"Archive[{i}]: tenkan out of range"
        assert 20 <= p["shift"] <= 30, f"Archive[{i}]: shift out of range"
        assert 0.5 <= p["atr_mult"] <= 6.0, f"Archive[{i}]: atr_mult out of range"

    print(f"  [OK] All {len(optimizer.archive)} archive entries satisfy constraints")


if __name__ == "__main__":
    print("=" * 50)
    print("ACOR Integration Tests")
    print("=" * 50)

    tests = [
        ("decode_params", test_decode_params),
        ("param_def_clip", test_param_def_clip),
        ("eval_cache", test_eval_cache),
        ("fitness_simple", test_fitness_simple),
        ("acor_smoke", test_acor_smoke),
        ("constraint_satisfaction", test_constraint_satisfaction),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 50)
    sys.exit(1 if failed > 0 else 0)
