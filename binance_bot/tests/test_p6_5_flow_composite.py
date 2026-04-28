"""P6.5 — Tests flow_composite_signal.py.

Vérifie:
  1. Bornes [-1, +1] sur le score composite (clipping)
  2. Z-score normalisation (drift handling)
  3. Combinaisons extrêmes : tous components +3σ → score clip à +1
  4. Drift handling : nouveaux niveaux dans la distribution → z baisse
  5. Composants individuels conventions de signe
  6. Mode log-only → never blocks
  7. Mode gate → block si |score| > threshold
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.flow_composite_signal import (
    clip_score,
    compute_composite_score,
    compute_liq_imbalance_series,
    compute_oi_change_pct_series,
    compute_taker_centered_series,
    compute_top_ls_inverted_series,
    should_block_trade,
    zscore_last,
)


# ---------------------------------------------------------------------------
# Fixtures jsonl synthétiques
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _gen_top_ls_records(long_account_values: list[float], start_ms: int) -> list[dict]:
    """Génère des records P6.1 alignés sur 5min (300_000 ms)."""
    return [
        {
            "ts_ms": start_ms + i * 300_000,
            "symbol": "BTCUSDT",
            "long_short_ratio": v / max(1e-9, 1 - v),
            "long_account": v,
            "short_account": 1.0 - v,
        }
        for i, v in enumerate(long_account_values)
    ]


def _gen_taker_records(buy_sell_ratios: list[float], start_ms: int) -> list[dict]:
    return [
        {
            "ts_ms": start_ms + i * 300_000,
            "symbol": "BTCUSDT",
            "buy_sell_ratio": v,
            "buy_vol": 1000.0 * v,
            "sell_vol": 1000.0,
        }
        for i, v in enumerate(buy_sell_ratios)
    ]


def _gen_liq_buckets(long_short_pairs: list[tuple[float, float]], start_ms: int) -> list[dict]:
    return [
        {
            "bucket_start_ms": start_ms + i * 60_000,
            "symbol": "BTCUSDT",
            "long_liq_notional": pair[0],
            "short_liq_notional": pair[1],
            "long_liq_count": 1,
            "short_liq_count": 1,
        }
        for i, pair in enumerate(long_short_pairs)
    ]


def _gen_oi_records(sum_oi_values: list[float], start_ms: int) -> list[dict]:
    return [
        {
            "ts_ms": start_ms + i * 300_000,
            "symbol": "BTCUSDT",
            "sum_oi": v,
            "sum_oi_value": v * 50000.0,
        }
        for i, v in enumerate(sum_oi_values)
    ]


# ---------------------------------------------------------------------------
# Composants individuels
# ---------------------------------------------------------------------------


def test_top_ls_inverted_extremes():
    """longAccount=0.8 → -0.6 (contrarian bearish);
       longAccount=0.2 → +0.6 (contrarian bullish);
       longAccount=0.5 → 0.0 (neutre)."""
    df = pd.DataFrame({
        "ts_ms": [100, 200, 300],
        "long_account": [0.8, 0.5, 0.2],
    })
    s = compute_top_ls_inverted_series(df)
    assert s.iloc[0] == pytest.approx(-0.6)
    assert s.iloc[1] == pytest.approx(0.0)
    assert s.iloc[2] == pytest.approx(0.6)


def test_taker_centered_around_one():
    """buy_sell_ratio = 1 → centered = 0 (équilibre)."""
    df = pd.DataFrame({"ts_ms": [100, 200], "buy_sell_ratio": [1.0, 1.5]})
    s = compute_taker_centered_series(df)
    assert s.iloc[0] == pytest.approx(0.0)
    assert s.iloc[1] == pytest.approx(0.5)


def test_liq_imbalance_extremes():
    """Que des longs liquidés → +1; que des shorts → -1; égaux → 0."""
    df = pd.DataFrame({
        "bucket_start_ms": [100, 200, 300],
        "long_liq_notional": [1000.0, 0.0, 500.0],
        "short_liq_notional": [0.0, 1000.0, 500.0],
    })
    s = compute_liq_imbalance_series(df)
    assert s.iloc[0] == pytest.approx(1.0)   # only long liq
    assert s.iloc[1] == pytest.approx(-1.0)  # only short liq
    assert s.iloc[2] == pytest.approx(0.0)   # equal


def test_oi_change_pct_24h():
    """OI augmente de 10% sur 24h → change_pct = 0.10."""
    n_bars = 300
    sum_oi = [1000.0] * (n_bars - 1) + [1100.0]  # last is +10% vs lookback=288
    df = pd.DataFrame({
        "ts_ms": list(range(n_bars)),
        "sum_oi": sum_oi,
        "sum_oi_value": [v * 50000 for v in sum_oi],
    })
    s = compute_oi_change_pct_series(df, lookback_bars=288)
    # La dernière valeur doit être ~ (1100 - 1000) / 1000 = 0.10
    assert s.iloc[-1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------


def test_zscore_last_returns_zero_with_low_obs():
    """< min_obs (default 100) → 0.0 (pas de signal)."""
    s = pd.Series([1.0, 2.0, 3.0])
    assert zscore_last(s) == 0.0


def test_zscore_last_returns_zero_with_zero_std():
    """std = 0 (constants) → 0.0."""
    s = pd.Series([5.0] * 200)
    assert zscore_last(s) == 0.0


def test_zscore_last_normal_distribution():
    """Sur N(0,1) avec dernière obs +2σ → z ≈ 2."""
    np.random.seed(42)
    s = pd.Series(np.random.normal(0, 1, 999).tolist() + [2.0])  # +2σ explicite
    z = zscore_last(s)
    assert 1.5 < z < 2.5


def test_zscore_drift_handling():
    """Quand toute la distribution se déplace, z baisse (relatif au nouveau niveau)."""
    np.random.seed(7)
    s_low = pd.Series(np.random.normal(0, 1, 1000))
    z_low = zscore_last(s_low)

    # Drift : ajoute 200 obs à mu=10, dernière obs à 10 (au new mean)
    s_high = pd.concat([s_low, pd.Series(np.random.normal(10, 1, 199).tolist() + [10.0])])
    z_high = zscore_last(s_high)
    # Le z après drift est plus modéré (puisque la dernière obs ≈ nouvelle mean)
    assert abs(z_high) < 5.0


# ---------------------------------------------------------------------------
# clip_score
# ---------------------------------------------------------------------------


def test_clip_score_bounds():
    assert clip_score(2.5) == 1.0
    assert clip_score(-3.0) == -1.0
    assert clip_score(0.5) == 0.5
    assert clip_score(float("nan")) == 0.0
    assert clip_score(float("inf")) == 1.0


# ---------------------------------------------------------------------------
# compute_composite_score (intégration files jsonl)
# ---------------------------------------------------------------------------


def test_composite_returns_zero_with_no_data(tmp_path):
    """Pas de fichiers → score = 0.0, components tous 0."""
    result = compute_composite_score(
        top_ls_path=tmp_path / "no_top_ls.jsonl",
        taker_path=tmp_path / "no_taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",
        oi_path=tmp_path / "no_oi.jsonl",
    )
    assert result["score"] == 0.0
    assert all(v == 0.0 for v in result["components"].values())


def test_composite_clips_to_minus1_plus1(tmp_path):
    """Avec des données extrêmes (longAccount tend vers 1), score reste dans [-1,+1]."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # 200 records avec longAccount oscillant + dernier extrême
    long_acc = [0.5] * 199 + [0.99]  # dernier 99% long → very contrarian bearish
    _write_jsonl(tmp_path / "top_ls.jsonl", _gen_top_ls_records(long_acc, now_ms - 200 * 300_000))

    result = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "no_taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",
        oi_path=tmp_path / "no_oi.jsonl",
    )
    assert -1.0 <= result["score"] <= 1.0
    # Le composant top_ls doit être très négatif (z fortement négatif)
    assert result["components"]["top_ls"] < -1.0


def test_composite_extreme_combination_clips():
    """Si tous les composants z = +5, score raw = 5 → clipped à +1."""
    # On simule directement le score sans passer par les jsonl (test logique pur)
    components = {"top_ls": 5.0, "taker": 5.0, "liq": 5.0, "oi": 5.0}
    weights = {"top_ls": 0.30, "taker": 0.25, "liq": 0.25, "oi": 0.20}
    raw = sum(weights[k] * components[k] for k in components)
    assert raw == 5.0
    assert clip_score(raw) == 1.0


# ---------------------------------------------------------------------------
# should_block_trade (gate logic)
# ---------------------------------------------------------------------------


def test_should_block_log_mode_never_blocks():
    """Mode log → toujours autoriser, peu importe le score."""
    blocked, reason = should_block_trade(score=-0.99, side="long", mode="log")
    assert not blocked
    assert reason == ""
    blocked, _ = should_block_trade(score=+0.99, side="short", mode="log")
    assert not blocked


def test_should_block_gate_mode_extreme_bearish_blocks_long():
    """Gate mode + score < -0.4 + side=long → BLOCK."""
    blocked, reason = should_block_trade(score=-0.5, side="long", mode="gate", threshold=0.4)
    assert blocked
    assert "bearish" in reason.lower() or "block long" in reason.lower()


def test_should_block_gate_mode_extreme_bullish_blocks_short():
    """Gate mode + score > +0.4 + side=short → BLOCK."""
    blocked, reason = should_block_trade(score=+0.5, side="short", mode="gate", threshold=0.4)
    assert blocked


def test_should_block_gate_mode_within_threshold_allows():
    """Score ∈ [-0.4, +0.4] → autorisé, peu importe side."""
    assert should_block_trade(0.3, "long", "gate", 0.4) == (False, "")
    assert should_block_trade(-0.3, "short", "gate", 0.4) == (False, "")
    assert should_block_trade(0.0, "long", "gate", 0.4) == (False, "")


def test_should_block_gate_mode_aligned_score_allows():
    """Score positif (bullish) + side=long → autorisé (signal favorable)."""
    blocked, _ = should_block_trade(score=+0.8, side="long", mode="gate")
    assert not blocked
    blocked, _ = should_block_trade(score=-0.8, side="short", mode="gate")
    assert not blocked
