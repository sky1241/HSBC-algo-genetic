"""P0bis — Tests du branchement cost_model dans le pipeline 14 ans.

Vérifie que `_round_trip_fee_cost_usdt` (helper module-level dans
`ichimoku_pipeline_web_v4_8_fixed.py`) :
  1. Calcule le bon montant de frais round-trip Binance VIP0 futures
  2. Donne 0 sur notional <= 0
  3. Match le module canonique src/cost_model.py
  4. Module correctement le mix taker/maker via maker_fill_ratio
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Permettre l'import du pipeline depuis tests/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

PIPELINE_PATH = ROOT / "ichimoku_pipeline_web_v4_8_fixed.py"


@pytest.fixture(scope="module")
def pipeline():
    """Charge le pipeline comme module pour accéder au helper privé."""
    spec = importlib.util.spec_from_file_location(
        "_pipeline_under_test", str(PIPELINE_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests obligatoires (spec P0bis)
# ---------------------------------------------------------------------------


def test_pipeline_fees_applied_correct_amount(pipeline):
    """1 trade BTC notional $1000, taker pur, fee VIP0 4 bps × 2.

    expected_fee_usd = 1000 × 4e-4 × 2 = 0.80 USDT
    """
    notional = 1000.0
    fees = pipeline.BinanceFutureFees()
    cost = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=0.0)
    expected = notional * 0.0004 * 2  # = 0.80
    assert abs(cost - expected) < 1e-9, (
        f"P0bis: round-trip taker pur incorrect. Attendu={expected}, reçu={cost}"
    )


def test_pipeline_fees_zero_when_notional_zero(pipeline):
    """notional = 0 → coût = 0 (bypass cost_model qui raise sur < 0)."""
    fees = pipeline.BinanceFutureFees()
    assert pipeline._round_trip_fee_cost_usdt(0.0, fees) == 0.0
    # Negative bypass: helper retourne 0 même si cost_model raise sur direct
    assert pipeline._round_trip_fee_cost_usdt(-100.0, fees) == 0.0


def test_pipeline_fees_match_cost_model_module(pipeline):
    """Helper pipeline = somme stricte de 2 appels cost_model.compute_trade_cost.

    Sanity check : pas de divergence numérique entre les deux systèmes.
    """
    from cost_model import BinanceFutureFees, compute_trade_cost

    notional = 12_345.67
    fees = BinanceFutureFees()
    # cost_model retourne valeur NÉGATIVE (cost from PnL view)
    direct_taker = abs(compute_trade_cost(notional, is_taker=True, fees=fees))
    expected_round_trip = 2.0 * direct_taker  # entry + exit

    via_helper = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=0.0)
    assert abs(via_helper - expected_round_trip) < 1e-9


def test_pipeline_fees_with_maker_ratio_50_50(pipeline):
    """maker_fill_ratio=0.5 → fees = 0.5 × maker_2bps + 0.5 × taker_4bps = 3 bps moyens.

    Round-trip = 6 bps × notional. Sur $1000 → $0.60.
    """
    notional = 1000.0
    fees = pipeline.BinanceFutureFees()
    cost_50_50 = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=0.5)
    expected = notional * 0.0006  # 6 bps round-trip
    assert abs(cost_50_50 - expected) < 1e-9, (
        f"P0bis: mix 50/50 incorrect. Attendu={expected}, reçu={cost_50_50}"
    )


# ---------------------------------------------------------------------------
# Tests complémentaires (robustesse helper)
# ---------------------------------------------------------------------------


def test_pipeline_fees_pure_maker_4bps_round_trip(pipeline):
    """maker_fill_ratio=1.0 → 100% maker = 2 bps × 2 = 4 bps round-trip."""
    notional = 1000.0
    fees = pipeline.BinanceFutureFees()
    cost = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=1.0)
    expected = notional * 0.0004  # 4 bps round-trip (2bps × 2)
    assert abs(cost - expected) < 1e-9


def test_pipeline_fees_clamp_ratio_above_one(pipeline):
    """maker_fill_ratio>1 doit être clampé à 1.0 (= 100% maker)."""
    notional = 1000.0
    fees = pipeline.BinanceFutureFees()
    cost_clamped = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=2.5)
    cost_max_maker = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=1.0)
    assert abs(cost_clamped - cost_max_maker) < 1e-9


def test_pipeline_fees_clamp_ratio_below_zero(pipeline):
    """maker_fill_ratio<0 doit être clampé à 0.0 (= 100% taker)."""
    notional = 1000.0
    fees = pipeline.BinanceFutureFees()
    cost_clamped = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=-0.3)
    cost_max_taker = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=0.0)
    assert abs(cost_clamped - cost_max_taker) < 1e-9


def test_pipeline_fees_legacy_spot_was_more_expensive(pipeline):
    """Comparaison : nouveau (8 bps) < ancien spot (20 bps) → strict inégalité.

    Garantit que P0bis a bien CORRIGÉ (pas dégradé) le drag par trade.
    Le drag legacy = notional × 0.001 × 2 = 20 bps round-trip.
    """
    notional = 10_000.0
    fees = pipeline.BinanceFutureFees()
    new_cost = pipeline._round_trip_fee_cost_usdt(notional, fees, maker_fill_ratio=0.0)
    legacy_spot_cost = notional * 0.001 * 2  # = 20 USDT
    assert new_cost < legacy_spot_cost, (
        f"P0bis n'a pas réduit le drag : nouveau={new_cost}, legacy={legacy_spot_cost}"
    )
    # Sanity quantitative : le ratio nouveau/legacy doit être 8/20 = 0.4
    assert abs(new_cost / legacy_spot_cost - 0.4) < 1e-9


def test_pipeline_fees_signature_backward_compatible(pipeline):
    """Les 2 fonctions backtest acceptent maker_fill_ratio absent (default 0.0)."""
    import inspect

    sig_long_short = inspect.signature(pipeline.backtest_long_short)
    assert "maker_fill_ratio" in sig_long_short.parameters
    assert sig_long_short.parameters["maker_fill_ratio"].default == 0.0

    sig_shared = inspect.signature(pipeline.backtest_shared_portfolio)
    assert "maker_fill_ratio" in sig_shared.parameters
    assert sig_shared.parameters["maker_fill_ratio"].default == 0.0
