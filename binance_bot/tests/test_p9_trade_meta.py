"""P9 — Tests trade meta-labeling (LdP 2018 ch. 3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.audit_log import AuditLog
from bot.trade_meta import (
    MetaLabelLogger,
    build_meta_label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides) -> dict:
    base = dict(
        trade_id="trade_001",
        timestamp_open=1700000000000,
        timestamp_close=1700003600000,
        symbol="BTCUSDT",
        side="LONG",
        entry_price=42000.0,
        exit_price=42500.0,
        qty=0.01,
        pnl_gross_usd=5.0,
        fees_paid_usd=0.34,
        funding_paid_usd=0.05,
        pnl_net_usd=4.61,
        context={
            "phase_K3": 1,
            "days_since_halving": 250,
            "day_of_week": 2,
            "hour_of_day": 14,
            "btc_dominance": 0.52,
        },
        pre_trade={
            "atr_at_entry": 800.0,
            "rv_predicted_har": 0.025,
            "regime_har": "normal",
            "vpin_at_entry": 0.42,
            "composite_signal": 0.31,
            "cloud_breakout_size_atr_units": 1.4,
            "volume_relative_30d": 1.15,
            "funding_rate_at_entry_bps": 1.2,
        },
        execution={
            "tf_signal_origin": "1h",
            "n_tf_confirming": 2,
            "slippage_bps": 1.5,
        },
        exit_info={"reason": "TP"},
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_meta_label — validations
# ---------------------------------------------------------------------------


def test_meta_label_all_required_fields_present():
    """Schema complet → payload valide avec toutes les clés top-level."""
    payload = build_meta_label(**_valid_kwargs())
    required = {
        "trade_id", "timestamp_open", "timestamp_close", "symbol", "side",
        "entry_price", "exit_price", "qty",
        "pnl_gross_usd", "fees_paid_usd", "funding_paid_usd", "pnl_net_usd",
        "context", "pre_trade", "execution", "exit",
    }
    assert required.issubset(set(payload.keys()))
    assert payload["side"] == "LONG"


def test_meta_label_rejects_invalid_side():
    with pytest.raises(ValueError, match="side"):
        build_meta_label(**_valid_kwargs(side="invalid_side"))


def test_meta_label_rejects_invalid_exit_reason():
    with pytest.raises(ValueError, match="exit reason"):
        build_meta_label(**_valid_kwargs(exit_info={"reason": "FOO_BAR"}))


def test_meta_label_rejects_missing_context_key():
    bad_ctx = {"phase_K3": 1, "day_of_week": 2}  # manque days_since_halving etc
    with pytest.raises(ValueError, match="context"):
        build_meta_label(**_valid_kwargs(context=bad_ctx))


def test_meta_label_rejects_missing_pre_trade_key():
    bad_pre = {"atr_at_entry": 800.0}
    with pytest.raises(ValueError, match="pre_trade"):
        build_meta_label(**_valid_kwargs(pre_trade=bad_pre))


def test_meta_label_handles_missing_optional_features():
    """Les valeurs None pour features optionnelles sont autorisées."""
    pre_trade = {
        "atr_at_entry": 800.0,
        "rv_predicted_har": None,    # HAR-RV pas dispo
        "regime_har": None,
        "vpin_at_entry": None,       # VPIN pas activé
        "composite_signal": 0.0,
        "cloud_breakout_size_atr_units": 1.0,
        "volume_relative_30d": 1.0,
        "funding_rate_at_entry_bps": None,
    }
    payload = build_meta_label(**_valid_kwargs(pre_trade=pre_trade))
    assert payload["pre_trade"]["vpin_at_entry"] is None


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_meta_label_json_serializable():
    payload = build_meta_label(**_valid_kwargs())
    s = json.dumps(payload, sort_keys=True)
    parsed = json.loads(s)
    assert parsed["trade_id"] == "trade_001"
    assert parsed["context"]["phase_K3"] == 1


# ---------------------------------------------------------------------------
# Hash chain (intégration avec AuditLog)
# ---------------------------------------------------------------------------


def test_meta_label_hash_chain_links_to_previous(tmp_path):
    path = tmp_path / "trades_meta.jsonl"
    logger = MetaLabelLogger(path=path)

    # Premier label → prev_hash = GENESIS
    e1 = logger.write_meta_label(**_valid_kwargs(trade_id="t1"))
    assert e1["seq"] == 0
    assert e1["prev_hash"] == "0" * 64

    # Deuxième → prev_hash = hash de e1
    e2 = logger.write_meta_label(**_valid_kwargs(trade_id="t2", entry_price=43000.0))
    assert e2["seq"] == 1
    assert e2["prev_hash"] == e1["hash"]

    # Verify chain : OK
    status = logger.verify()
    assert status.valid is True
    assert status.n_entries == 2


def test_meta_label_hash_chain_detects_tampering(tmp_path):
    """Si quelqu'un modifie le payload après écriture, verify_chain() détecte."""
    path = tmp_path / "trades_meta.jsonl"
    logger = MetaLabelLogger(path=path)
    logger.write_meta_label(**_valid_kwargs(trade_id="t1"))
    logger.write_meta_label(**_valid_kwargs(trade_id="t2"))

    lines = path.read_text().strip().split("\n")
    e = json.loads(lines[0])
    e["payload"]["pnl_net_usd"] = 999999.0  # tampered
    lines[0] = json.dumps(e)
    path.write_text("\n".join(lines) + "\n")

    status = logger.verify()
    assert status.valid is False
    assert "hash mismatch" in (status.error or "")


# ---------------------------------------------------------------------------
# Anti-leak PII / API keys
# ---------------------------------------------------------------------------


def test_meta_label_no_pii_no_api_keys():
    """Reject si une clé contient 'api_key', 'secret', 'token', etc."""
    bad_exec = {
        "tf_signal_origin": "1h",
        "n_tf_confirming": 2,
        "slippage_bps": 1.5,
        "api_key": "sk_test_xyz",  # leak
    }
    with pytest.raises(ValueError, match="PII|leak|forbidden"):
        build_meta_label(**_valid_kwargs(execution=bad_exec))


def test_meta_label_no_secret_substring_in_value():
    """Reject si une valeur ressemble à un secret 'token=abc' ou 'password: x'."""
    bad_pre = dict(_valid_kwargs()["pre_trade"])
    bad_pre["regime_har"] = "authorization: Bearer eyJhbGc..."
    with pytest.raises(ValueError, match="PII|leak|forbidden"):
        build_meta_label(**_valid_kwargs(pre_trade=bad_pre))


def test_meta_label_allows_innocent_strings():
    """Le filtre PII ne doit pas bloquer des chaînes innocentes (regime label)."""
    payload = build_meta_label(**_valid_kwargs())
    assert payload["pre_trade"]["regime_har"] == "normal"


# ---------------------------------------------------------------------------
# Branchement paper_trader.log_close
# ---------------------------------------------------------------------------


def test_paper_trader_writes_meta_label_on_close(tmp_path):
    """log_close avec meta_logger + meta_context → entry append à trades_meta.jsonl."""
    from bot.paper_trader import PaperTrader

    csv_path = tmp_path / "paper.csv"
    meta_path = tmp_path / "trades_meta.jsonl"
    meta_logger = MetaLabelLogger(path=meta_path)

    trader = PaperTrader(log_path=csv_path, meta_logger=meta_logger)

    kw = _valid_kwargs()
    meta_ctx = {
        "trade_id": kw["trade_id"],
        "timestamp_open": kw["timestamp_open"],
        "timestamp_close": kw["timestamp_close"],
        "symbol": kw["symbol"],
        "context": kw["context"],
        "pre_trade": kw["pre_trade"],
        "execution": kw["execution"],
        "exit": kw["exit_info"],
    }

    trader.log_close(
        action="close_long",
        qty=0.01,
        exit_price=42500.0,
        entry_price=42000.0,
        live_order_id="live_42",
        signal_id="trade_001",
        held_seconds=3600.0,
        meta_context=meta_ctx,
    )

    entries = list(meta_logger.iter_entries())
    assert len(entries) == 1
    assert entries[0]["payload"]["trade_id"] == "trade_001"
    assert entries[0]["payload"]["side"] == "LONG"
    status = meta_logger.verify()
    assert status.valid is True


def test_paper_trader_no_meta_logger_does_not_break(tmp_path):
    """Si meta_logger=None (default), log_close fonctionne normalement (back-compat)."""
    from bot.paper_trader import PaperTrader

    trader = PaperTrader(log_path=tmp_path / "paper.csv")
    trader.log_close(
        action="close_long",
        qty=0.01,
        exit_price=42500.0,
        entry_price=42000.0,
        live_order_id="x",
        signal_id="t1",
    )
    # Pas d'erreur, le CSV doit être écrit
    assert (tmp_path / "paper.csv").exists()
