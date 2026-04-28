"""P7-bis (T2) — Tests bucket_size_init compute V/N=50."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.bucket_size_init import (
    _FALLBACK_BUCKET_SIZE_USDT,
    compute_bucket_sizes,
    fetch_quote_volume_daily,
)


def test_bucket_size_auto_compute_uses_v50_rule_from_klines():
    """Klines 7j retournent volumes connus → mean/50 retourné."""
    # Mock 7 jours de klines, quote_volume = 5_000_000_000 chacun
    fake_klines = [
        [0, "0", "0", "0", "0", "0", 0, "5000000000.0", 0, "0", "0", "0"]
        for _ in range(7)
    ]
    with patch("services.bucket_size_init.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = fake_klines
        mock_req.get.return_value = resp
        result = compute_bucket_sizes(symbols=["BTCUSDT"], n_buckets=50)
    # mean = 5e9, divisé par 50 = 1e8
    assert result["BTCUSDT"] == pytest.approx(100_000_000.0, rel=1e-9)


def test_bucket_size_falls_back_on_api_fail():
    """HTTP 500 ou exception → fallback dict utilisé."""
    with patch("services.bucket_size_init.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 500
        mock_req.get.return_value = resp
        result = compute_bucket_sizes(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert result["BTCUSDT"] == _FALLBACK_BUCKET_SIZE_USDT["BTCUSDT"]
    assert result["ETHUSDT"] == _FALLBACK_BUCKET_SIZE_USDT["ETHUSDT"]
    assert result["SOLUSDT"] == _FALLBACK_BUCKET_SIZE_USDT["SOLUSDT"]


def test_bucket_size_falls_back_on_request_exception():
    """requests.get raise → fallback."""
    with patch("services.bucket_size_init.requests") as mock_req:
        mock_req.get.side_effect = Exception("network down")
        result = compute_bucket_sizes(symbols=["BTCUSDT"])
    assert result["BTCUSDT"] == _FALLBACK_BUCKET_SIZE_USDT["BTCUSDT"]


def test_bucket_size_handles_slash_symbol():
    """ETH/USDT normalisé en ETHUSDT."""
    fake_klines = [
        [0, "0", "0", "0", "0", "0", 0, "1000.0", 0, "0", "0", "0"]
        for _ in range(7)
    ]
    with patch("services.bucket_size_init.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = fake_klines
        mock_req.get.return_value = resp
        result = compute_bucket_sizes(symbols=["ETH/USDT"])
    assert "ETHUSDT" in result
    assert "ETH/USDT" not in result


def test_bucket_size_custom_n_buckets():
    """n_buckets=100 → bucket_size = mean/100."""
    fake_klines = [
        [0, "0", "0", "0", "0", "0", 0, "10000.0", 0, "0", "0", "0"]
        for _ in range(7)
    ]
    with patch("services.bucket_size_init.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = fake_klines
        mock_req.get.return_value = resp
        result = compute_bucket_sizes(symbols=["BTCUSDT"], n_buckets=100)
    assert result["BTCUSDT"] == pytest.approx(100.0, rel=1e-9)


def test_fetch_quote_volume_daily_extracts_index_7():
    """Verify on extrait bien l'index 7 (quote_volume) du kline."""
    fake_klines = [
        [0, "0", "0", "0", "0", "0", 0, "1234567.89", 0, "0", "0", "0"],
        [0, "0", "0", "0", "0", "0", 0, "9876543.21", 0, "0", "0", "0"],
    ]
    with patch("services.bucket_size_init.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = fake_klines
        mock_req.get.return_value = resp
        vols = fetch_quote_volume_daily("BTCUSDT", days=2)
    assert vols == [1234567.89, 9876543.21]
