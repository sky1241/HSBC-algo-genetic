"""P-MTF-2 — Tests du filtre directionnel H2 Ichimoku."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.trend_filter_h2 import classify_h2_trend, is_h2_trend_stale


def _make_df(close: float, cloud_top: float, cloud_bottom: float) -> pd.DataFrame:
    """Construit un DF minimal avec une dernière ligne aux valeurs voulues."""
    return pd.DataFrame(
        {
            "close": [50_000.0, 50_500.0, close],
            "cloud_top": [49_000.0, 49_500.0, cloud_top],
            "cloud_bottom": [48_000.0, 48_500.0, cloud_bottom],
        }
    )


# ---------------------------------------------------------------------------
# classify_h2_trend
# ---------------------------------------------------------------------------


def test_close_above_cloud_returns_long():
    df = _make_df(close=60_000.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "long"
    assert r["reason"] == "above_cloud"
    assert r["last_close"] == 60_000.0


def test_close_below_cloud_returns_short():
    df = _make_df(close=40_000.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "short"
    assert r["reason"] == "below_cloud"


def test_close_inside_cloud_returns_flat():
    df = _make_df(close=52_500.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "flat"
    assert r["reason"] == "inside_cloud"


def test_close_at_cloud_top_strict_returns_flat():
    """close == cloud_top exactement → flat (close doit être strictement >)."""
    df = _make_df(close=55_000.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "flat"


def test_close_just_above_cloud_with_margin_returns_flat():
    """Avec margin 1%, close à 1.005×cloud_top → flat (pas assez au-dessus)."""
    df = _make_df(close=55_550.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df, margin_pct=0.01)  # 1% requis
    assert r["direction"] == "flat"  # +1.0% < +1.0% requis (strict)


def test_close_well_above_cloud_with_margin_returns_long():
    """Avec margin 1%, close à 1.02×cloud_top → long (assez au-dessus)."""
    df = _make_df(close=56_100.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df, margin_pct=0.01)
    assert r["direction"] == "long"


def test_empty_df_returns_flat_safe():
    r = classify_h2_trend(pd.DataFrame())
    assert r["direction"] == "flat"
    assert r["reason"] == "no_data"
    assert r["last_close"] is None


def test_none_df_returns_flat_safe():
    r = classify_h2_trend(None)
    assert r["direction"] == "flat"
    assert r["reason"] == "no_data"


def test_missing_columns_returns_flat_safe():
    df = pd.DataFrame({"close": [50_000.0], "wrong_col": [49_000.0]})
    r = classify_h2_trend(df)
    assert r["direction"] == "flat"
    assert r["reason"] == "missing_columns"


def test_nan_close_returns_flat_safe():
    df = _make_df(close=float("nan"), cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "flat"
    assert r["reason"] == "nan_values"


def test_nan_cloud_returns_flat_safe():
    """Au démarrage Ichimoku, cloud_top/bottom peuvent être NaN (shift)."""
    df = _make_df(close=60_000.0, cloud_top=float("nan"), cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert r["direction"] == "flat"
    assert r["reason"] == "nan_values"


def test_distance_pct_computed_correctly():
    df = _make_df(close=60_000.0, cloud_top=55_000.0, cloud_bottom=45_000.0)
    r = classify_h2_trend(df)
    # cloud_mid = 50000, distance = (60000 - 50000) / 50000 = 0.20
    assert r["distance_pct"] == pytest.approx(0.20)


def test_computed_at_iso_present_and_valid():
    df = _make_df(close=60_000.0, cloud_top=55_000.0, cloud_bottom=50_000.0)
    r = classify_h2_trend(df)
    assert "computed_at_iso" in r
    # Doit être parsable
    parsed = datetime.fromisoformat(r["computed_at_iso"])
    assert parsed.tzinfo is not None  # UTC timezone-aware


# ---------------------------------------------------------------------------
# is_h2_trend_stale
# ---------------------------------------------------------------------------


def test_is_h2_trend_stale_none_returns_true():
    assert is_h2_trend_stale(None) is True


def test_is_h2_trend_stale_empty_dict_returns_true():
    assert is_h2_trend_stale({}) is True


def test_is_h2_trend_stale_missing_iso_returns_true():
    assert is_h2_trend_stale({"direction": "long"}) is True


def test_is_h2_trend_stale_invalid_iso_returns_true():
    assert is_h2_trend_stale({"computed_at_iso": "not-a-date"}) is True


def test_is_h2_trend_fresh_returns_false():
    now = datetime.now(timezone.utc)
    h2 = {"computed_at_iso": (now - timedelta(minutes=30)).isoformat()}
    assert is_h2_trend_stale(h2, max_age_hours=3.0, now=now) is False


def test_is_h2_trend_stale_old_returns_true():
    now = datetime.now(timezone.utc)
    h2 = {"computed_at_iso": (now - timedelta(hours=5)).isoformat()}
    assert is_h2_trend_stale(h2, max_age_hours=3.0, now=now) is True


def test_is_h2_trend_stale_at_threshold_returns_false():
    """Pile au seuil = pas stale (strict >)."""
    now = datetime.now(timezone.utc)
    h2 = {"computed_at_iso": (now - timedelta(hours=3)).isoformat()}
    assert is_h2_trend_stale(h2, max_age_hours=3.0, now=now) is False


def test_is_h2_trend_stale_naive_iso_assumed_utc():
    """Si l'ISO est naïf (sans tz), on suppose UTC."""
    now = datetime.now(timezone.utc)
    naive_iso = (now - timedelta(minutes=30)).replace(tzinfo=None).isoformat()
    h2 = {"computed_at_iso": naive_iso}
    assert is_h2_trend_stale(h2, max_age_hours=3.0, now=now) is False
