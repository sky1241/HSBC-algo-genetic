import pandas as pd
import pytest

from src.io_loader import align_funding_to_ohlcv


def make_prices(periods: int = 8) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="2h", tz="UTC")
    data = {
        "open": pd.Series(range(periods), dtype=float).values,
        "high": pd.Series(range(periods), dtype=float).values,
        "low": pd.Series(range(periods), dtype=float).values,
        "close": pd.Series(range(periods), dtype=float).values,
        "volume": pd.Series([1.0] * periods, dtype=float).values,
    }
    return pd.DataFrame(data, index=index)


def make_funding(timestamps: list[str], values: list[float]) -> pd.DataFrame:
    index = pd.to_datetime(timestamps, utc=True)
    return pd.DataFrame({"funding": values}, index=index)


def test_align_funding_even_distribution():
    prices = make_prices()
    funding = make_funding([
        "2024-01-01 08:00",
        "2024-01-01 16:00",
    ], [0.08, 0.12])

    aligned = align_funding_to_ohlcv(prices, funding)
    funding_series = aligned["funding"]

    assert funding_series.index.equals(prices.index)
    assert not funding_series.isna().any()
    assert funding_series.iloc[:4].tolist() == pytest.approx([0.02] * 4)
    assert funding_series.iloc[4:].tolist() == pytest.approx([0.03] * 4)
    assert funding_series.sum() == pytest.approx(funding["funding"].sum())


def test_align_funding_missing_window_raises():
    prices = make_prices()
    funding = make_funding(["2024-01-01 08:00"], [0.08])

    with pytest.raises(ValueError, match="funding"):
        align_funding_to_ohlcv(prices, funding)
