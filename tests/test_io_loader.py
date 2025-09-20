import pandas as pd

from src import io_loader


def _make_prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="2h")
    data = {
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100, 101, 102, 103, 104],
        "volume": [10, 11, 12, 13, 14],
    }
    df = pd.DataFrame(data, index=index)
    df.index.name = "timestamp"
    return df


def test_load_funding_and_align(tmp_path):
    prices = _make_prices()
    price_path = tmp_path / "prices.csv"
    prices.reset_index().to_csv(price_path, index=False)
    loaded_prices = io_loader.load_ohlcv_csv(price_path)

    funding_index = pd.date_range("2024-01-01", periods=2, freq="8h")
    funding_df = pd.DataFrame(
        {
            "timestamp": funding_index,
            "funding_rate": [0.0008, 0.0004],
        }
    )
    funding_path = tmp_path / "funding.csv"
    funding_df.to_csv(funding_path, index=False)

    funding_series = io_loader.load_funding_csv(funding_path)
    aligned = io_loader.align_funding_to_ohlcv(loaded_prices, funding_series)

    expected = pd.Series(
        [0.0002, 0.0002, 0.0002, 0.0002, 0.0001],
        index=loaded_prices.index,
        name="funding_rate",
    )
    pd.testing.assert_series_equal(aligned, expected)
