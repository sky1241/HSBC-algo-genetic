from __future__ import annotations

import pandas as pd
import pandas.testing as tm

from src import stats_eval


def _build_returns(tz: str | None = None) -> pd.Series:
    index = pd.DatetimeIndex(
        [
            "2021-01-10",
            "2021-01-20",
            "2021-02-10",
            "2021-02-20",
        ],
        tz=tz,
    )
    return pd.Series([0.01, 0.02, 0.03, 0.04], index=index, dtype=float)


def _expected_monthly_returns() -> pd.Series:
    expected_index = pd.period_range("2021-01", "2021-02", freq="M")
    expected_values = [
        (1.01 * 1.02) - 1.0,
        (1.03 * 1.04) - 1.0,
    ]
    return pd.Series(expected_values, index=expected_index, dtype=float)


def test_compute_monthly_returns_with_naive_index() -> None:
    returns = _build_returns()
    monthly = stats_eval.compute_monthly_returns(returns)
    expected = _expected_monthly_returns()
    assert isinstance(monthly.index, pd.PeriodIndex)
    tm.assert_index_equal(monthly.index, expected.index)
    tm.assert_series_equal(monthly, expected, rtol=1e-9, atol=0.0)


def test_compute_monthly_returns_with_timezone_aware_index() -> None:
    returns = _build_returns("UTC")
    monthly = stats_eval.compute_monthly_returns(returns)
    expected = _expected_monthly_returns()
    assert isinstance(monthly.index, pd.PeriodIndex)
    tm.assert_index_equal(monthly.index, expected.index)
    tm.assert_series_equal(monthly, expected, rtol=1e-9, atol=0.0)
