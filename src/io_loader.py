"""Utility helpers for loading and validating OHLCV datasets.

The functions in this module are intentionally lightweight so they can be
reused both by scripts (``run_oos.py``) and unit tests. They normalise the
expected columns, ensure a :class:`~pandas.DatetimeIndex` and provide small
helpers such as frequency inference.  Funding series are also handled so the
pipeline can include taker fees and funding costs in the equity curve as
required by the *Pipeline PhasenAware Ichimoku* prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _normalise_columns(columns: Iterable[str]) -> list[str]:
    """Return a list of lower-case column names stripped from surrounding
    whitespace.
    """

    return [c.strip().lower() for c in columns]


def _detect_timestamp_column(df: pd.DataFrame) -> str:
    """Return the column containing timestamp information.

    The loader accepts common aliases such as ``timestamp``, ``date`` or ``time``.
    A :class:`ValueError` is raised when none can be found.
    """

    lowered = {c.lower(): c for c in df.columns}
    for candidate in ("timestamp", "date", "time", "datetime"):
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Impossible de détecter la colonne temporelle (timestamp/date/time)."
    )


def load_ohlcv_csv(path: str | Path, tz: str | None = "UTC") -> pd.DataFrame:
    """Load an OHLCV CSV file and return a validated DataFrame.

    Parameters
    ----------
    path:
        Location of the CSV file.  The file must contain at least the columns
        defined in :data:`REQUIRED_COLUMNS` (case insensitive).
    tz:
        Optional timezone to localise the resulting index.
    """

    csv_path = Path(path)
    if not csv_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Le fichier CSV est vide")

    timestamp_col = _detect_timestamp_column(df)
    df.columns = _normalise_columns(df.columns)
    timestamp_series = pd.to_datetime(df[timestamp_col.lower()], utc=True)
    if tz:
        timestamp_series = timestamp_series.dt.tz_convert(tz)
    df = df.drop(columns=[timestamp_col.lower()])

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Colonne requise manquante dans le CSV: '{column}'"
            )

    df = df[list(REQUIRED_COLUMNS)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.DatetimeIndex(timestamp_series, name="timestamp")
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def load_funding_csv(
    path: str | Path,
    *,
    value_column: str | None = None,
    tz: str | None = "UTC",
) -> pd.Series:
    """Load a funding CSV file and return a normalised series.

    Parameters
    ----------
    path:
        Location of the CSV file containing funding rates (8h sampling by
        default on crypto perpetuals).
    value_column:
        Optional column holding the funding rate. When omitted the loader tries
        common aliases such as ``funding_rate`` or ``funding``.
    tz:
        Optional timezone to localise the resulting index.
    """

    csv_path = Path(path)
    if not csv_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Le fichier CSV de funding est vide")

    timestamp_col = _detect_timestamp_column(df)
    df.columns = _normalise_columns(df.columns)
    timestamp_series = pd.to_datetime(df[timestamp_col.lower()], utc=True)
    if tz:
        timestamp_series = timestamp_series.dt.tz_convert(tz)

    candidates = [value_column] if value_column else []
    candidates.extend(["funding_rate", "funding", "rate"])
    column_name: str | None = None
    for candidate in candidates:
        if candidate and candidate.lower() in df.columns:
            column_name = candidate.lower()
            break
    if column_name is None:
        raise ValueError("Impossible de détecter la colonne de funding")

    series = pd.to_numeric(df[column_name], errors="coerce")
    series.index = pd.DatetimeIndex(timestamp_series, name="timestamp")
    series = series.sort_index()
    series = series.dropna()
    return series.rename("funding_rate")


def ensure_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a dataframe shape and columns.

    The function simply checks for the mandatory OHLCV columns and that the
    index is a :class:`~pandas.DatetimeIndex`.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être un DatetimeIndex")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes: {missing}")
    return df


def _infer_hours_from_index(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 0.0
    return float(np.median(deltas) / 3600.0)


def align_funding_to_ohlcv(
    prices: pd.DataFrame,
    funding: pd.Series,
    target_freq: str | None = None,
) -> pd.Series:
    """Align an 8h funding series to the OHLCV sampling frequency.

    The funding rates are distributed *pro rata* over the shorter price bars so
    that, for example, a 0.01 funding paid every eight hours becomes 0.0025 per
    two-hour bar.  Missing values are forward-filled while leading gaps are
    filled with zeros to avoid introducing lookahead bias.
    """

    ensure_ohlcv_dataframe(prices)
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise TypeError("L'index de la série de funding doit être temporel")
    funding = funding.sort_index()

    price_index = prices.index
    if target_freq:
        price_hours = float(pd.to_timedelta(target_freq).total_seconds() / 3600.0)
    else:
        price_hours = _infer_hours_from_index(price_index)
    funding_hours = _infer_hours_from_index(funding.index)
    if funding_hours <= 0.0:
        raise ValueError("Impossible d'inférer la fréquence du funding")
    if price_hours <= 0.0:
        raise ValueError("Impossible d'inférer la fréquence des prix")

    scale = price_hours / funding_hours
    aligned = funding.reindex(price_index, method="ffill")
    aligned = aligned.fillna(0.0)
    return aligned.astype(float) * float(scale)


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the number of periods per year from a datetime index.

    The heuristic takes the median spacing between consecutive timestamps and
    extrapolates it to a 365-day year.  Falling back to 252 when the index is
    too short or irregular.
    """

    if len(index) < 2:
        return 252
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    median_delta = float(np.median(deltas)) if len(deltas) else 0.0
    if median_delta <= 0.0:
        return 252
    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    periods = int(round(seconds_per_year / median_delta))
    return max(periods, 1)


def restrict_years(df: pd.DataFrame, start_year: int | None, end_year: int | None) -> pd.DataFrame:
    """Return a slice of ``df`` constrained to the inclusive year range."""

    ensure_ohlcv_dataframe(df)
    mask = pd.Series(True, index=df.index)
    if start_year is not None:
        mask &= df.index.year >= int(start_year)
    if end_year is not None:
        mask &= df.index.year <= int(end_year)
    return df.loc[mask]


__all__ = [
    "REQUIRED_COLUMNS",
    "load_ohlcv_csv",
    "load_funding_csv",
    "ensure_ohlcv_dataframe",
    "align_funding_to_ohlcv",
    "infer_periods_per_year",
    "restrict_years",
]
