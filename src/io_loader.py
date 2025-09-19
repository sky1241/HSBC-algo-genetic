"""Utilities to load cached market data for backtests.

This module centralises the I/O logic that used to live inside the
monolithic pipeline script.  The helper functions read OHLCV and funding
CSV files, enforce a strict coverage policy (no missing timestamps,
duplicates or timezone drift) and expose the minimal structure required
by the backtest: aligned close prices and pro‑rated funding rates.

The implementation favours explicit validation so that corrupted caches
are rejected early, which is critical for reproducible research.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "load_ohlcv",
    "load_funding",
    "align_funding_to_ohlcv",
]


_TIMESTAMP_CANDIDATES: Final[tuple[str, ...]] = (
    "timestamp",
    "time",
    "date",
    "datetime",
    "open_time",
    "close_time",
    "ts",
)


def _normalise_paths(csv_path: str | Path | Iterable[str | Path]) -> list[Path]:
    """Return a list of concrete paths from a single path or an iterable."""

    if isinstance(csv_path, (str, Path)):
        paths = [Path(csv_path)]
    else:
        paths = [Path(p) for p in csv_path]
    if not paths:
        raise ValueError("aucun fichier CSV fourni")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"fichiers introuvables: {', '.join(missing)}")
    return paths


def _parse_timestamp_column(df: pd.DataFrame, source: Path) -> pd.Series:
    """Return a timezone-aware timestamp series from a raw dataframe."""

    df_columns = {str(c).strip().lower(): c for c in df.columns}
    ts_col = next((df_columns[c] for c in _TIMESTAMP_CANDIDATES if c in df_columns), None)
    if ts_col is None:
        raise ValueError(f"colonne timestamp manquante dans {source}")

    ts = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts):
        ts = pd.to_numeric(ts, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"timestamps invalides dans {source}")
        multiplier = 1000 if ts.gt(10**11).any() else 1
        ts = pd.to_datetime(ts / multiplier, unit="s", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"timestamps non parsables dans {source}")
    return ts


def _ensure_no_conflicting_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Raise if duplicated timestamps carry diverging OHLCV values."""

    duplicated = df.index.duplicated(keep=False)
    if not duplicated.any():
        return df

    conflicts = []
    for ts, group in df[duplicated].groupby(level=0):
        values = group.to_numpy(dtype=float, copy=False)
        baseline = values[0]
        if not np.allclose(values, baseline, rtol=1e-6, atol=1e-8, equal_nan=True):
            conflicts.append(ts)
    if conflicts:
        raise ValueError(
            f"données incompatibles pour {label}: doublons divergents sur {len(conflicts)} timestamps"
        )
    return df[~df.index.duplicated(keep="last")]


def _check_regular_index(index: pd.DatetimeIndex, freq: str, label: str) -> None:
    """Ensure the index covers every expected timestamp."""

    if index.tz is None:
        raise ValueError(f"index sans timezone pour {label}")
    expected = pd.date_range(index[0], index[-1], freq=freq.lower(), tz=index.tz)
    missing = expected.difference(index)
    if not missing.empty:
        raise ValueError(
            f"données incomplètes pour {label}: {len(missing)} timestamps manquants (freq {freq})"
        )


def load_ohlcv(csv_path: str | Path | Sequence[str | Path], tz: str | None = "UTC") -> pd.DataFrame:
    """Load and validate OHLCV candles.

    Parameters
    ----------
    csv_path:
        Single path or iterable of CSV files to merge chronologically.
    tz:
        Target timezone for the returned index. The CSVs are assumed to be
        expressed in UTC (as produced by ccxt caches).

    Returns
    -------
    DataFrame
        A dataframe indexed by tz-aware timestamps with the canonical OHLCV
        columns. Raises on missing timestamps, duplicates or NaN volumes.
    """

    paths = _normalise_paths(csv_path)
    frames: list[pd.DataFrame] = []
    col_aliases = {
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "close": ("close", "adj_close", "price"),
        "volume": ("volume", "vol", "qty", "quote_volume"),
    }

    for path in paths:
        raw = pd.read_csv(path)
        if raw.empty:
            raise ValueError(f"fichier vide: {path}")
        ts = _parse_timestamp_column(raw, path)
        raw = raw.copy()
        raw["timestamp"] = ts
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        rename_map: dict[str, str] = {}
        for target, aliases in col_aliases.items():
            for alias in aliases:
                if alias in raw.columns:
                    rename_map[alias] = target
                    break
        raw = raw.rename(columns=rename_map)
        missing = [col for col in col_aliases if col not in raw.columns]
        if missing:
            raise ValueError(f"colonnes OHLCV manquantes dans {path}: {', '.join(sorted(missing))}")

        ohlcv = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for col in ("open", "high", "low", "close", "volume"):
            ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
        if ohlcv[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise ValueError(f"valeurs OHLCV invalides dans {path}")
        ohlcv = ohlcv.set_index("timestamp").sort_index()
        if ohlcv.index.has_duplicates:
            raise ValueError(f"timestamps dupliqués dans {path}")
        frames.append(ohlcv)

    combined = pd.concat(frames)
    if combined.empty:
        raise ValueError("aucune donnée OHLCV chargée")
    combined = combined.sort_index()
    combined = _ensure_no_conflicting_duplicates(combined, "OHLCV")

    combined.index = combined.index.tz_localize("UTC") if combined.index.tz is None else combined.index.tz_convert("UTC")
    if tz:
        combined.index = combined.index.tz_convert(tz)

    freq = pd.infer_freq(combined.index)
    if freq is None:
        raise ValueError("fréquence des bougies indéterminée")
    try:
        freq_td = pd.to_timedelta(freq)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"fréquence non supportée: {freq}") from exc
    expected_td = pd.Timedelta(hours=2)
    if freq_td != expected_td:
        raise ValueError(f"fréquence inattendue pour OHLCV: {freq} (attendu: 2H)")

    _check_regular_index(combined.index, "2H", "OHLCV")

    if combined["volume"].isna().any():
        raise ValueError("volume manquant détecté dans les données OHLCV")

    return combined


def load_funding(csv_path: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Load futures funding rates sampled every 8 hours."""

    paths = _normalise_paths(csv_path)
    frames: list[pd.DataFrame] = []
    funding_aliases = ("funding", "funding_rate", "rate", "funding8h")

    for path in paths:
        raw = pd.read_csv(path)
        if raw.empty:
            raise ValueError(f"fichier vide: {path}")
        ts = _parse_timestamp_column(raw, path)
        raw = raw.copy()
        raw["timestamp"] = ts
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        col = next((c for c in funding_aliases if c in raw.columns), None)
        if col is None:
            raise ValueError(f"colonne funding absente dans {path}")
        series = pd.to_numeric(raw[col], errors="coerce")
        if series.isna().any():
            raise ValueError(f"valeurs de funding invalides dans {path}")
        df = pd.DataFrame({"funding": series.values}, index=ts)
        if df.index.has_duplicates:
            raise ValueError(f"timestamps dupliqués dans {path}")
        frames.append(df.sort_index())

    combined = pd.concat(frames)
    if combined.empty:
        raise ValueError("aucune donnée de funding chargée")
    combined = combined.sort_index()
    combined = _ensure_no_conflicting_duplicates(combined, "funding")
    combined.index = combined.index.tz_localize("UTC") if combined.index.tz is None else combined.index.tz_convert("UTC")

    freq = pd.infer_freq(combined.index)
    if freq is None:
        raise ValueError("fréquence des données de funding indéterminée")
    try:
        freq_td = pd.to_timedelta(freq)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"fréquence non supportée: {freq}") from exc
    expected_td = pd.Timedelta(hours=8)
    if freq_td != expected_td:
        raise ValueError(f"fréquence inattendue pour le funding: {freq} (attendu: 8H)")
    _check_regular_index(combined.index, "8H", "funding")

    return combined


def align_funding_to_ohlcv(
    df_prices: pd.DataFrame,
    df_funding: pd.DataFrame,
    freq: str = "2H",
) -> dict[str, pd.Series]:
    """Align OHLCV and funding data on a common timeframe.

    Funding rates are provided on an 8-hour grid. They are redistributed on
    the target frequency (2-hour candles by default) on a pro-rata basis so
    that the cumulated funding over an 8-hour window matches the original
    payment.
    """

    if df_prices.empty:
        raise ValueError("df_prices est vide")
    if df_funding.empty:
        raise ValueError("df_funding est vide")

    if not isinstance(df_prices.index, pd.DatetimeIndex) or not isinstance(df_funding.index, pd.DatetimeIndex):
        raise TypeError("les index doivent être de type DatetimeIndex")
    if df_prices.index.tz is None or df_funding.index.tz is None:
        raise ValueError("les index doivent être aware (avec timezone)")
    if df_prices.index.tz != df_funding.index.tz:
        raise ValueError("timezone incohérente entre prix et funding")

    freq = freq.lower()
    try:
        target_delta = pd.to_timedelta(freq)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"fréquence cible invalide: {freq}") from exc
    if target_delta <= pd.Timedelta(0):
        raise ValueError(f"fréquence cible invalide: {freq}")

    inferred_freq = pd.infer_freq(df_prices.index)
    if inferred_freq is None:
        raise ValueError("impossible d'inférer la fréquence des prix")
    if pd.to_timedelta(inferred_freq) != target_delta:
        raise ValueError(
            f"fréquence inattendue pour les prix: {inferred_freq} (attendu: {freq})"
        )
    _check_regular_index(df_prices.index, freq, "prix")

    base_interval = pd.Timedelta(hours=8)
    steps = base_interval / target_delta
    steps_int = int(round(float(steps)))
    if steps_int <= 0 or not np.isclose(steps, steps_int):
        raise ValueError("la fréquence cible doit diviser 8H exactement")

    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    for ts, value in df_funding["funding"].items():
        start = ts - base_interval + target_delta
        per_slice = float(value) / steps_int
        for i in range(steps_int):
            target_ts = start + i * target_delta
            timestamps.append(target_ts)
            values.append(per_slice)

    funding_series = pd.Series(values, index=pd.DatetimeIndex(timestamps, tz=df_funding.index.tz))
    funding_series = funding_series.groupby(level=0).sum().sort_index()

    # Ensure coverage before reindexing
    first_price, last_price = df_prices.index[0], df_prices.index[-1]
    if funding_series.index.min() > first_price or funding_series.index.max() < last_price:
        raise ValueError("le funding ne couvre pas entièrement la période des prix")

    funding_series = funding_series.reindex(df_prices.index)
    if funding_series.isna().any():
        raise ValueError("trous détectés après alignement du funding")

    close_series = df_prices["close"].astype(float).copy()
    close_series.name = "close"
    funding_series.name = "funding"
    return {"close": close_series, "funding": funding_series}

