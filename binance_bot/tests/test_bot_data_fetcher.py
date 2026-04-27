"""Tests pour DataFetcher — focus BUG-004 (filtre bougie non-clôturée)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from services.data_fetcher import DataFetcher


def _df_with_candles(timestamps_iso: list) -> pd.DataFrame:
    return pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps_iso, utc=True),
        'open': [50000.0] * len(timestamps_iso),
        'high': [50100.0] * len(timestamps_iso),
        'low': [49900.0] * len(timestamps_iso),
        'close': [50050.0] * len(timestamps_iso),
        'volume': [10.0] * len(timestamps_iso),
    })


# ============================================================
# BUG-004 — closed-bar filter
# ============================================================

def test_drops_unclosed_last_candle_2h():
    """Bougie ouverte à T n'est clôturée qu'à T+2h."""
    df = _df_with_candles([
        '2026-04-26T12:00:00Z',
        '2026-04-26T14:00:00Z',
        '2026-04-26T16:00:00Z',  # ouverte à 16h, clôturera à 18h
    ])
    # Maintenant on est à 17h → la bougie de 16h n'est pas clôturée
    now = pd.Timestamp('2026-04-26T17:00:00Z')

    out = DataFetcher._filter_unclosed_last_candle(df, '2h', now_utc=now)

    assert len(out) == 2
    assert out.iloc[-1]['timestamp'] == pd.Timestamp('2026-04-26T14:00:00Z')


def test_keeps_last_candle_when_just_closed():
    """Si la bougie vient de clôturer pile, on la garde."""
    df = _df_with_candles([
        '2026-04-26T12:00:00Z',
        '2026-04-26T14:00:00Z',
        '2026-04-26T16:00:00Z',
    ])
    # Maintenant on est à 18h pile → la bougie 16h est clôturée à 18h
    now = pd.Timestamp('2026-04-26T18:00:00Z')

    out = DataFetcher._filter_unclosed_last_candle(df, '2h', now_utc=now)

    assert len(out) == 3


def test_keeps_last_candle_well_after_close():
    """Bougie clôturée depuis longtemps → on garde."""
    df = _df_with_candles([
        '2026-04-26T12:00:00Z',
        '2026-04-26T14:00:00Z',
    ])
    now = pd.Timestamp('2026-04-26T20:00:00Z')

    out = DataFetcher._filter_unclosed_last_candle(df, '2h', now_utc=now)

    assert len(out) == 2


def test_works_with_1h_timeframe():
    df = _df_with_candles([
        '2026-04-26T15:00:00Z',
        '2026-04-26T16:00:00Z',  # ouverte 16h, clôt 17h
    ])
    now = pd.Timestamp('2026-04-26T16:30:00Z')

    out = DataFetcher._filter_unclosed_last_candle(df, '1h', now_utc=now)

    assert len(out) == 1


def test_works_with_1d_timeframe():
    df = _df_with_candles([
        '2026-04-25T00:00:00Z',
        '2026-04-26T00:00:00Z',  # ouverte 26/04, clôt 27/04
    ])
    now = pd.Timestamp('2026-04-26T15:00:00Z')

    out = DataFetcher._filter_unclosed_last_candle(df, '1d', now_utc=now)

    assert len(out) == 1
    assert out.iloc[-1]['timestamp'] == pd.Timestamp('2026-04-25T00:00:00Z')


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    out = DataFetcher._filter_unclosed_last_candle(df, '2h')
    assert out.empty


def test_uses_real_now_when_not_provided():
    """Smoke test: l'appel sans now_utc utilise pd.Timestamp.now(tz='UTC')."""
    # Bougie ouverte loin dans le passé → toujours considérée clôturée
    df = _df_with_candles(['2020-01-01T00:00:00Z'])
    out = DataFetcher._filter_unclosed_last_candle(df, '2h')
    assert len(out) == 1
