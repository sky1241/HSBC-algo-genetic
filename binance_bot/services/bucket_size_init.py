"""P7-bis (T2) — Auto-compute bucket_size_v par symbol selon Easley 2012.

Formule canonique : bucket_size_v = mean(quote_volume_daily_7d) / N=50.
Source de vérité : Binance Futures klines REST API.

Au premier lancement du daemon vpin_live_runner (ou si yaml clé absente
ou plus vieille que 7 jours), on appelle compute_bucket_sizes() pour
fetch les volumes 7 derniers jours et calculer V/N par symbole.

Fallback si fetch fail (ex: rate limit, timeout) : valeurs hardcodées
ordres de grandeur Binance perp daily fin 2025 :
    BTCUSDT : 40_000_000 USDT  (~$2B daily / 50)
    ETHUSDT : 16_000_000 USDT  (~$800M daily / 50)
    SOLUSDT :  4_000_000 USDT  (~$200M daily / 50)

Ces valeurs sont conservatrices ; à recalibrer post-soak avec data
réelle observée sur 30j.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_DEFAULT_LOOKBACK_DAYS = 7
_DEFAULT_N_BUCKETS = 50
_FALLBACK_BUCKET_SIZE_USDT = {
    "BTCUSDT": 40_000_000.0,
    "ETHUSDT": 16_000_000.0,
    "SOLUSDT":  4_000_000.0,
}


def fetch_quote_volume_daily(symbol: str, days: int = _DEFAULT_LOOKBACK_DAYS) -> Optional[list[float]]:
    """Fetch klines daily Binance Futures, retourne quote_volume[].

    Returns:
        Liste de quote_volume USDT pour les `days` derniers jours, ou
        None si fetch échoue.
    """
    if requests is None:
        return None
    try:
        r = requests.get(
            _KLINES_URL,
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(f"klines {symbol}: HTTP {r.status_code}")
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        # Format Binance kline:
        # [open_time, open, high, low, close, volume, close_time,
        #  quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, _]
        # Index 7 = quote_volume (USDT pour les pairs USDT-margined)
        return [float(k[7]) for k in data]
    except Exception as e:
        logger.warning(f"klines {symbol} failed: {e}")
        return None


def compute_bucket_sizes(
    symbols: list[str] | tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    n_buckets: int = _DEFAULT_N_BUCKETS,
    fallback: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Compute bucket_size_v par symbol via klines daily 7j.

    Pour chaque symbol :
      - Fetch quote_volume_daily 7d
      - bucket_size = mean(quote_volume_7d) / n_buckets  (Easley 2012 V/N)
      - Si fetch fail → fallback dict

    Returns:
        {symbol: bucket_size_v} en USDT.
    """
    if fallback is None:
        fallback = _FALLBACK_BUCKET_SIZE_USDT
    n = max(1, int(n_buckets))
    out: dict[str, float] = {}
    for sym in symbols:
        sym_clean = str(sym).replace("/", "").upper()
        vols = fetch_quote_volume_daily(sym_clean, days=_DEFAULT_LOOKBACK_DAYS)
        if vols is None or len(vols) == 0:
            fb = float(fallback.get(sym_clean, 1_000_000.0))
            logger.warning(f"{sym_clean}: fallback bucket_size={fb}")
            out[sym_clean] = fb
            continue
        avg = sum(vols) / len(vols)
        out[sym_clean] = float(avg) / n
    return out


__all__ = [
    "fetch_quote_volume_daily",
    "compute_bucket_sizes",
]
