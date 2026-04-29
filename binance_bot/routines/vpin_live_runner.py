#!/usr/bin/env python3
"""P7-bis — Daemon entrypoint VPIN live collector (systemd hsbc-vpin-live).

Lance VPINLiveCollectorWS pour BTCUSDT/ETHUSDT/SOLUSDT en parallèle.
2 WS threads par symbole (@trade + @bookTicker). Heartbeat log toutes
les 5min. SIGTERM stop drain proper.

Bucket sizes auto-calculées au premier lancement via bucket_size_init
(klines daily 7j / 50). Cachées dans bot_settings.yaml clé
`vpin_bucket_size_usdt` ; recalculées si absentes ou stale (>7j).

Lancé par hsbc-vpin-live.service (Type=simple, Restart=always).

Activation manuelle requise (PAS enable par défaut) :
    systemctl --user link binance_bot/systemd/hsbc-vpin-live.service
    systemctl --user enable --now hsbc-vpin-live
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from services.bucket_size_init import compute_bucket_sizes
from services.vpin_live_collector import VPINLiveCollectorWS


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
STORE = ROOT / "data" / "vpin_live.jsonl"
SETTINGS_PATH = ROOT / "configs" / "bot_settings.yaml"
HEARTBEAT_INTERVAL_SEC = 300  # log toutes les 5 min
BUCKET_SIZE_TTL_DAYS = 7

logger = logging.getLogger("vpin_live_runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning(f"settings load failed: {e}")
        return {}


def _save_settings(cfg: dict) -> None:
    """Surgical line-based update — préserve les comments du yaml.

    yaml.safe_dump strippait tous les comments (régression observée
    2026-04-29 lors de l'activation P7-bis). On édite seulement les 4
    lignes qu'on a réellement besoin de modifier après recompute des
    bucket sizes : 3× bucket_size + 1× last_compute_iso.
    """
    try:
        original = SETTINGS_PATH.read_text(encoding="utf-8").splitlines()
        sizes = cfg.get("vpin_bucket_size_usdt") or {}
        last_iso = cfg.get("vpin_bucket_size_last_compute_iso")

        out: list[str] = []
        in_bucket_block = False
        for line in original:
            stripped = line.strip()
            # Ouverture du bloc bucket_size
            if stripped.startswith("vpin_bucket_size_usdt:"):
                in_bucket_block = True
                out.append(line)
                continue
            # Mise à jour des sous-clés du bloc bucket_size
            if in_bucket_block:
                # Sortie du bloc = ligne sans indent ou clé top-level
                if line and not line.startswith(("  ", "\t")):
                    in_bucket_block = False
                else:
                    for sym in sizes:
                        if stripped.startswith(f"{sym}:"):
                            indent = line[: len(line) - len(line.lstrip())]
                            out.append(f"{indent}{sym}: {sizes[sym]}")
                            break
                    else:
                        out.append(line)
                    continue
            # Mise à jour last_compute_iso (clé top-level)
            if stripped.startswith("vpin_bucket_size_last_compute_iso:"):
                out.append(f"vpin_bucket_size_last_compute_iso: '{last_iso}'")
                continue
            out.append(line)

        SETTINGS_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning(f"settings save failed: {e}")


def _bucket_sizes_stale_or_missing(cfg: dict) -> bool:
    """True si la clé vpin_bucket_size_usdt est absente, vide, ou si le
    timestamp last_compute est plus vieux que TTL (7 jours)."""
    sizes = cfg.get("vpin_bucket_size_usdt") or {}
    if not isinstance(sizes, dict) or not sizes:
        return True
    # Toutes les valeurs doivent être numériques > 0
    for sym in SYMBOLS:
        v = sizes.get(sym)
        if not isinstance(v, (int, float)) or v <= 0:
            return True
    last_iso = cfg.get("vpin_bucket_size_last_compute_iso")
    if not last_iso:
        return True
    try:
        last_dt = datetime.fromisoformat(str(last_iso).replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
        if age_days > BUCKET_SIZE_TTL_DAYS:
            return True
    except Exception:
        return True
    return False


def _ensure_bucket_sizes(cfg: dict) -> dict[str, float]:
    """Retourne {symbol: bucket_size_v}. Recompute si stale, persist yaml."""
    if _bucket_sizes_stale_or_missing(cfg):
        logger.info("bucket sizes stale/missing → recompute via klines 7d")
        sizes = compute_bucket_sizes(symbols=list(SYMBOLS))
        cfg["vpin_bucket_size_usdt"] = sizes
        cfg["vpin_bucket_size_last_compute_iso"] = (
            datetime.now(timezone.utc).isoformat()
        )
        _save_settings(cfg)
        logger.info(f"bucket sizes computed: {sizes}")
        return sizes
    return {sym: float(cfg["vpin_bucket_size_usdt"][sym]) for sym in SYMBOLS}


def main() -> int:
    cfg = _load_settings()
    enabled = cfg.get("vpin_collector_enabled", False)
    if not enabled:
        logger.info("vpin_collector_enabled=false dans bot_settings.yaml — exit clean")
        return 0

    bucket_sizes = _ensure_bucket_sizes(cfg)
    logger.info(f"starting VPIN live collector for {SYMBOLS} → {STORE}")
    logger.info(f"bucket sizes: {bucket_sizes}")

    collector = VPINLiveCollectorWS(
        symbols=list(SYMBOLS),
        bucket_sizes=bucket_sizes,
        store_path=STORE,
    )

    stop_requested = {"flag": False}

    def _handle_sigterm(signum, frame):
        logger.info(f"received signal {signum} — shutting down")
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    collector.start()
    last_heartbeat = time.time()
    try:
        while not stop_requested["flag"]:
            time.sleep(1.0)
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                stats = collector.stats()
                logger.info(f"heartbeat stats={stats}")
                last_heartbeat = now
    finally:
        collector.stop(drain=True)
        logger.info("daemon exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
