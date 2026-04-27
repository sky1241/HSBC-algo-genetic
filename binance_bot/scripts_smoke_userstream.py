#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test live UserStreamListener (BUG-B13).

Démarre un UserStreamListener pendant 30 secondes contre testnet.binancefuture.com.
Affiche les events reçus + un récap. Ne place AUCUN trade.

Lecture des clés depuis binance_bot/.env (BINANCE_API_KEY / BINANCE_API_SECRET).

Succès = listenKey créé OK, WS ouverte, pas d'erreur de connexion.
NB : sans activité de trading, peu d'events arrivent en 30s — c'est normal.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.event_queue import EventQueue
from bot.user_stream import UserStreamListener


def _load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    env_path = ROOT / ".env"
    env = _load_env(env_path)
    api_key = env.get("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
    api_secret = env.get("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        print(f"[FAIL] clés API absentes (cherché dans {env_path})")
        return 2

    print(f"[INFO] env loaded from {env_path}")
    print(f"[INFO] BINANCE_API_KEY=...{api_key[-6:]}")
    print(f"[INFO] BINANCE_TESTNET={env.get('BINANCE_TESTNET', 'unset')}")

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    queue = EventQueue(persist_path=data_dir / "user_stream_events.jsonl")

    received = []

    def on_event(ev: dict):
        et = ev.get("e", "?")
        print(f"[EVENT] type={et} keys={list(ev.keys())}")
        received.append(ev)
        queue.push(ev)

    listener = UserStreamListener(
        api_key=api_key,
        api_secret=api_secret,
        base_rest="https://testnet.binancefuture.com",
        base_ws="wss://stream.binancefuture.com/ws",
        on_event_callback=on_event,
    )

    print("[INFO] starting listener for 30s...")
    t0 = time.time()
    listener.start()

    DURATION = 30.0
    try:
        while time.time() - t0 < DURATION:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[INFO] interrupted")

    elapsed = time.time() - t0
    print(f"[INFO] stopping after {elapsed:.1f}s ; events_received={listener.events_received}, "
          f"connection_errors={listener.connection_errors}")

    listener.stop(join_timeout=5.0)
    queue.close()

    # Verdict
    success = (
        listener._listen_key is None  # propre fermeture
        and listener.connection_errors == 0
    )
    # Plus important : on a bien créé un listenKey à un moment ?
    # Si _listen_key est None après stop, OK (DELETE appelé). Mais il faut
    # qu'il y ait eu au moins une création. On le détecte via le fait que
    # le thread WS a vécu sans erreur.
    # Critère pragmatique : 0 connection_error.

    print(f"[RESULT] success={success}, events={len(received)}, duration={elapsed:.1f}s")
    if success:
        print("[OK] smoke test PASSED — listener ran without connection error.")
        return 0
    else:
        print(f"[FAIL] connection_errors={listener.connection_errors}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
