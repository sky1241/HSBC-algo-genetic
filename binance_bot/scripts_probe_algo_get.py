#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe script: trouver les endpoints REST Binance USDM Futures pour
LISTER (GET) et ANNULER (DELETE) les algo orders conditionnels.

Contexte (2025-12-09+): Binance a migré STOP_MARKET / TAKE_PROFIT_MARKET
vers `/fapi/v1/algoOrder` (POST). Le POST marche dans le bot. On cherche
ici les bons paths GET et DELETE — toutes les variantes "conditional"
essayées renvoient `-5000 Path invalid`.

Findings (doc officielle confirmée 2026-04-26):
  - GET   /fapi/v1/openAlgoOrders        → liste les algo orders ouverts (open).
        Params: timestamp (req), symbol (opt, weight 1 sinon 40),
                 algoType (opt), algoId (opt), recvWindow (opt).
  - GET   /fapi/v1/algoOrder              → query un seul algo order.
        Params: timestamp (req), algoId | clientAlgoId (l'un des deux).
  - DELETE /fapi/v1/algoOrder             → annule un algo order.
        Params: timestamp (req), algoId | clientAlgoId.
  - GET   /fapi/v1/allAlgoOrders          → historique (active + canceled + finished).

Source: developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/
        Current-All-Algo-Open-Orders / Cancel-Algo-Order / Query-Algo-Order

NB: l'erreur `-5000 Path invalid` venait du fait que les noms étaient mal
orthographiés (camelCase et "open" attaché à "Algo"). Le bon path est
`openAlgoOrders` (Algo entre open et Orders).

Lancement:
    cd /home/ludov/HSBC-algo-genetic
    .venv/bin/python binance_bot/scripts_probe_algo_get.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from typing import Optional

import requests


API_KEY = os.environ.get(
    "BINANCE_TESTNET_KEY",
    "Wj6JhrTJZu2b6K3XHzZz2p16Q5UBXPX4UA3ZF5N01dDTdgmbX2lKqZ3ciMoT0Wpq",
)
API_SECRET = os.environ.get(
    "BINANCE_TESTNET_SECRET",
    "I53AmvRU5mlxZygS3wt9ZZ7zUai35MadMtKxK5S5jEgxb2LHVlmWbFJHk3NKSizB",
)
BASE = "https://testnet.binancefuture.com"


def _sign(params: dict) -> str:
    query = urllib.parse.urlencode(params)
    sig = hmac.new(
        API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    return query + "&signature=" + sig


def call(method: str, path: str, params: Optional[dict] = None) -> dict:
    """Appel signé. Retourne {http, body, ok}."""
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 5000
    qs = _sign(p)
    url = f"{BASE}{path}?{qs}"
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        resp = requests.request(method, url, headers=headers, timeout=10)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        ok = (
            isinstance(body, list)
            or (isinstance(body, dict) and not (body.get("code") and int(body.get("code", 0)) < 0))
        )
        return {"http": resp.status_code, "body": body, "ok": ok}
    except Exception as e:
        return {"http": -1, "body": str(e), "ok": False}


def main() -> int:
    print("=" * 70)
    print("PROBE Binance USDM Futures algo GET/DELETE endpoints")
    print(f"BASE: {BASE}")
    print("=" * 70)

    # ============================================================
    # 1) GET endpoints — paths plausibles
    # ============================================================
    candidates_get = [
        # Officiel (selon doc 2026-04)
        ("GET", "/fapi/v1/openAlgoOrders", {}),
        ("GET", "/fapi/v1/openAlgoOrders", {"symbol": "BTCUSDT"}),
        ("GET", "/fapi/v1/openAlgoOrders", {"algoType": "CONDITIONAL"}),
        ("GET", "/fapi/v1/openAlgoOrders", {"symbol": "BTCUSDT", "algoType": "CONDITIONAL"}),
        ("GET", "/fapi/v1/allAlgoOrders", {"symbol": "BTCUSDT"}),
        # Variantes au cas où
        ("GET", "/fapi/v1/algoOrder/openOrders", {}),
        ("GET", "/fapi/v1/algo/futures/openOrders", {}),
        ("GET", "/fapi/v2/algoOrder", {}),
        ("GET", "/papi/v1/algo/futures/openOrders", {}),  # portfolio margin
    ]

    survivors_get = []
    for method, path, params in candidates_get:
        res = call(method, path, params)
        tag = "OK " if res["ok"] else "ERR"
        body_str = json.dumps(res["body"])[:160] if not isinstance(res["body"], str) else res["body"][:160]
        print(f"[{tag}] {method:6s} {path:45s} params={params} → http={res['http']} body={body_str}")
        if res["ok"]:
            survivors_get.append((method, path, params, res["body"]))

    # ============================================================
    # 2) GET single algo order — sur les algoIds connus orphelins
    # ============================================================
    known_algo_ids = [
        1000000058811835,
        1000000058875483,
        1000000058875487,
        1000000058875704,
        1000000058875705,
    ]
    print()
    print("--- query /fapi/v1/algoOrder (single) ---")
    for aid in known_algo_ids[:2]:  # 2 suffisent pour confirmer
        res = call("GET", "/fapi/v1/algoOrder", {"algoId": aid})
        tag = "OK " if res["ok"] else "ERR"
        body_str = json.dumps(res["body"])[:200] if not isinstance(res["body"], str) else res["body"][:200]
        print(f"[{tag}] GET /fapi/v1/algoOrder algoId={aid} → http={res['http']} body={body_str}")

    # ============================================================
    # 3) NE PAS annuler (l'utilisateur veut conserver les orphelins
    #    pour vérifier 24h plus tard que le bot tourne sans erreur).
    #    On affiche juste le path validé.
    # ============================================================
    print()
    print("--- DELETE /fapi/v1/algoOrder (NON exécuté — orphelins préservés) ---")
    print("    Pour annuler: requests.delete(BASE + '/fapi/v1/algoOrder?' + signed)")
    print("    Params: algoId=<id> + timestamp + signature, header X-MBX-APIKEY.")

    # ============================================================
    # 4) Synthèse
    # ============================================================
    print()
    print("=" * 70)
    print("SURVIVORS (GET):")
    for m, p, params, body in survivors_get:
        n = len(body) if isinstance(body, list) else 1
        print(f"  - {m} {p} params={params} → {n} item(s)")
    print("=" * 70)
    return 0 if survivors_get else 1


if __name__ == "__main__":
    sys.exit(main())
