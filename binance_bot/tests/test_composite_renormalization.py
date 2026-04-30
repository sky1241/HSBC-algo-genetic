"""COMPOSITE-001 — Tests de renormalisation dynamique des poids.

Vérifie que `compute_composite_score` :
  1. Garde les poids de design quand 4 features sont actives
  2. Renormalise les poids restants à somme=1.0 quand liq absent
  3. Préserve la valeur du score quand toutes les composantes sont égales
     (preuve mathématique de la correction de la renormalisation)
  4. Retourne degraded=True quand au moins 1 feature est manquante
  5. Gère le cas "aucune feature active" sans crasher

La signature `compute_composite_score` prend des chemins de fichiers JSONL
(pas des scalaires comme initialement spécifié). On utilise donc la fixture
pytest `tmp_path` pour générer des JSONL synthétiques.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from binance_bot.services.flow_composite_signal import compute_composite_score


# ---------------------------------------------------------------------------
# Helpers : générateurs de JSONL synthétiques
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _gen_top_ls_records(n: int, base_long_account: float, now_ms: int) -> list[dict]:
    """Génère N records P6.1 top_ls avec longAccount oscillant autour de base."""
    out = []
    for i in range(n):
        # Petit bruit déterministe pour avoir std > 0
        noise = ((i % 5) - 2) * 0.01
        out.append({
            "ts_ms": now_ms - (n - i) * 300_000,  # 1 record / 5 min
            "long_account": max(0.0, min(1.0, base_long_account + noise)),
        })
    return out


def _gen_taker_records(n: int, base_ratio: float, now_ms: int) -> list[dict]:
    out = []
    for i in range(n):
        noise = ((i % 5) - 2) * 0.02
        out.append({
            "ts_ms": now_ms - (n - i) * 300_000,
            "buy_sell_ratio": max(0.1, base_ratio + noise),
        })
    return out


def _gen_oi_records(n: int, base_oi: float, now_ms: int) -> list[dict]:
    """sum_oi varie linéairement pour produire un OI change pct calculable.

    n doit être > _DEFAULT_OI_LOOKBACK_24H_BARS=288 pour que le shift fonctionne.
    """
    out = []
    for i in range(n):
        # OI croît légèrement avec bruit
        out.append({
            "ts_ms": now_ms - (n - i) * 300_000,
            "sum_oi": base_oi * (1.0 + 0.001 * i + 0.0005 * (i % 3)),
        })
    return out


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Tests : renormalisation dynamique
# ---------------------------------------------------------------------------


def test_composite_full_features_design_weights_preserved(tmp_path):
    """4 features actives → poids identiques au design (0.30/0.25/0.25/0.20)."""
    now = _now_ms()
    n = 400  # > 288 (lookback OI) ET > 100 (min_obs zscore_last)

    _write_jsonl(tmp_path / "top_ls.jsonl", _gen_top_ls_records(n, 0.5, now))
    _write_jsonl(tmp_path / "taker.jsonl", _gen_taker_records(n, 1.0, now))
    # Liq : 200 records avec long_liq + short_liq variant
    liq_records = [
        {"ts_ms": now - (200 - i) * 300_000,
         "long_liq_notional": 1000.0 + i * 10,
         "short_liq_notional": 800.0 + i * 8}
        for i in range(200)
    ]
    _write_jsonl(tmp_path / "liq.jsonl", liq_records)
    _write_jsonl(tmp_path / "oi.jsonl", _gen_oi_records(n, 1_000_000.0, now))

    result = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "taker.jsonl",
        liq_path=tmp_path / "liq.jsonl",
        oi_path=tmp_path / "oi.jsonl",
        days_back=30,
    )

    assert result["weights"]["top_ls"] == pytest.approx(0.30)
    assert result["weights"]["taker"] == pytest.approx(0.25)
    assert result["weights"]["liq"] == pytest.approx(0.25)
    assert result["weights"]["oi"] == pytest.approx(0.20)
    assert result["degraded"] is False
    assert set(result["active_features"]) == {"top_ls", "taker", "liq", "oi"}
    assert sum(result["weights"].values()) == pytest.approx(1.0)


def test_composite_liq_missing_renormalizes_to_sum_1(tmp_path):
    """liq absent (fichier vide) → poids des 3 autres renormalisés à somme=1.0.

    Vérifie aussi les nouvelles proportions :
      top_ls : 0.30 / (0.30 + 0.25 + 0.20) = 0.30 / 0.75 = 0.40
      taker  : 0.25 / 0.75 = 0.333...
      oi     : 0.20 / 0.75 = 0.266...
    """
    now = _now_ms()
    n = 400

    _write_jsonl(tmp_path / "top_ls.jsonl", _gen_top_ls_records(n, 0.5, now))
    _write_jsonl(tmp_path / "taker.jsonl", _gen_taker_records(n, 1.0, now))
    # Pas de fichier liq (équivalent WS-001) → on passe un path inexistant
    _write_jsonl(tmp_path / "oi.jsonl", _gen_oi_records(n, 1_000_000.0, now))

    result = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",  # n'existe pas
        oi_path=tmp_path / "oi.jsonl",
        days_back=30,
    )

    assert result["degraded"] is True
    assert "liq" not in result["weights"]
    assert "liq" not in result["components"]
    assert set(result["active_features"]) == {"top_ls", "taker", "oi"}
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["weights"]["top_ls"] == pytest.approx(0.40, abs=1e-9)
    assert result["weights"]["taker"] == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert result["weights"]["oi"] == pytest.approx(0.20 / 0.75, abs=1e-9)
    # Les n_obs doivent toujours rapporter les 4 (pour audit drift)
    assert "liq" in result["n_obs"]
    assert result["n_obs"]["liq"] == 0


def test_composite_score_invariant_when_all_components_equal(tmp_path, monkeypatch):
    """Mathématique : si toutes les composantes ACTIVES ont le même z-score k,
    le score composite vaut k (somme des poids normalisés × k = 1 × k = k),
    indépendamment du nombre de features actives.

    On vérifie en mockant zscore_last pour qu'il retourne toujours 0.5.
    """
    from binance_bot.services import flow_composite_signal as mod

    monkeypatch.setattr(mod, "zscore_last", lambda *a, **kw: 0.5)

    now = _now_ms()
    n = 400

    # Cas 1 : 4 features actives
    _write_jsonl(tmp_path / "top_ls.jsonl", _gen_top_ls_records(n, 0.5, now))
    _write_jsonl(tmp_path / "taker.jsonl", _gen_taker_records(n, 1.0, now))
    liq_records = [
        {"ts_ms": now - (200 - i) * 300_000,
         "long_liq_notional": 1000.0 + i * 10,
         "short_liq_notional": 800.0 + i * 8}
        for i in range(200)
    ]
    _write_jsonl(tmp_path / "liq.jsonl", liq_records)
    _write_jsonl(tmp_path / "oi.jsonl", _gen_oi_records(n, 1_000_000.0, now))

    r4 = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "taker.jsonl",
        liq_path=tmp_path / "liq.jsonl",
        oi_path=tmp_path / "oi.jsonl",
        days_back=30,
    )

    # Cas 2 : 3 features actives (liq absente)
    r3 = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",
        oi_path=tmp_path / "oi.jsonl",
        days_back=30,
    )

    assert r4["score"] == pytest.approx(0.5, abs=1e-9)
    assert r3["score"] == pytest.approx(0.5, abs=1e-9)
    assert r4["score"] == pytest.approx(r3["score"], abs=1e-9)
    assert r3["degraded"] is True
    assert r4["degraded"] is False


def test_composite_no_active_features_returns_zero_degraded(tmp_path):
    """Tous les fichiers absents ou vides → score=0, degraded=True, components={}."""
    result = compute_composite_score(
        top_ls_path=tmp_path / "no_top.jsonl",
        taker_path=tmp_path / "no_taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",
        oi_path=tmp_path / "no_oi.jsonl",
    )
    assert result["score"] == 0.0
    assert result["raw_score"] == 0.0
    assert result["components"] == {}
    assert result["weights"] == {}
    assert result["active_features"] == []
    assert result["degraded"] is True
    # base_weights et n_obs toujours présents pour audit
    assert result["base_weights"] == {"top_ls": 0.30, "taker": 0.25, "liq": 0.25, "oi": 0.20}
    assert result["n_obs"] == {"top_ls": 0, "taker": 0, "liq": 0, "oi": 0}


def test_composite_audit_trail_base_weights_preserved(tmp_path):
    """`base_weights` doit TOUJOURS contenir les 4 poids de design,
    même quand `weights` ne contient que les actifs renormalisés.
    Permet d'auditer rétrospectivement la dégradation.
    """
    now = _now_ms()
    n = 400

    _write_jsonl(tmp_path / "top_ls.jsonl", _gen_top_ls_records(n, 0.5, now))
    _write_jsonl(tmp_path / "taker.jsonl", _gen_taker_records(n, 1.0, now))
    _write_jsonl(tmp_path / "oi.jsonl", _gen_oi_records(n, 1_000_000.0, now))

    result = compute_composite_score(
        top_ls_path=tmp_path / "top_ls.jsonl",
        taker_path=tmp_path / "taker.jsonl",
        liq_path=tmp_path / "no_liq.jsonl",
        oi_path=tmp_path / "oi.jsonl",
        days_back=30,
    )

    expected_base = {"top_ls": 0.30, "taker": 0.25, "liq": 0.25, "oi": 0.20}
    assert result["base_weights"] == expected_base
    assert sum(result["base_weights"].values()) == pytest.approx(1.0)
    # Les weights effectifs (renormalisés) ne contiennent PAS liq
    assert "liq" not in result["weights"]
    # Mais base_weights le contient toujours pour traçabilité
    assert "liq" in result["base_weights"]
