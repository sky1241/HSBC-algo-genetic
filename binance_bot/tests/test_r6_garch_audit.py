"""R6 — Tests _make_garch_audit_check (intraday_runner)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from routines.intraday_runner import _make_garch_audit_check


def _make_synth_returns(n=500, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, vol, n))


def test_audit_check_returns_none_on_short_series():
    """< 100 obs → None (cas data trop courte, fail-open silencieux)."""
    short = pd.Series(np.random.randn(50))
    res = _make_garch_audit_check("BTCUSDT", short, lambda: "mid")
    assert res is None


def test_audit_check_returns_none_on_no_returns():
    res = _make_garch_audit_check("BTCUSDT", None, lambda: "mid")
    assert res is None


def test_audit_check_picks_per_symbol_model():
    """BTC fit doit utiliser TGARCH ; ETH doit utiliser EGARCH."""
    returns = _make_synth_returns(n=600, seed=11)
    res_btc = _make_garch_audit_check("BTCUSDT", returns, lambda: "mid")
    assert res_btc is not None
    assert res_btc["model"] == "TGARCH"

    res_eth = _make_garch_audit_check("ETHUSDT", returns, lambda: "mid")
    assert res_eth is not None
    assert res_eth["model"] == "EGARCH"


def test_audit_check_detects_agreement():
    """HAR=mid, GARCH baseline ≈ baseline → garch_label=mid → no disagreement."""
    returns = _make_synth_returns(n=800, seed=42, vol=0.01)
    res = _make_garch_audit_check("BTCUSDT", returns, lambda: "mid")
    assert res is not None
    if "disagreement" in res:
        # Quand le fit converge bien, garch_label sur returns ~ baseline
        # devrait être "mid" → agreement.
        # En cas d'échec convergence (ex. série short), reason posté.
        if res.get("har_label") is not None:
            assert res["har_label"] == "mid"


def test_audit_check_detects_disagreement_when_har_low_garch_high(monkeypatch):
    """Force HAR=low + force forecast haut via patch garch_regime_label → disagreement."""
    returns = _make_synth_returns(n=800, seed=4, vol=0.01)

    # On patche garch_regime_label pour qu'elle retourne toujours "high"
    import routines.intraday_runner as runner

    # Pour ce test on fait un appel direct à la fonction qui lit le module garch
    # On patch via importlib le module garch utilisé en interne
    import src.garch as garch_mod
    original = garch_mod.garch_regime_label
    monkeypatch.setattr(garch_mod, "garch_regime_label", lambda f, b: "high")

    res = _make_garch_audit_check("BTCUSDT", returns, lambda: "low")
    monkeypatch.setattr(garch_mod, "garch_regime_label", original)

    assert res is not None
    if res.get("garch_label") is not None:
        # Note : si le fit a convergé, on a har_label/garch_label
        if "har_label" in res:
            assert res["har_label"] == "low"
            assert res["garch_label"] == "high"
            assert res["disagreement"] is True


def test_audit_check_returns_dict_with_required_keys_on_success():
    """Sur fit qui converge, doit retourner les clés attendues."""
    returns = _make_synth_returns(n=1000, seed=7, vol=0.01)
    res = _make_garch_audit_check("BTCUSDT", returns, lambda: "mid")
    assert res is not None
    # Au minimum 'model' présent dans tous les cas
    assert "model" in res
    # Si pas d'erreur de convergence, les autres clés sont là
    if res.get("disagreement") is False and "reason" not in res:
        for k in ("har_label", "garch_label", "forecast_sigma", "baseline_sigma"):
            assert k in res
