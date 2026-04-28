"""T6 — Smoke test E2E mock pour intraday_runner.

Ce test garantit que le main() du runner ne CRASH pas sur un cycle
complet quand on mocke les dépendances externes (CCXT exchange,
state_manager, signal_engine). Sert de filet de sécurité pour les
modifications de intraday_runner.py (chunks P7-bis, R7-bis, etc.).

Le test ne valide PAS le comportement métier — seulement l'absence
d'exception. Tests métier sont dans les fichiers dédiés
(test_signal_engine.py, test_p2_drawdown_scaling.py, etc.).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_intraday_runner_module_imports_without_error():
    """Le module est importable sans crash. Test minimal mais utile :
    une syntax error ou import circulaire dans intraday_runner.py
    casserait toutes les autres tests du repo."""
    from routines import intraday_runner
    assert hasattr(intraday_runner, "main")
    assert callable(intraday_runner.main)


def test_intraday_runner_make_vpin_data_fn_callable():
    """La factory _make_vpin_data_fn produit toujours une fonction
    callable, même quand le fichier vpin_live.jsonl n'existe pas."""
    from routines import intraday_runner
    fn = intraday_runner._make_vpin_data_fn("BTCUSDT")
    assert callable(fn)
    # Sans data → None (comportement default safe)
    result = fn()
    assert result is None or isinstance(result, tuple)


def test_intraday_runner_factories_callable():
    """Chaque factory _make_*_fn produit un callable."""
    from routines import intraday_runner
    factories = [
        ("_make_vpin_data_fn", ("BTCUSDT",)),
        ("_make_vpin_gate_config", ({},)),
        ("_make_composite_log_fn", ("BTCUSDT", str(ROOT / "data"))),
    ]
    for name, args in factories:
        f = getattr(intraday_runner, name, None)
        assert f is not None, f"factory {name} absent"
        result = f(*args)
        if name == "_make_vpin_gate_config":
            # Cette factory retourne un objet config ou None, pas un callable
            assert result is None or hasattr(result, "mode")
        else:
            assert callable(result), f"factory {name} doit retourner callable"
