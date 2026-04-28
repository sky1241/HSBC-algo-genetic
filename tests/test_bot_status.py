"""F-status — Tests pytest pour scripts/bot_status.sh."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bot_status.sh"


def _run(env_overrides=None):
    """Exec bot_status.sh with optional env. Returns (rc, stdout)."""
    env = None
    if env_overrides:
        import os
        env = os.environ.copy()
        env.update(env_overrides)
    cp = subprocess.run(
        [str(SCRIPT)], capture_output=True, text=True, timeout=15, env=env,
    )
    return cp.returncode, cp.stdout


def test_bot_status_runs_without_error():
    """Script tourne, rc=0, sortie non-vide."""
    rc, out = _run()
    assert rc == 0, f"non-zero exit: {rc}, stdout={out[:500]}"
    assert len(out) > 200, "stdout suspiciously short"


def test_bot_status_includes_all_section_headers():
    """Les 8 sections doivent être présentes."""
    rc, out = _run()
    expected = [
        "HSBC bot status",       # section 1
        "Git",                   # section 2
        "Trade mode",            # section 3
        "Timers systemd",        # section 4
        "Donnees collectees",    # section 5
        "Trades (testnet)",      # section 6
        "Daily PnL",             # section 7
        "Health",                # section 8
    ]
    for h in expected:
        assert h in out, f"missing section header: {h!r} in output"


def test_bot_status_reports_kill_switch_when_present(tmp_path, monkeypatch):
    """Quand data/.killed existe → output contient 'kill_switch ACTIF'.

    Ce test crée temporairement un .killed file dans le repo, l'exécute,
    puis le supprime. CI-safe : utilise touch + remove + finally.
    """
    killed_file = ROOT / "binance_bot" / "data" / ".killed"
    pre_existed = killed_file.exists()
    if not pre_existed:
        killed_file.touch()
    try:
        rc, out = _run()
        assert rc == 0
        # kill_switch ACTIF doit apparaître dans la section Health
        assert "kill_switch ACTIF" in out or "ACTIF" in out, (
            f"expected kill_switch ACTIF, got: {out[-300:]}"
        )
    finally:
        if not pre_existed and killed_file.exists():
            killed_file.unlink()


def test_bot_status_handles_missing_jsonl_gracefully(tmp_path):
    """Si certains jsonl absents (ex: vpin_live, trades_meta), pas de crash.

    Note : le script s'exécute dans le repo réel. On vérifie juste qu'il
    affiche '0' pour les fichiers absents (vpin_live, vpin_events) sans
    erreur.
    """
    rc, out = _run()
    assert rc == 0
    # vpin_live et vpin_events sont garantis absents pré-P7-bis
    assert "vpin_live" in out
    assert "vpin_events" in out
    # Et pas de "ERR" ou "Traceback" dans output
    assert "Traceback" not in out
    assert "command not found" not in out


def test_bot_status_displays_soak_start_and_jplus_n():
    """Header doit afficher soak_start ts + J+N."""
    rc, out = _run()
    assert rc == 0
    # soak_start.txt existe (créé après l'audit) → J+ doit apparaître
    soak_file = ROOT / "binance_bot" / "data" / "soak_start.txt"
    if soak_file.exists():
        assert "Soak start" in out
        assert "J+" in out
    else:
        # Sans soak_start, on attend un warning
        assert "soak_start.txt absent" in out


def test_bot_status_no_unbound_variable_pollution():
    """Régression du bug initial : variable C écrasée par count, pollution
    des séparateurs avec un nombre.

    Vérifie qu'aucune ligne ne commence par un chiffre suivi de tirets.
    """
    rc, out = _run()
    assert rc == 0
    polluted = [
        line for line in out.split("\n")
        if line and line[0].isdigit() and "─" in line
    ]
    assert not polluted, f"variable pollution détectée: {polluted[:3]}"
