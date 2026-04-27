"""Tests pour backup_runner — B18."""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from backup_runner import (
    create_backup,
    rotate_old_backups,
    verify_backup,
    CRITICAL_FILES,
)


@pytest.fixture
def fake_repo(tmp_path):
    """Construit un mini-repo avec quelques fichiers critiques."""
    repo = tmp_path / "repo"
    for rel in CRITICAL_FILES:
        full = repo / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(f"content of {rel}")
    return repo


def test_create_backup_produces_targz(fake_repo, tmp_path):
    bk_dir = tmp_path / "backups"
    bk = create_backup(repo_root=fake_repo, backup_dir=bk_dir)
    assert bk.exists()
    assert bk.suffix == ".gz"
    assert bk.stat().st_size > 0


def test_backup_contains_all_critical_files(fake_repo, tmp_path):
    bk_dir = tmp_path / "backups"
    bk = create_backup(repo_root=fake_repo, backup_dir=bk_dir)
    with tarfile.open(bk, "r:gz") as tar:
        names = set(tar.getnames())
    for rel in CRITICAL_FILES:
        assert rel in names


def test_verify_backup_valid(fake_repo, tmp_path):
    bk_dir = tmp_path / "backups"
    bk = create_backup(repo_root=fake_repo, backup_dir=bk_dir)
    info = verify_backup(bk, repo_root=fake_repo)
    assert info["ok"] is True
    assert info["n_members"] == len(CRITICAL_FILES)
    assert info["missing"] == []


def test_verify_backup_detects_missing(fake_repo, tmp_path):
    """Si un critical file existe dans le repo mais manque dans le tarball → flag."""
    bk_dir = tmp_path / "backups"
    bk = create_backup(repo_root=fake_repo, backup_dir=bk_dir)
    # Reconstruit un tarball avec un fichier manquant
    incomplete = tmp_path / "incomplete.tar.gz"
    with tarfile.open(incomplete, "w:gz") as tar:
        tar.add(fake_repo / CRITICAL_FILES[0], arcname=CRITICAL_FILES[0])
    info = verify_backup(incomplete, repo_root=fake_repo)
    assert info["ok"] is False
    assert len(info["missing"]) == len(CRITICAL_FILES) - 1


def test_create_backup_skips_missing_files(tmp_path):
    """Si un critical file n'existe pas dans le repo, on continue (skip)."""
    repo = tmp_path / "repo"
    # Repo avec UN seul fichier
    f = repo / CRITICAL_FILES[0]
    f.parent.mkdir(parents=True)
    f.write_text("x")

    bk = create_backup(repo_root=repo, backup_dir=tmp_path / "bk")
    with tarfile.open(bk, "r:gz") as tar:
        names = tar.getnames()
    assert CRITICAL_FILES[0] in names
    assert len(names) == 1


def test_rotate_old_backups_removes_old_files(tmp_path):
    bk_dir = tmp_path / "backups"
    bk_dir.mkdir()
    # Crée un fichier ancien
    old = bk_dir / "hsbc-bot-2024-01-01T00-00-00Z.tar.gz"
    old.write_text("old")
    import os
    import time
    # mtime = 30 jours ago
    old_ts = time.time() - 30 * 86400
    os.utime(old, (old_ts, old_ts))

    # Crée un fichier récent
    new = bk_dir / "hsbc-bot-recent.tar.gz"
    new.write_text("new")

    removed = rotate_old_backups(bk_dir, retention_days=7)
    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_rotate_no_op_on_empty_dir(tmp_path):
    """Pas de backups → 0 removed, pas de crash."""
    n = rotate_old_backups(tmp_path / "absent", retention_days=7)
    assert n == 0


def test_full_round_trip_extract_recovers_content(fake_repo, tmp_path):
    """Backup + extract redonne les mêmes fichiers, byte-pour-byte."""
    bk_dir = tmp_path / "backups"
    bk = create_backup(repo_root=fake_repo, backup_dir=bk_dir)
    extract_to = tmp_path / "restored"
    extract_to.mkdir()
    with tarfile.open(bk, "r:gz") as tar:
        tar.extractall(extract_to)
    # Chaque fichier restauré doit exister et avoir le bon contenu
    for rel in CRITICAL_FILES:
        original = fake_repo / rel
        restored = extract_to / rel
        assert restored.exists(), f"{rel} not restored"
        assert restored.read_text() == original.read_text()
