#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backup runner — B18.

Crée un tarball horodaté des fichiers critiques du bot (state, audit log, paper log)
et le sauvegarde hors du repo + rotation 7 jours.

Optionnel: upload vers Backblaze B2 si BACKBLAZE_B2_KEY_ID + BACKBLAZE_B2_APP_KEY
+ BACKBLAZE_B2_BUCKET sont set dans .env. Pour cette session, seulement local.

Usage: lancé par systemd timer toutes les 6h.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BACKUP_DIR = Path.home() / "Backup" / "hsbc-bot"
RETENTION_DAYS = 7

CRITICAL_FILES = [
    "binance_bot/data/state.json",
    "binance_bot/data/trades_audit.jsonl",
    "binance_bot/data/paper_log.csv",
    "binance_bot/data/balance_history.jsonl",
    "binance_bot/data/health.json",
    "binance_bot/configs/bot_settings.yaml",
    "BUGS.md",
]


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def create_backup(repo_root: Path = REPO, backup_dir: Path = BACKUP_DIR) -> Path:
    """Crée un .tar.gz avec les fichiers critiques. Retourne le path."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    out_path = backup_dir / f"hsbc-bot-{stamp}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in CRITICAL_FILES:
            full = repo_root / rel
            if full.exists():
                tar.add(full, arcname=rel)
            else:
                print(f"  ⚠️ skip missing: {rel}")
    return out_path


def rotate_old_backups(backup_dir: Path = BACKUP_DIR, retention_days: int = RETENTION_DAYS) -> int:
    """Supprime les backups > retention_days. Retourne le nb supprimés."""
    if not backup_dir.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed = 0
    for f in backup_dir.glob("hsbc-bot-*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception as e:
            print(f"  ⚠️ rotation failed for {f}: {e}")
    return removed


def upload_to_b2(local_path: Path) -> bool:
    """Upload optional vers Backblaze B2 si credentials dans .env.

    Requiert b2 CLI ou b2sdk. Si pas dispo, skip silencieusement.
    """
    key_id = os.environ.get("BACKBLAZE_B2_KEY_ID")
    app_key = os.environ.get("BACKBLAZE_B2_APP_KEY")
    bucket = os.environ.get("BACKBLAZE_B2_BUCKET")
    if not (key_id and app_key and bucket):
        return False
    # Vérifie que b2 CLI est dispo
    try:
        subprocess.check_output(["b2", "version"], stderr=subprocess.STDOUT, timeout=5)
    except Exception:
        print("  ⚠️ B2 CLI absent — skip cloud upload (install: pip install b2)")
        return False
    try:
        subprocess.check_call(
            ["b2", "authorize-account", key_id, app_key],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        subprocess.check_call(
            ["b2", "upload-file", bucket, str(local_path), local_path.name],
            timeout=120,
        )
        print(f"  ☁️  Uploaded to B2://{bucket}/{local_path.name}")
        return True
    except Exception as e:
        print(f"  ⚠️ B2 upload failed: {e}")
        return False


def verify_backup(backup_path: Path, repo_root: Path = REPO) -> dict:
    """Vérifie que le tar.gz est lisible et contient les fichiers attendus.

    Returns dict avec 'ok', 'n_members', 'missing'.
    """
    info = {"ok": False, "n_members": 0, "missing": []}
    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            members = tar.getnames()
            info["n_members"] = len(members)
            for rel in CRITICAL_FILES:
                full = repo_root / rel
                if full.exists() and rel not in members:
                    info["missing"].append(rel)
        info["ok"] = len(info["missing"]) == 0
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    print(f"📦 Backup HSBC bot — {_now_stamp()}")
    bk_path = create_backup()
    print(f"  Created: {bk_path} ({bk_path.stat().st_size} bytes)")

    verify = verify_backup(bk_path)
    print(f"  Verify: ok={verify['ok']}, members={verify['n_members']}, missing={verify['missing']}")
    if not verify["ok"]:
        print(f"  ❌ Backup verification FAILED")
        return 1

    upload_to_b2(bk_path)

    removed = rotate_old_backups()
    print(f"  Rotated: {removed} old backup(s) removed")

    # Snapshot pour le watchdog
    summary = {
        "last_backup": str(bk_path),
        "size_bytes": bk_path.stat().st_size,
        "verify_ok": verify["ok"],
        "ts": _now_stamp(),
    }
    snap = REPO / "binance_bot" / "data" / "backup_status.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
