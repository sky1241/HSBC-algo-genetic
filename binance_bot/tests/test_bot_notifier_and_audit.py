"""Tests pour TelegramNotifier (B14) et AuditLog (B15)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from bot.notifier import CRITICAL, INFO, WARN, TelegramNotifier
from bot.audit_log import AuditLog, ChainStatus, GENESIS_HASH


# ============================================================
# B14 — TelegramNotifier
# ============================================================

def test_notifier_not_configured_when_no_token():
    n = TelegramNotifier(bot_token="", chat_id="")
    assert n.configured is False


def test_notifier_configured_when_both_set():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    assert n.configured is True


def test_notifier_send_no_op_when_not_configured():
    """Pas de token → ne crash pas, retourne False, log seulement."""
    n = TelegramNotifier(bot_token="", chat_id="")
    with patch("bot.notifier.requests.post") as p:
        assert n.info("hello") is False
        assert n.warn("warning") is False
        assert n.critical("crit") is False
        p.assert_not_called()


def test_notifier_sends_to_telegram_when_configured():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"ok": True}
    with patch("bot.notifier.requests.post", return_value=fake_resp) as p:
        ok = n.critical("kill switch fired")
    assert ok is True
    p.assert_called_once()
    url = p.call_args.args[0]
    body = p.call_args.kwargs["json"]
    assert url == "https://api.telegram.org/bott/sendMessage"
    assert body["chat_id"] == "c"
    assert "kill switch fired" in body["text"]
    assert "🛑" in body["text"]
    assert body["disable_notification"] is False  # CRITICAL bruyant


def test_notifier_info_is_silent():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"ok": True}
    with patch("bot.notifier.requests.post", return_value=fake_resp) as p:
        n.info("trade ouvert")
    body = p.call_args.kwargs["json"]
    assert body["disable_notification"] is True


def test_notifier_handles_telegram_failure_gracefully():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    with patch("bot.notifier.requests.post", side_effect=Exception("net down")):
        # Ne doit pas raise — juste retourner False
        assert n.warn("test") is False


def test_notifier_handles_telegram_non_200():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    fake_resp = MagicMock()
    fake_resp.status_code = 429
    fake_resp.text = "Too Many Requests"
    with patch("bot.notifier.requests.post", return_value=fake_resp):
        assert n.warn("test") is False


def test_notifier_heartbeat():
    n = TelegramNotifier(bot_token="t", chat_id="c")
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"ok": True}
    with patch("bot.notifier.requests.post", return_value=fake_resp) as p:
        n.heartbeat("uptime 24h, balance 5000")
    assert p.called
    body = p.call_args.kwargs["json"]
    assert "heartbeat OK" in body["text"]
    assert "uptime 24h" in body["text"]


# ============================================================
# B15 — AuditLog hash chain
# ============================================================

@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "trades.jsonl"


def test_audit_first_entry_uses_genesis_prev_hash(audit_path):
    al = AuditLog(audit_path)
    e = al.append({"action": "open_long", "qty": 0.001})
    assert e["seq"] == 0
    assert e["prev_hash"] == GENESIS_HASH


def test_audit_chain_links_entries(audit_path):
    al = AuditLog(audit_path)
    e1 = al.append({"action": "open_long"})
    e2 = al.append({"action": "close_long"})
    assert e2["seq"] == 1
    assert e2["prev_hash"] == e1["hash"]


def test_audit_verify_chain_valid_after_appends(audit_path):
    al = AuditLog(audit_path)
    al.append({"a": 1})
    al.append({"b": 2})
    al.append({"c": 3})
    status = al.verify_chain()
    assert status.valid is True
    assert status.n_entries == 3
    assert status.error is None


def test_audit_verify_detects_payload_tampering(audit_path):
    al = AuditLog(audit_path)
    al.append({"action": "open_long", "qty": 0.001})
    al.append({"action": "close_long", "qty": 0.001})

    # Tamper: modify the payload of entry 0 manually
    lines = audit_path.read_text().splitlines()
    import json
    e0 = json.loads(lines[0])
    e0["payload"]["qty"] = 999  # mutation
    lines[0] = json.dumps(e0, sort_keys=True)
    audit_path.write_text("\n".join(lines) + "\n")

    status = al.verify_chain()
    assert status.valid is False
    assert "hash mismatch" in (status.error or "")


def test_audit_verify_detects_seq_skip(audit_path):
    al = AuditLog(audit_path)
    al.append({"x": 1})
    al.append({"x": 2})
    al.append({"x": 3})
    # Supprime la ligne 1 (entry middle)
    lines = audit_path.read_text().splitlines()
    audit_path.write_text(lines[0] + "\n" + lines[2] + "\n")
    status = al.verify_chain()
    assert status.valid is False


def test_audit_empty_file_is_valid(audit_path):
    al = AuditLog(audit_path)
    status = al.verify_chain()
    assert status.valid is True
    assert status.n_entries == 0
    assert status.last_hash == GENESIS_HASH


def test_audit_iter_entries_yields_in_order(audit_path):
    al = AuditLog(audit_path)
    al.append({"k": "a"})
    al.append({"k": "b"})
    payloads = [e["payload"]["k"] for e in al.iter_entries()]
    assert payloads == ["a", "b"]


def test_audit_canonical_serialization_is_stable(audit_path):
    """Même payload (clés ordre différent) ⇒ même hash."""
    al = AuditLog(audit_path)
    e1 = al.append({"a": 1, "b": 2})
    al2 = AuditLog(audit_path.with_suffix(".2.jsonl"))
    e2 = al2.append({"b": 2, "a": 1})  # ordre inversé
    # même prev (genesis) + même canonical ⇒ même hash
    assert e1["hash"] == e2["hash"]
