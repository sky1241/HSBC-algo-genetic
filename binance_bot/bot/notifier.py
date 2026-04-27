#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Notifier multi-niveau (Telegram + log fallback) — B14.

Niveaux : INFO (trades), WARN (drawdown, latence), CRITICAL (kill switch, exception).
Si TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID sont set → Telegram + log.
Sinon → log uniquement (jamais bloquant).

Heartbeat: méthode `heartbeat()` à appeler périodiquement (e.g. par le watchdog).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

LOG = logging.getLogger("hsbc.notifier")
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

# Levels
INFO = "INFO"
WARN = "WARN"
CRITICAL = "CRITICAL"
ICONS = {INFO: "ℹ️", WARN: "⚠️", CRITICAL: "🛑"}


class TelegramNotifier:
    """Envoie sur Telegram si configuré, sinon no-op (log only)."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_s: float = 5.0,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, level: str, message: str) -> bool:
        """Envoie le message. Retourne True si Telegram a répondu OK, False sinon.

        Toujours log local (stdout/logger) même si Telegram down — pas bloquant.
        """
        icon = ICONS.get(level, "•")
        formatted = f"{icon} *{level}* — {message}"
        LOG.info("[%s] %s", level, message)

        if not self.configured:
            return False

        url = TELEGRAM_API.format(token=self.bot_token)
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": formatted,
                    "parse_mode": "Markdown",
                    "disable_notification": level == INFO,  # silent pour INFO
                },
                timeout=self.timeout_s,
            )
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            LOG.warning("Telegram non-200: %s — %s", resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            LOG.warning("Telegram failed: %s", e)
            return False

    def info(self, message: str) -> bool:
        return self.send(INFO, message)

    def warn(self, message: str) -> bool:
        return self.send(WARN, message)

    def critical(self, message: str) -> bool:
        return self.send(CRITICAL, message)

    def heartbeat(self, extra: str = "") -> bool:
        """Heartbeat silencieux ('bot alive')."""
        msg = "heartbeat OK"
        if extra:
            msg += f" — {extra}"
        return self.info(msg)


__all__ = ["TelegramNotifier", "INFO", "WARN", "CRITICAL"]
