"""Gửi cảnh báo: Telegram (TELEGRAM_TOKEN + TELEGRAM_CHAT_ID) hoặc chỉ ghi log."""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("fraudrt.alerts")


def alert_text(tx: dict, score: float) -> str:
    return (f"Giao dịch nghi gian lận #{tx['tx_id']} (điểm {score:.2f}): {tx['type']} {float(tx['amount']):,.0f} "
            f"từ {tx['orig']} đến {tx['dest']}")  # fmt: skip


class LogAlerter:
    def __init__(self):
        self.sent = []

    def __call__(self, tx: dict, score: float) -> None:
        self.sent.append(alert_text(tx, score))
        log.warning(self.sent[-1])


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str, max_per_minute: int = 20):
        self.url, self.chat_id = f"https://api.telegram.org/bot{token}/sendMessage", chat_id
        self.max_per_minute, self.window, self.count = max_per_minute, 0, 0

    def __call__(self, tx: dict, score: float) -> None:
        import time

        minute = int(time.time() // 60)
        if minute != self.window:
            self.window, self.count = minute, 0
        self.count += 1
        if self.count > self.max_per_minute:  # không để một đợt tấn công làm ngập nhóm (Telegram cũng giới hạn)
            return
        try:
            httpx.post(self.url, json={"chat_id": self.chat_id, "text": alert_text(tx, score)}, timeout=5)
        except httpx.HTTPError:
            log.exception("không gửi được cảnh báo Telegram")


def from_env():
    if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        return TelegramAlerter(os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"])
    return LogAlerter()
