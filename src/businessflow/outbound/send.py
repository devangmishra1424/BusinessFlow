"""Stage 3: the "send" step.

send_reminder is still a clearly-labeled stub, same convention as
payment_tools.generate_payment_link's synthetic link -- the proactive
outbound-reminder feature has no real trigger wired to call it yet, so
there's nothing to actually deliver to.

notify_restructuring_decision below is different: a real channel now
exists (channels/telegram_bot.py, python-telegram-bot -- this project's
"zero real outbound-channel credentials" was true when send_reminder was
written, not any more), and this is that channel's first outbound (not
just reply-to-an-incoming-message) use. It sends for real when the
account has a linked telegram_chat_id (accounts.store.set_telegram_chat_id,
written once a borrower verifies over Telegram), and falls back to a
logged event -- same shape as send_reminder -- when there's no chat to
reach (a browser-only borrower, or one who never verified over Telegram).
"""

import logging
import os

from telegram import Bot
from telegram.error import TelegramError

from businessflow.accounts import store

logger = logging.getLogger(__name__)


def send_reminder(account_id: str, kind: str, message: str) -> None:
    store.log_event(account_id, "reminder_sent", {"kind": kind, "message": message})


async def _send_telegram_message(chat_id: int, text: str) -> bool:
    """Returns True only if Telegram actually accepted the message --
    False (not raised) if the token is missing, or Telegram itself
    rejects delivery (e.g. the borrower blocked the bot), so the caller
    can fall back to a logged event instead of losing the notification
    silently. A genuinely unexpected error is not this case and
    propagates, per this project's "don't swallow the unexpected" rule."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return False
    try:
        async with Bot(token=token) as bot:
            await bot.send_message(chat_id=chat_id, text=text)
        return True
    except TelegramError:
        logger.warning("notify: Telegram delivery failed for chat_id=%s", chat_id, exc_info=True)
        return False


async def notify_restructuring_decision(account_id: str, approved: bool, message: str) -> bool:
    """Called from ops/api.py right after a human approves or rejects a
    restructuring request. Returns True if the borrower was actually
    reached over Telegram -- always also logs the attempt (delivered or
    not) as a real event, so an operator can see what happened even when
    there was nowhere to actually send it."""
    account = store.get_account(account_id)
    delivered = False
    if account and account.telegram_chat_id:
        delivered = await _send_telegram_message(account.telegram_chat_id, message)

    store.log_event(
        account_id,
        "restructuring_decision_notified",
        {"approved": approved, "message": message, "delivered_via_telegram": delivered},
    )
    return delivered
