"""Stage 3: the "send" step.

Both send_reminder and notify_restructuring_decision below share one real
channel: channels/telegram_bot.py, python-telegram-bot -- this project's
"zero real outbound-channel credentials" was true once, not any more.
Each sends for real when the account has a linked telegram_chat_id
(accounts.store.set_telegram_chat_id, written once a borrower verifies
over Telegram), and falls back to a logged event -- visible to an
operator via observability/metrics.py even with nowhere to actually
deliver to -- when there's no chat to reach (a browser-only borrower, or
one who never verified over Telegram).

send_reminder itself is still synchronous -- its callers (outbound/run.py,
scripts/run_outbound_pass.py, scripts/run_outbound_scheduler.py) are none
of them already inside an event loop, so asyncio.run() here is safe (this
is exactly the nested-run() crash channels/telegram_bot.py hit earlier
this session, which only happens when the caller already has a loop).
"""

import asyncio
import logging
import os

from telegram import Bot
from telegram.error import TelegramError

from businessflow.accounts import store

logger = logging.getLogger(__name__)


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


async def _deliver_and_log(account_id: str, message: str, event_type: str, extra_details: dict) -> bool:
    account = store.get_account(account_id)
    delivered = False
    if account and account.telegram_chat_id:
        delivered = await _send_telegram_message(account.telegram_chat_id, message)

    store.log_event(account_id, event_type, {**extra_details, "message": message, "delivered_via_telegram": delivered})
    return delivered


def send_reminder(account_id: str, kind: str, message: str) -> bool:
    """Called from outbound/run.py's daily pass. Returns True if the
    borrower was actually reached over Telegram (see module docstring)."""
    return asyncio.run(_deliver_and_log(account_id, message, "reminder_sent", {"kind": kind}))


async def notify_restructuring_decision(account_id: str, approved: bool, message: str) -> bool:
    """Called from ops/api.py right after a human approves or rejects a
    restructuring request. Returns True if the borrower was actually
    reached over Telegram."""
    return await _deliver_and_log(account_id, message, "restructuring_decision_notified", {"approved": approved})
