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

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from businessflow.accounts import store
from businessflow.channels.reminder_actions import action_rows

logger = logging.getLogger(__name__)


def _build_reply_markup(
    payment_url: str | None, payment_amount: float | None, actions: list[list[tuple[str, str]]] | None
) -> InlineKeyboardMarkup | None:
    """The "Pay now" URL button first (when there is a link), then one row per group of tappable actions."""
    rows: list[list[InlineKeyboardButton]] = []
    if payment_url:
        label = f"💳 Pay ₹{payment_amount:,.0f} now" if payment_amount is not None else "💳 Pay now"
        rows.append([InlineKeyboardButton(label, url=payment_url)])
    for row in actions or []:
        rows.append([InlineKeyboardButton(text, callback_data=data) for text, data in row])
    return InlineKeyboardMarkup(rows) if rows else None


async def _send_telegram_message(
    chat_id: int,
    text: str,
    payment_url: str | None = None,
    payment_amount: float | None = None,
    actions: list[list[tuple[str, str]]] | None = None,
) -> bool:
    """Returns True only if Telegram actually accepted the message --
    False (not raised) if the token is missing, or Telegram itself
    rejects delivery (e.g. the borrower blocked the bot), so the caller
    can fall back to a logged event instead of losing the notification
    silently. A genuinely unexpected error is not this case and
    propagates, per this project's "don't swallow the unexpected" rule.

    actions, when given, are rows of (label, callback_data) tappable
    buttons below the message (channels/reminder_actions.py builds them
    and handles the tap).

    payment_url, when given, attaches a real Telegram URL button below
    the message -- not woven into the message text itself (see
    outbound/compose.py's own docstring on why), and not a callback-data
    button either: this needs to open a real page (channels/browser_api.py's
    /pay/{token}), which only a URL button can do."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return False
    reply_markup = _build_reply_markup(payment_url, payment_amount, actions)
    try:
        async with Bot(token=token) as bot:
            await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
        return True
    except TelegramError:
        logger.warning("notify: Telegram delivery failed for chat_id=%s", chat_id, exc_info=True)
        return False


async def notify_telegram_chat(chat_id: int, text: str) -> bool:
    """A plain message to one chat, not about any one case (e.g. "this account is now linked elsewhere"). True if
    Telegram accepted it."""
    return await _send_telegram_message(chat_id, text)


async def _deliver_and_log(
    account_id: str,
    message: str,
    event_type: str,
    extra_details: dict,
    payment_url: str | None = None,
    payment_amount: float | None = None,
    with_actions: bool = False,
    telegram_text: str | None = None,
) -> bool:
    """telegram_text, when given, is what Telegram shows instead of message -- for a message that needs context
    there (which case it is about) that the dashboard already gives by showing it inside that case."""
    account = store.get_account(account_id)
    delivered = False
    if account and account.telegram_chat_id:
        actions = action_rows(account_id, account.language_preference) if with_actions else None
        delivered = await _send_telegram_message(
            account.telegram_chat_id, telegram_text or message, payment_url, payment_amount, actions
        )

    store.log_event(account_id, event_type, {**extra_details, "message": message, "delivered_via_telegram": delivered})
    return delivered


def send_reminder(
    account_id: str, kind: str, message: str, payment_url: str | None = None, payment_amount: float | None = None,
    tone: str | None = None,
) -> bool:
    """Called from outbound/run.py's daily pass. Returns True if the
    borrower was actually reached over Telegram (see module docstring).
    payment_url/payment_amount are optional -- run.py only mints a real
    payment token (and passes both) for the reminder kinds where "pay
    now" actually makes sense (see run.py itself for which). Every
    reminder also carries the tappable actions of
    channels/reminder_actions.py (promise to pay, talk to a person,
    dispute), handled in telegram_bot.on_callback_query."""
    return asyncio.run(
        # tone: the risk engine's choice, so reminders can be judged by whether payment followed (trace_report.py)
        _deliver_and_log(account_id, message, "reminder_sent", {"kind": kind, "tone": tone}, payment_url, payment_amount,
                         with_actions=True)
    )


async def notify_restructuring_decision(account_id: str, approved: bool, message: str, case_ref: str | None = None) -> bool:
    """Called from ops/api.py right after a human approves or rejects a
    restructuring request. Returns True if the borrower was actually
    reached over Telegram. case_ref (the escalation id) puts the message in
    that request's thread on the borrower's dashboard."""
    details = {"approved": approved, **({"case_ref": case_ref} if case_ref else {})}
    return await _deliver_and_log(account_id, message, "restructuring_decision_notified", details)


async def notify_case_message(account_id: str, case_ref: str, message: str, about: str, reply_in_chat: bool) -> bool:
    """A message from ops about one case (accounts/cases.py) -- shown in that
    case's thread on both dashboards, and sent over Telegram when linked,
    prefixed with what it is about (`about`, e.g. "the payment you
    reported"), since a Telegram chat has no thread to put it in. Returns True
    if the borrower was actually reached over Telegram.

    reply_in_chat: whether a reply typed in the Telegram chat reaches this
    case. True for a dispute or payment claim (the agent adds the borrower's
    further details to the open one, see flag_dispute); False for an
    escalation, where only the dashboard's reply box does -- so the footer
    never points the borrower somewhere their reply would get lost."""
    footer = "You can reply here, or on your dashboard." if reply_in_chat else "You can reply on your dashboard, under Requests & messages."
    telegram_text = f"Our team, about {about}:\n{message}\n\n{footer}"
    return await _deliver_and_log(
        account_id, message, "case_message", {"case_ref": case_ref, "author": "ops"}, telegram_text=telegram_text
    )


async def notify_dispute_resolved(account_id: str, message: str, case_ref: str | None = None) -> bool:
    """Called from ops/api.py right after staff resolve a dispute -- the
    borrower's account stopped being frozen and their banner disappeared, but
    nothing ever told them the outcome. Same real-or-logged delivery as every
    other notification here; the dashboard's "Requests & messages" shows it
    either way. Returns True if the borrower was actually reached over
    Telegram. case_ref puts it in that case's thread."""
    return await _deliver_and_log(
        account_id, message, "dispute_resolution_notified", {"case_ref": case_ref} if case_ref else {}
    )


async def notify_charge_waived(account_id: str, message: str, case_ref: str | None = None) -> bool:
    """Called from ops/api.py right after staff waive a charge (a late fee) -- the amount due dropped, and the
    borrower should know why. Same real-or-logged delivery as every notification here; shown on the dashboard either
    way (in the case's thread when case_ref is given). Returns True if reached over Telegram."""
    return await _deliver_and_log(account_id, message, "charge_waived_notified", {"case_ref": case_ref} if case_ref else {})


async def notify_clarification_request(account_id: str, message: str) -> bool:
    """Called from ops/api.py right after an operator sends a
    clarification request about an account's flags -- message is the
    operator's own final wording (possibly LLM-polished via
    outbound/compose.py, always human-reviewed before this is called).
    Returns True if the borrower was actually reached over Telegram."""
    return await _deliver_and_log(account_id, message, "clarification_request_sent", {})


def send_ops_alert(text: str) -> bool:
    """For an internal signal meant for a human operator, not a borrower
    (e.g. scripts/run_eval_monitor.py's nightly regression check) --
    delivered to a fixed admin chat via OPS_ALERT_TELEGRAM_CHAT_ID, the
    same Bot/token every borrower-facing message here already uses.
    Returns False (never raises) if that chat id isn't configured, or if
    Telegram itself rejects delivery -- callers must log loudly themselves
    as the real fallback, since there's no account to log an event
    against here the way _deliver_and_log's callers can."""
    chat_id = os.environ.get("OPS_ALERT_TELEGRAM_CHAT_ID")
    if not chat_id:
        return False
    return asyncio.run(_send_telegram_message(int(chat_id), text))
