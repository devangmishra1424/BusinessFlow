"""Tappable actions under a proactive Telegram reminder.

A reminder used to be one-way: text plus a "Pay now" link. The borrower who cannot pay today but
means to had to open the chat and type a sentence, and that sentence went through a full LLM turn
(40-55 s measured, see eval/results/latency_benchmark.json). These buttons turn the commonest
replies into one tap that calls the same tools the LLM would, with no model in the loop:

    "I'll pay tomorrow / in 3 days / in a week"  -> tools.account_tools.log_promise_to_pay
    "Talk to a person"                           -> tools.escalation_tools.escalate_to_human
    "I dispute this"                             -> tells them to send /dispute <what happened>
                                                    (a dispute needs the reason in the borrower's words)

Kept light on purpose: this module imports no torch, ASR, TTS or telegram-bot code, so outbound/send.py
(which builds the buttons) and channels/telegram_bot.py (which handles the tap) can share it, and the
logic is testable without a database or a Telegram connection (see ActionDeps).

Trust: a tap is acted on only when it comes from the chat linked to that account
(accounts.telegram_chat_id, written once a real access-key check passed). The account id travels in
the callback data, so a stale or forwarded button cannot act on a different account.

Callback data is "rem:<action>:<account_id>[:<days>]", well under Telegram's 64-byte limit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

logger = logging.getLogger(__name__)

PROMISE_DAYS = (1, 3, 7)
_PREFIX = "rem"
_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$")

# Runtime languages only: the account's "hinglish" preference runs in English here, same as the chat.
_TEXT = {
    "en": {
        "tomorrow": "📅 I'll pay tomorrow",
        "in3": "📅 In 3 days",
        "week": "📅 In a week",
        "human": "🧑 Talk to a person",
        "dispute": "⚠️ I dispute this",
        "ptp_done": "Noted -- I've recorded that you'll pay ₹{amount} by {date}. If anything changes, just message me here.",
        "ptp_again": "That is already on record: ₹{amount} by {date}.",
        "nothing_due": "Nothing is due on your account right now.",
        "human_done": "I've forwarded your request to a human agent (reference {ref}). They'll get back to you soon.",
        "dispute_help": (
            "Tell me what happened with /dispute <what happened> -- e.g. /dispute I already paid this via UPI on the 3rd. "
            "Once it is flagged, no further automated collection action is taken until a human reviews it."
        ),
        "not_yours": "This button isn't linked to your account. Send your account ID and access key together to verify.",
        "stale": "That option isn't available any more.",
    },
    "hi": {
        "tomorrow": "📅 कल भुगतान होगा",
        "in3": "📅 3 दिन में",
        "week": "📅 एक हफ़्ते में",
        "human": "🧑 किसी इंसान से बात करें",
        "dispute": "⚠️ मुझे आपत्ति है",
        "ptp_done": "ठीक है -- ₹{amount} का भुगतान {date} तक करने की बात दर्ज कर ली गई है। कुछ बदले तो यहीं मैसेज करें।",
        "ptp_again": "यह पहले से दर्ज है: ₹{amount}, {date} तक।",
        "nothing_due": "अभी आपके खाते में कोई बकाया नहीं है।",
        "human_done": "आपका अनुरोध एक एजेंट को भेज दिया गया है (संदर्भ {ref})। वे जल्द ही संपर्क करेंगे।",
        "dispute_help": (
            "/dispute <क्या हुआ> भेजें -- जैसे /dispute मैंने यह रकम 3 तारीख को UPI से दे दी थी। "
            "फ़्लैग होने के बाद, किसी इंसान के देखने तक कोई स्वचालित वसूली कार्रवाई नहीं होगी।"
        ),
        "not_yours": "यह बटन आपके खाते से जुड़ा नहीं है। पुष्टि के लिए अपना अकाउंट ID और एक्सेस की एक साथ भेजें।",
        "stale": "यह विकल्प अब उपलब्ध नहीं है।",
    },
}
_PROMISE_LABEL = {1: "tomorrow", 3: "in3", 7: "week"}


@dataclass(frozen=True)
class ActionRequest:
    action: str  # "ptp" | "human" | "dispute"
    account_id: str
    days: int | None = None  # only for "ptp"


@dataclass(frozen=True)
class ActionResult:
    text: str
    # True once the action was carried out, so the buttons can come off the reminder (a second tap is
    # not needed and, for "human", would open a second escalation).
    done: bool


def runtime_language(language_preference: str | None) -> str:
    return "hi" if language_preference == "hi" else "en"


def encode(action: str, account_id: str, days: int | None = None) -> str:
    data = f"{_PREFIX}:{action}:{account_id}" + (f":{days}" if days is not None else "")
    if len(data.encode()) > 64:
        raise ValueError(f"callback data is over Telegram's 64-byte limit: {data!r}")
    return data


def decode(data: str) -> ActionRequest | None:
    """None for anything that is not a well-formed reminder action. Strict, because the data comes back from the client."""
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != _PREFIX or not _ACCOUNT_ID.match(parts[2]):
        return None
    action, account_id = parts[1], parts[2]
    if action in ("human", "dispute") and len(parts) == 3:
        return ActionRequest(action, account_id)
    if action == "ptp" and len(parts) == 4 and parts[3].isdigit() and int(parts[3]) in PROMISE_DAYS:
        return ActionRequest(action, account_id, int(parts[3]))
    return None


def action_rows(account_id: str, language_preference: str | None) -> list[list[tuple[str, str]]]:
    """Rows of (button label, callback data) to put under a reminder."""
    t = _TEXT[runtime_language(language_preference)]
    return [
        [(t[_PROMISE_LABEL[d]], encode("ptp", account_id, d)) for d in PROMISE_DAYS],
        [(t["human"], encode("human", account_id)), (t["dispute"], encode("dispute", account_id))],
    ]


def promise_date(today: date, days: int) -> date:
    return today + timedelta(days=days)


@dataclass
class ActionDeps:
    """What run_action touches, as plain callables, so the logic can be exercised with simple stand-ins and no database."""

    get_account: Callable[[str], object | None]
    today: Callable[[], date]
    amount_due: Callable[[object, date], float]
    log_promise: Callable[[str, str, float], dict]
    escalate: Callable[[str, str], dict]
    log_event: Callable[[str, str, dict], None]


def real_deps() -> ActionDeps:
    from businessflow.accounts import store
    from businessflow.accounts.dues import amount_due_now
    from businessflow.tools.account_tools import log_promise_to_pay
    from businessflow.tools.escalation_tools import escalate_to_human

    return ActionDeps(
        get_account=store.get_account,
        today=store.current_date,
        amount_due=amount_due_now,
        log_promise=log_promise_to_pay,
        escalate=escalate_to_human,
        log_event=store.log_event,
    )


def run_action(chat_id: int, data: str, deps: ActionDeps | None = None) -> ActionResult | None:
    """Carries out a tapped reminder button. None when `data` is not a reminder action at all (the caller handles other buttons)."""
    request = decode(data)
    if request is None:
        return None
    d = deps or real_deps()

    account = d.get_account(request.account_id)
    if account is None or account.telegram_chat_id != chat_id:
        # A forwarded or stale button, or a chat that was re-linked to another account. Say nothing about whose account it is.
        logger.warning("reminder action %s refused: chat_id=%s is not the chat linked to %s", request.action, chat_id, request.account_id)
        return ActionResult(_TEXT["en"]["not_yours"], done=False)

    t = _TEXT[runtime_language(account.language_preference)]
    today = d.today()

    if request.action == "dispute":
        return ActionResult(t["dispute_help"], done=False)

    if request.action == "human":
        reason = "Borrower tapped 'talk to a person' on a reminder"
        result = d.escalate(account.account_id, reason)
        d.log_event(account.account_id, "tool_called", {"tool": "escalate_to_human", "arguments": {"account_id": account.account_id, "reason": reason}, "result": result})
        return ActionResult(t["human_done"].format(ref=result["escalation_id"]), done=True)

    # "ptp": the amount is what is due now (EMI less credit, plus the late fee once it applies): the figure the reminder quoted.
    amount = d.amount_due(account, today)
    if amount <= 0:
        return ActionResult(t["nothing_due"], done=True)
    promised = promise_date(today, request.days)
    result = d.log_promise(account.account_id, promised.isoformat(), amount)
    d.log_event(account.account_id, "tool_called", {"tool": "log_promise_to_pay", "arguments": {"account_id": account.account_id, "promised_date": promised.isoformat(), "promised_amount": amount}, "result": result})
    key = "ptp_again" if result.get("already_logged") is True else "ptp_done"
    return ActionResult(t[key].format(amount=f"{amount:,.2f}", date=f"{promised:%d %b}"), done=True)
