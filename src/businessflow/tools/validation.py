"""Checks on what the model hands to the tools that change something.

Everything a tool receives from the model is untrusted input: it is text the model produced after reading a borrower's
message, a retrieved document, or an earlier session's recap, any of which may have been written to steer it. The system
prompt asks the model to behave; these checks are what still holds when it does not. A tool that mints a payment link,
records a promise or opens a dispute validates its arguments itself, so the limit is the same however the model arrived
at the call.

Failures raise ValueError. agent/loop.py returns the message to the model as the tool result, so it can tell the
borrower what is wrong instead of the conversation failing.
"""

import math
import re
from datetime import date, timedelta

from businessflow.accounts.policy import LATE_FEE_FLAT_AMOUNT

MAX_REASON_CHARS = 1000
MAX_PROMISE_DAYS_AHEAD = 90

# C0 controls (except tab/newline), DEL, and the invisible Unicode line/direction overrides that can make stored text
# render differently from what it says (a reason shown to ops, or quoted back to a model in a later session).
_CONTROL_CHARS = re.compile("[\x00-\x08\x0b-\x1f\x7f  ‪-‮⁦-⁩]")


def finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return float(value)


def max_payable(account) -> float:
    """The most this account could plausibly owe: every remaining EMI plus one late fee. A real settlement or prepayment
    is below this; an amount above it is a mistake or an attempt to record something that never happened."""
    return round(float(account.emi_amount) * max(account.months_remaining, 1) + LATE_FEE_FLAT_AMOUNT, 2)


def money(value, field: str, *, ceiling: float) -> float:
    """A positive, finite amount of rupees, rounded to paise and no more than ceiling."""
    amount = round(finite_number(value, field), 2)
    if amount <= 0:
        raise ValueError(f"{field} must be more than zero, got {amount:,.2f}")
    if amount > ceiling:
        raise ValueError(f"{field} {amount:,.2f} is more than the most this account could owe ({ceiling:,.2f})")
    return amount


def free_text(value, field: str) -> str:
    """Text a human will read later (a dispute or escalation reason): control characters removed, trimmed, and capped at
    MAX_REASON_CHARS. Empty is an error, because a reason nobody can read is no reason."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text, got {type(value).__name__}")
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    if len(cleaned) > MAX_REASON_CHARS:
        cleaned = cleaned[: MAX_REASON_CHARS - 1].rstrip() + "…"
    return cleaned


def promise_date(value, today: date) -> date:
    """An ISO date that is today or later and no more than MAX_PROMISE_DAYS_AHEAD days away."""
    if not isinstance(value, str):
        raise ValueError(f"promised_date must be a YYYY-MM-DD string, got {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"promised_date must be a YYYY-MM-DD date, got {value!r}") from e
    if parsed < today:
        raise ValueError(f"promised_date {parsed.isoformat()} is in the past (today is {today.isoformat()})")
    if parsed > today + timedelta(days=MAX_PROMISE_DAYS_AHEAD):
        raise ValueError(f"promised_date {parsed.isoformat()} is more than {MAX_PROMISE_DAYS_AHEAD} days away")
    return parsed
