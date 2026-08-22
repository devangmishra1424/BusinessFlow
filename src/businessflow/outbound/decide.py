"""Stage 1 of proactive outbound: pure, countable logic -- which
accounts get a heads-up before their EMI is due, and which get a
follow-up because they're already past the grace period, as of today.
No AI, no scoring -- a plain date comparison per account, matching the
blueprint's own "no AI involved" framing for deadline-proximity
flagging (§13's flag system), applied here to outbound timing instead.

An open dispute suppresses automated outbound too, same policy as
automated restructuring (accounts/policy.py's
DISPUTE_BLOCKS_AUTOMATED_RESTRUCTURING) -- reaching out with a routine
payment reminder while a dispute is open and awaiting human review
would contradict the account's own frozen state.
"""

from dataclasses import dataclass
from datetime import date

from businessflow.accounts import store
from businessflow.accounts.models import Account
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, HEADS_UP_DAYS_BEFORE_DUE


@dataclass
class OutboundReminder:
    account_id: str
    kind: str  # "heads_up" | "follow_up"
    days: int  # days until due (heads_up) or days past due (follow_up)


def decide_reminder(account: Account, as_of: date | None = None) -> OutboundReminder | None:
    if as_of is None:
        as_of = store.current_date()

    if account.dispute_open:
        return None

    days_until_due = (account.emi_due_date - as_of).days
    if 0 < days_until_due <= HEADS_UP_DAYS_BEFORE_DUE:
        return OutboundReminder(account.account_id, "heads_up", days_until_due)

    days_past_due = account.days_past_due(as_of)
    if days_past_due > GRACE_PERIOD_DAYS:
        return OutboundReminder(account.account_id, "follow_up", days_past_due)

    return None


def decide_reminders(account_ids: list[str] | None = None, as_of: date | None = None) -> list[OutboundReminder]:
    accounts = store.list_accounts() if account_ids is None else [store.get_account_or_raise(a) for a in account_ids]
    return [r for r in (decide_reminder(a, as_of) for a in accounts) if r is not None]
