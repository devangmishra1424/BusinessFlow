"""Account flagging for the business/ops view -- deliberately countable,
explainable triggers over real fields (days past due, dispute status,
broken-promise count), never an opaque ML risk score. A black-box
judgment of a real person's payment behavior is a fairness risk; a rule
you can point to and explain isn't.

Every flag carries its own reason string, not just a label -- "why is
this account flagged" has to be answerable in plain language, per the
blueprint's own explicit design (never just a score).
"""

from dataclasses import dataclass

from businessflow.accounts import store
from businessflow.accounts.models import Account
from businessflow.accounts.policy import BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION, GRACE_PERIOD_DAYS


@dataclass
class Flag:
    label: str  # 'overdue' | 'disputed' | 'broken_promises'
    reason: str


def compute_flags(account: Account, as_of=None) -> list[Flag]:
    """as_of defaults to the system's demo-date anchor (store.current_date())
    -- the same anchor every days-past-due calculation elsewhere in this
    project already uses, so a flag here can never disagree with what a
    tool call would say about the same account."""
    if as_of is None:
        as_of = store.current_date()

    flags = []

    days_past_due = account.days_past_due(as_of)
    if days_past_due > GRACE_PERIOD_DAYS:
        flags.append(Flag(
            "overdue",
            f"{days_past_due} days past due (beyond the {GRACE_PERIOD_DAYS}-day grace period)",
        ))

    if account.dispute_open:
        real_reason = store.get_latest_open_dispute_reason(account.account_id)
        flags.append(Flag("disputed", real_reason or "has an open, unresolved dispute"))

    broken = account.broken_promise_count()
    if broken >= BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION:
        flags.append(Flag(
            "broken_promises",
            f"{broken} broken promises to pay (at or above the {BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION}-promise threshold)",
        ))

    return flags


def is_clean(account: Account, as_of=None) -> bool:
    return not compute_flags(account, as_of)
