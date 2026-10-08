"""One definition of "what does this borrower owe right now".

Before this module the same question had three different answers: the
borrower's overdue banner minted a link for EMI + late fee, outbound
reminders minted one for the bare EMI (ignoring any credit), and the pay page
compared whichever it got against EMI - credit -- so a perfectly ordinary
overdue payment landed on a "this is more than what's due, how should we apply
the extra?" question. Everything that quotes, mints or checks an amount now
goes through here.

Pure functions over an Account: no I/O, no clock of their own (callers pass
as_of), so every surface agrees and they are cheap to test.
"""

from datetime import date

from businessflow.accounts.models import Account
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, LATE_FEE_FLAT_AMOUNT


def late_fee_due(account: Account, as_of: date) -> float:
    """The flat late fee (accounts/policy.py), once the EMI is past the grace
    period; 0.0 before that, and 0.0 for a fully repaid loan (days_past_due
    is 0 there). The fee is informational: nothing in the ledger records it
    as assessed, paid or waived."""
    return float(LATE_FEE_FLAT_AMOUNT) if account.days_past_due(as_of) > GRACE_PERIOD_DAYS else 0.0


def emi_due_now(account: Account) -> float:
    """What this cycle's EMI still needs: EMI minus any credit already
    banked from an earlier extra/partial payment, never negative. 0.0 for a
    fully repaid loan."""
    if account.months_remaining <= 0:
        return 0.0
    return max(0.0, round(account.emi_amount - account.pending_emi_credit, 2))


def amount_due_now(account: Account, as_of: date) -> float:
    """EMI still due this cycle plus the late fee if it applies -- the one
    number the dashboard, reminders, Telegram and the pay page should agree
    on."""
    return round(emi_due_now(account) + late_fee_due(account, as_of), 2)


def reminder_amounts(account: Account, kind: str) -> tuple[float, float, float]:
    """(emi, late_fee, total) a proactive reminder of this kind should quote
    and mint a payment link for. Keyed off the reminder's own kind rather
    than a clock: outbound/decide.py only ever issues a "follow_up" once the
    EMI is past the grace period, i.e. exactly when the late fee applies --
    so this can't disagree with the decision that produced the reminder."""
    emi = emi_due_now(account)
    fee = float(LATE_FEE_FLAT_AMOUNT) if kind == "follow_up" else 0.0
    return emi, fee, round(emi + fee, 2)
