"""One definition of "what does this borrower owe right now", and of how a payment is applied.

Before this module the same question had three different answers: the
borrower's overdue banner minted a link for EMI + late fee, outbound
reminders minted one for the bare EMI (ignoring any credit), and the pay page
compared whichever it got against EMI - credit -- so a perfectly ordinary
overdue payment landed on a "this is more than what's due, how should we apply
the extra?" question. Everything that quotes, mints or checks an amount now
goes through here.

The late fee is a real charge (schema.sql's charges table, accounts/store.py's
assess_late_fees): it is assessed once per missed instalment when the grace
period has passed, and what is owed is the EMI still due plus every charge
still open. Before, the fee was a figure each screen worked out from the date,
so nothing could say it had been paid or waived.

Pure functions over an Account: no I/O, no clock of their own (callers pass
as_of / payment_date), so every surface agrees and they are cheap to test.
Callers that need charges to be current call store.assess_late_fees first.
"""

from dataclasses import dataclass
from datetime import date

from businessflow.accounts.models import Account
from businessflow.accounts.policy import GRACE_PERIOD_DAYS

_CENT = 0.005


def late_fee_assessable(account: Account, as_of: date) -> bool:
    """Whether the current instalment has passed the grace period unpaid as of as_of, i.e. a late fee for it is due
    to be assessed (store.assess_late_fees inserts it, once). Never for a fully repaid loan (days_past_due is 0)."""
    return account.days_past_due(as_of) > GRACE_PERIOD_DAYS


def charges_due(account: Account) -> float:
    """Every charge still owed on the account (a late fee not yet paid or waived)."""
    return round(sum(c.outstanding for c in account.open_charges()), 2)


def emi_due_now(account: Account) -> float:
    """What this cycle's EMI still needs: EMI minus any credit already
    banked from an earlier extra/partial payment, never negative. 0.0 for a
    fully repaid loan."""
    if account.months_remaining <= 0:
        return 0.0
    return max(0.0, round(account.emi_amount - account.pending_emi_credit, 2))


def amount_due_now(account: Account, as_of: date | None = None) -> float:
    """EMI still due this cycle plus every open charge -- the one number the dashboard, reminders, Telegram and the
    pay page should agree on. as_of is kept for callers' readability only: charges are what was assessed, so a caller
    that wants today's fee included runs store.assess_late_fees(as_of) first."""
    return round(emi_due_now(account) + charges_due(account), 2)


def reminder_amounts(account: Account, kind: str | None = None) -> tuple[float, float, float]:
    """(emi, charges, total) a proactive reminder should quote and mint a payment link for: the EMI still due and the
    charges actually on the account (outbound/run.py assesses late fees before it decides reminders, so a follow-up
    past the grace period carries its fee). kind is accepted for callers that pass it; the amounts no longer depend on
    it -- they come from the books, not from the reminder's type."""
    emi = emi_due_now(account)
    fees = charges_due(account)
    return emi, fees, round(emi + fees, 2)


@dataclass(frozen=True)
class Allocation:
    """How one payment is applied, decided before anything is written (the pay page shows it, record_payment applies it).

    to_emi: the part that goes to the current instalment. emi_retired: True when to_emi covers what that instalment
    still needs, so the cycle closes (a month off, the due date moves on). When it does not, short_of_emi says by how
    much, and the payment is a part-payment: the borrower chooses whether it counts toward this EMI (the caller's
    apply_extra_to_next).
    to_charges: (charge_id, amount) pairs, oldest charge first; a charge can be paid in part.
    extra: what is left after the EMI and the charges -- a prepayment, applied by the caller's payment_scheme.
    """
    to_emi: float
    emi_retired: bool
    short_of_emi: float
    to_charges: tuple[tuple[int, float], ...]
    extra: float

    @property
    def charges_total(self) -> float:
        return round(sum(a for _, a in self.to_charges), 2)


def allocate_payment(account: Account, amount: float, payment_date: date) -> Allocation:
    """The order a payment is applied in. Written to keep the borrower current first:

    - The current instalment is due (its due date is on or before the payment date): EMI first, then open charges
      (oldest first), then anything left is extra. Paying the EMI keeps the loan current; a late fee left unpaid stays
      visibly owed but never makes the instalment late. (RBI's 2023 penal-charges rules ask that such charges are
      kept apart from the loan and never compound; applying the EMI first is the borrower-friendly reading.)
    - The current instalment is not due yet: open charges first (they are already owed), then the EMI (paying early
      is ordinary), then extra.

    A payment that does not reach the EMI after that order is a part-payment toward it (short_of_emi > 0); charges
    are never paid by a part-payment of an overdue EMI -- the EMI comes first.
    """
    remaining = round(float(amount), 2)
    open_charges = account.open_charges()
    emi_owed = emi_due_now(account)
    instalment_due = emi_owed > 0 and account.emi_due_date <= payment_date
    to_charges: list[tuple[int, float]] = []

    def pay_charges() -> None:
        nonlocal remaining
        for c in open_charges:
            if remaining <= _CENT:
                break
            part = round(min(c.outstanding, remaining), 2)
            to_charges.append((c.id, part))
            remaining = round(remaining - part, 2)

    to_emi, retired, short = 0.0, False, 0.0
    if instalment_due:
        if remaining + 0.01 >= emi_owed:
            to_emi, retired = emi_owed, True
            remaining = round(remaining - emi_owed, 2)
            pay_charges()
        else:
            to_emi, short, remaining = remaining, round(emi_owed - remaining, 2), 0.0
    else:
        pay_charges()
        if emi_owed > 0 and remaining > _CENT:
            if remaining + 0.01 >= emi_owed:
                to_emi, retired = emi_owed, True
                remaining = round(remaining - emi_owed, 2)
            else:
                to_emi, short, remaining = remaining, round(emi_owed - remaining, 2), 0.0
    return Allocation(
        to_emi=round(to_emi, 2), emi_retired=retired, short_of_emi=short,
        to_charges=tuple(to_charges), extra=max(0.0, round(remaining, 2)),
    )
