"""Tools that read or record facts against a borrower's account."""

from datetime import date

from businessflow.accounts import store
from businessflow.accounts.policy import PROMISE_TOLERANCE_DAYS
from businessflow.tools.server import mcp


@mcp.tool
def get_payment_status(account_id: str) -> dict:
    """Look up a borrower's current payment status: original loan amount,
    EMI due date, days past due, months of EMIs remaining, an approximate
    outstanding balance, and whether a dispute is open on the account.

    Found live: a borrower's single message often bundles several of
    these ("how much is my loan, how many months are left, what's my
    EMI") -- this tool answers all of them from account_id alone, so one
    call covers a compound question like that instead of leaving some
    parts unanswered. outstanding_balance_approx uses the same
    simplification calculate_hypothetical already relies on
    (emi_amount * months_remaining) -- not a real amortization schedule,
    and labeled as approximate for that reason, not a hidden precision
    claim. Does NOT include an interest rate or APR -- this system does
    not track one as a separate field; if a borrower asks for it
    specifically, say so rather than inferring or guessing one from the
    EMI and principal."""
    account = store.get_account_or_raise(account_id)
    as_of = store.current_date()
    return {
        "account_id": account.account_id,
        "borrower_name": account.borrower_name,
        "business_name": account.business_name,
        "principal_amount": account.principal_amount,
        "emi_amount": account.emi_amount,
        "emi_due_date": account.emi_due_date.isoformat(),
        "days_past_due": account.days_past_due(as_of),
        "tenure_months": account.tenure_months,
        "months_remaining": account.months_remaining,
        "outstanding_balance_approx": round(account.emi_amount * account.months_remaining, 2),
        "dispute_open": account.dispute_open,
        "risk_tier": account.risk_tier,
        "broken_promise_count": account.broken_promise_count(),
    }


@mcp.tool
def log_promise_to_pay(account_id: str, promised_date: str, promised_amount: float) -> dict:
    """Record a borrower's promise to pay a specific amount by a specific
    date. promised_date is an ISO date string (YYYY-MM-DD). Whether the
    promise was kept is evaluated later, once that date has passed, against
    a tolerance band of a few days either side -- not the exact date."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    parsed_date = date.fromisoformat(promised_date)
    newly_logged = store.add_promise(
        account_id, made_on=store.current_date(), promised_date=parsed_date, promised_amount=promised_amount
    )
    return {
        "account_id": account_id,
        "promised_date": parsed_date.isoformat(),
        "promised_amount": promised_amount,
        "tolerance_days": PROMISE_TOLERANCE_DAYS,
        "logged": True,
        "already_logged": not newly_logged,  # True if this exact promise was already on record today
    }


@mcp.tool
def flag_dispute(account_id: str, reason: str) -> dict:
    """Open a dispute flag on the account so no further automated collection
    action is taken on it until a human reviews the reason given."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    newly_opened = store.open_dispute(account_id, reason)
    return {
        "account_id": account_id,
        "dispute_open": True,
        "reason": reason,
        "already_open": not newly_opened,  # True if the account already had a dispute open
    }
