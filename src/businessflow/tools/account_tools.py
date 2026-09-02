"""Tools that read or record facts against a borrower's account."""

from datetime import date

from businessflow.accounts import store
from businessflow.accounts.policy import GRACE_PERIOD_DAYS, LATE_FEE_FLAT_AMOUNT, PROMISE_TOLERANCE_DAYS
from businessflow.tools.server import mcp

# payment_history exists on the real Account model and Postgres table (see
# accounts/models.py, accounts/store.py's _load_payment_history), but until
# now it was only ever exposed through the ops-only, staff-gated HTTP API --
# no borrower-facing tool let this agent answer "can you tell me my recent
# payment history" at all. Confirmed via a direct audit of this file against
# that table, the same class of gap _GROUND_NACH_FAILURES (agent/client.py)
# closed for the NACH mandate field. Capped, not unbounded: a borrower
# asking for "all of it" gets a real, bounded maximum instead of an
# open-ended query.
_MAX_PAYMENT_HISTORY_LIMIT = 20


@mcp.tool
def get_payment_status(account_id: str) -> dict:
    """Look up a borrower's current payment status: original loan amount,
    EMI due date, days past due, months of EMIs remaining, an approximate
    outstanding balance, whether the account's NACH auto-debit mandate is
    currently active, whether a late fee applies, and whether a dispute is
    open on the account.

    Found live: a borrower's single message often bundles several of
    these ("how much is my loan, how many months are left, what's my
    EMI") -- this tool answers all of them from account_id alone, so one
    call covers a compound question like that instead of leaving some
    parts unanswered. outstanding_balance_approx uses the same
    simplification calculate_hypothetical already relies on
    (emi_amount * months_remaining) -- not a real amortization schedule,
    and labeled as approximate for that reason, not a hidden precision
    claim. interest_rate_pct is None for most accounts right now -- it's
    only populated once a borrower's signed loan agreement has been
    uploaded and successfully parsed (see rag/extraction.py's
    extract_loan_terms); if a borrower asks for it and this is None, say
    so rather than inferring or guessing one from the EMI and principal.
    nach_mandate_active reflects only whether the mandate is CURRENTLY
    active -- it cannot say why a specific debit attempt bounced; ground
    a "why did my auto-debit fail" question in this field plus
    check_policy's nach_mandate_troubleshooting.md, not a guess.
    late_fee_applicable is true (with late_fee_amount set) only once
    days_past_due exceeds the grace period; within the grace period, or
    not past due at all, it's false and late_fee_amount is None."""
    account = store.get_account_or_raise(account_id)
    as_of = store.current_date()
    days_past_due = account.days_past_due(as_of)
    late_fee_applicable = days_past_due > GRACE_PERIOD_DAYS
    return {
        "account_id": account.account_id,
        "borrower_name": account.borrower_name,
        "business_name": account.business_name,
        "principal_amount": account.principal_amount,
        "emi_amount": account.emi_amount,
        "emi_due_date": account.emi_due_date.isoformat(),
        "days_past_due": days_past_due,
        "tenure_months": account.tenure_months,
        "months_remaining": account.months_remaining,
        "outstanding_balance_approx": round(account.emi_amount * account.months_remaining - account.pending_emi_credit, 2),
        "interest_rate_pct": account.interest_rate_pct,
        "nach_mandate_active": account.nach_mandate_active,
        "late_fee_applicable": late_fee_applicable,
        "late_fee_amount": float(LATE_FEE_FLAT_AMOUNT) if late_fee_applicable else None,
        "dispute_open": account.dispute_open,
        "risk_tier": account.risk_tier,
        "broken_promise_count": account.broken_promise_count(),
        # A running credit from an earlier off-cycle extra payment the
        # borrower chose to apply, or an overpayment's excess -- reduces
        # what's actually due next cycle. Zero for the overwhelming
        # majority of accounts, which never make an off-cycle payment.
        "pending_emi_credit": account.pending_emi_credit,
    }


@mcp.tool
def get_payment_history(account_id: str, limit: int = 5) -> dict:
    """Look up a borrower's most recent payment history: date, amount,
    whether it was on time, and what kind of payment it was, most recent
    first. kind is "regular" for an ordinary EMI-cycle payment,
    "extra_unapplied"/"extra_applied" for an off-cycle payment the
    borrower chose not to/chose to credit toward their next EMI, or
    "overpayment_applied" for a payment larger than what was due that
    cycle (the excess auto-credited toward the next one) -- see
    accounts.store.record_payment for the full decision table.

    limit is clamped to a real, bounded maximum
    (_MAX_PAYMENT_HISTORY_LIMIT) rather than raising -- a borrower asking
    for "all of it" should just get a reasonable cap, not an error. A
    non-positive limit is clamped up to 1 for the same reason: this
    should never error on a caller's odd input, only bound it."""
    account = store.get_account_or_raise(account_id)
    effective_limit = max(1, min(limit, _MAX_PAYMENT_HISTORY_LIMIT))
    records = sorted(account.payment_history, key=lambda r: r.date, reverse=True)[:effective_limit]
    return {
        "account_id": account.account_id,
        "payment_history": [
            {"date": r.date.isoformat(), "amount": r.amount, "on_time": r.on_time, "kind": r.kind} for r in records
        ],
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
