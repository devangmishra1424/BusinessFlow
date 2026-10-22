"""Tools that read or record facts against a borrower's account."""

from datetime import date
from typing import Literal, TypedDict

from businessflow.accounts import cases, store
from businessflow.accounts.dues import amount_due_now, charges_due
from businessflow.accounts.promises import Part, promise_view_for, validate_parts
from businessflow.accounts.promises import terms_text as promise_terms_text
from businessflow.accounts.policy import (
    PAYMENT_CLAIM_REVIEW_WORKING_DAYS,
)
from businessflow.tools import validation
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


@mcp.tool(
    description=(
        "Look up a borrower's current payment status: loan amount, EMI due date, days "
        "past due, months remaining, approximate outstanding balance, NACH mandate "
        "status, late fee applicability, and dispute status. outstanding_balance_approx "
        "is an approximation (emi_amount * months_remaining), not a real amortization "
        "schedule. amount_due_now is what the borrower owes right now (this cycle's EMI "
        "less any credit, plus every late fee still owed) -- quote it directly instead of adding figures yourself. "
        "A late fee is a real charge on the account: charged once for each EMI still unpaid after the grace period, "
        "never compounded, never charged interest. charges lists each one still owed (amount, which EMI it is for); "
        "a payment pays this month's EMI first, then the fee. Only our team can waive a fee -- never say one is or "
        "will be waived. "
        "interest_rate_pct is often None -- if so, say you don't have it "
        "rather than guessing one. nach_mandate_active only reflects current status, not "
        "why a debit failed -- for 'why did my auto-debit fail', combine this with "
        "check_policy on NACH troubleshooting, don't guess. late_fee_amount (the late fees still owed) is set only "
        "when late_fee_applicable is true."
    )
)
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
    as_of = store.current_date()
    # The fee is on the books the first time anyone looks, not only after the nightly pass (store.assess_late_fees).
    store.assess_late_fees(as_of, account_id)
    store.resolve_promises(as_of, account_id)
    account = store.get_account_or_raise(account_id)
    days_past_due = account.days_past_due(as_of)
    fees_owed = charges_due(account)
    late_fee_applicable = fees_owed > 0
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
        "late_fee_amount": fees_owed if late_fee_applicable else None,
        "charges": [
            {"kind": c.kind, "for_emi_due_date": c.for_due_date.isoformat(), "amount": c.amount,
             "amount_still_owed": c.outstanding}
            for c in account.open_charges()
        ],
        # Late fees a person on the team waived in the last 60 days, with their reason -- so "was my fee removed?" has
        # a grounded answer, and saying it was is not blocked as an invented waiver (guardrail/grounding.py).
        "recently_waived": [
            {"for_emi_due_date": c.for_due_date.isoformat(), "amount_waived": round(c.amount - c.amount_paid, 2),
             "waived_on": c.waived_on.isoformat(), "reason": c.waived_reason}
            for c in account.charges
            if c.status == "waived" and c.waived_on and (as_of - c.waived_on).days <= 60
        ],
        # What the borrower owes right now -- one grounded figure for the
        # agent to quote (so it never has to add EMI + fee itself, which the
        # grounding guardrail would block as unverified arithmetic) and the
        # same number the dashboard's Pay button and reminders use.
        "amount_due_now": amount_due_now(account, as_of),
        "dispute_open": account.dispute_open,
        "risk_tier": account.risk_tier,
        "broken_promise_count": account.broken_promise_count(),
        # The borrower's current promise (or most recent one), judged by its own words: terms, status
        # ('on_track' | 'kept' | 'broken' | 'replaced'), each part with how much is paid, and the next part still owed.
        "promise": promise_view_for(account, as_of),
        # A running credit from an earlier off-cycle extra payment the
        # borrower chose to apply, or an overpayment's excess -- reduces
        # what's actually due next cycle. Zero for the overwhelming
        # majority of accounts, which never make an off-cycle payment.
        "pending_emi_credit": account.pending_emi_credit,
    }


@mcp.tool(
    description=(
        "Look up a borrower's recent payments: date, amount, on-time flag, and kind, "
        "most recent first. kind is 'regular' for a normal EMI payment, "
        "'extra_unapplied'/'extra_applied' for an off-cycle payment not credited/credited "
        "toward the next EMI, 'overpayment_applied' when a payment exceeded what was "
        "due that cycle (excess credited to the next one), or 'charge' when it only paid a late fee; "
        "charges_paid is how much of a payment went to late fees. limit is clamped to a safe "
        "range rather than erroring on an odd value."
    )
)
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
            {"date": r.date.isoformat(), "amount": r.amount, "on_time": r.on_time, "kind": r.kind, "charges_paid": r.charges_paid} for r in records
        ],
    }


class PromisePartIn(TypedDict):
    date: str  # YYYY-MM-DD
    amount: float


@mcp.tool(
    description=(
        "Record a borrower's promise to pay, in their exact words: promised_date (YYYY-MM-DD) and promised_amount for "
        "the first (or only) payment, and more_parts for any later ones ([{date, amount}], up to 3 more, later dates). "
        "A promise is judged by those words alone: each part must be paid in full by its date, or the whole promise is "
        "broken -- so pin the terms down first (how much, by which date, in one go or in parts). Two steps, always: "
        "call it with borrower_confirmed=false (the default) and it returns read_back, the terms written out; read "
        "them back to the borrower in their language and ask them to confirm. Only after a clear yes, call it again "
        "with exactly the same terms and borrower_confirmed=true to record it. A new promise replaces one still open."
    )
)
def log_promise_to_pay(
    account_id: str, promised_date: str, promised_amount: float,
    more_parts: list[PromisePartIn] | None = None, borrower_confirmed: bool = False,
) -> dict:
    """Records a promise plan from chat (accounts/promises.py), only after its exact terms were read back to the
    borrower: the first call returns the read-back text (and notes that it was read back); the confirming call, with
    the same terms and borrower_confirmed=True, records it. Enforced here, not only in the prompt: a promise the
    borrower never heard read back cannot be recorded from chat. Raises ValueError on terms that are not a real plan
    (past or far-off dates, dates out of order, more than the account could owe)."""
    account = store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    today = store.current_date()
    raw = [{"date": promised_date, "amount": promised_amount}, *(more_parts or [])]
    parts = []
    for i, r in enumerate(raw, 1):
        d = validation.promise_date(r.get("date") if isinstance(r, dict) else None, today)
        amount = validation.money(r.get("amount") if isinstance(r, dict) else None, f"part {i} amount", ceiling=validation.max_payable(account))
        parts.append(Part(d, amount))
    parts = validate_parts(parts, today, ceiling=validation.max_payable(account))
    terms = promise_terms_text(parts)

    if not borrower_confirmed or not store.promise_was_read_back(account_id, terms):
        store.log_promise_read_back(account_id, terms)
        return {
            "account_id": account_id,
            "logged": False,
            "read_back": terms,
            "next_step": (
                "Read these terms back to the borrower, in their language, and ask them to confirm. Only if they "
                "clearly say yes, call log_promise_to_pay again with exactly these terms and borrower_confirmed=true."
            ),
        }
    result = store.add_promise_plan(account_id, parts, made_on=today, made_via="chat", terms_text=terms)
    return {
        "account_id": account_id,
        "logged": True,
        "terms": terms,
        "plan_id": result["plan_id"],
        "already_logged": result["already_logged"],  # True if this exact promise was already on record today
        "replaced_earlier_promise": result["replaced_plan_id"] is not None,
        "how_it_is_judged": "Each part must be paid in full by its date; if one is missed, the promise is broken.",
    }


@mcp.tool(
    description=(
        "Record something a borrower raised about their account for a person on our team to check; "
        "automated reminders pause until it is closed. kind='payment_claim' when they say they ALREADY "
        f"PAID (an EMI, a late fee, any amount) and it isn't showing: the team checks the bank record within "
        f"{PAYMENT_CLAIM_REVIEW_WORKING_DAYS} working days and records the payment. kind='dispute' when they "
        "say an amount, fee or due date is wrong. reason is the borrower's own words, with any amount, date, "
        "payment method or reference they gave. Only one can be open at a time: if one already is, this adds "
        "their new words to it, so call it again when they send more details (a UTR, a date)."
    )
)
def flag_dispute(account_id: str, reason: str, kind: Literal["dispute", "payment_claim"] = "dispute") -> dict:
    """Opens a dispute or a payment claim (accounts/cases.py), or adds the
    borrower's words to the one already open. Found live: "I have paid my
    EMI, can you update it in the profile" was filed as a dispute and
    answered "A human agent will review it" -- nobody could tell a payment to
    look for from a disagreement, and the borrower wasn't told when."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    reason = validation.free_text(reason, "reason")
    if kind not in cases.DISPUTE_KINDS:
        raise ValueError(f"kind must be one of {cases.DISPUTE_KINDS}, got {kind!r}")
    case = store.open_dispute(account_id, reason, kind)
    case_ref = cases.dispute_case_ref(case["dispute_id"])
    if not case["opened"] and reason != case.get("reason"):
        # More from the borrower about the open case (a reference, a date): kept in its thread, where ops reads it.
        # The same words again (a retry) are not added twice.
        store.add_case_message(account_id, case_ref, "borrower", reason)
    result = {
        "account_id": account_id,
        "dispute_open": True,
        "reason": reason,
        "already_open": not case["opened"],  # True if one was already open; reason was added to it
        "kind": case["kind"],  # of the case now open -- the existing one's when already_open
        "case_ref": case_ref,
    }
    if case["kind"] == cases.PAYMENT_CLAIM:
        result["review_within_working_days"] = PAYMENT_CLAIM_REVIEW_WORKING_DAYS
    return result
