"""Tools that touch money -- all synthetic. No real payment gateway exists
anywhere in this project; generate_payment_link returns a fake local URL, and
nothing here ever calls out to a real payment processor."""

from businessflow.accounts import store
from businessflow.accounts.policy import (
    DISPUTE_BLOCKS_AUTOMATED_RESTRUCTURING,
    BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION,
    MAX_RESTRUCTURING_EXTENSION_MONTHS,
    MIN_PARTIAL_PAYMENT_PCT,
    RESTRUCTURING_TYPES,
    SETTLEMENT_DISCOUNT_PCT,
)
from businessflow.tools.server import mcp


def _blocked_from_automated_restructuring(account) -> str | None:
    """Returns a reason string if the account must go to a human instead of
    an automated offer, or None if automated restructuring is fine."""
    if DISPUTE_BLOCKS_AUTOMATED_RESTRUCTURING and account.dispute_open:
        return "account has an open dispute -- needs a human, not an automated offer"
    if account.broken_promise_count() >= BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION:
        return (
            f"account has {account.broken_promise_count()} broken promises "
            f"(>= {BROKEN_PROMISES_BEFORE_MANDATORY_ESCALATION}) -- needs a human"
        )
    return None


@mcp.tool
def generate_payment_link(account_id: str, amount: float) -> dict:
    """Generate a synthetic payment link for the given amount. This is a
    demo stub only -- it does not move real money and is not connected to
    any real payment processor."""
    account = store.get_account_or_raise(account_id)
    return {
        "account_id": account.account_id,
        "amount": amount,
        "payment_link": f"https://demo.businessflow.local/pay/{account.account_id}?amount={amount}",
        "synthetic": True,
    }


@mcp.tool
def propose_partial_payment(account_id: str, proposed_amount: float) -> dict:
    """Check whether a reduced payment for this cycle is within policy, and
    if so, record it as accepted. Rejects anything below the policy minimum,
    and refuses to make an automated offer at all on accounts that need a
    human (open dispute or a pattern of broken promises)."""
    account = store.get_account_or_raise(account_id)

    block_reason = _blocked_from_automated_restructuring(account)
    if block_reason:
        return {"account_id": account.account_id, "eligible": False, "reason": block_reason}

    minimum = round(account.emi_amount * MIN_PARTIAL_PAYMENT_PCT, 2)
    # Round before comparing -- a proposed amount computed as e.g.
    # emi_amount * 0.70 can land a fraction of a paisa under `minimum` from
    # float imprecision alone, and shouldn't be rejected for that.
    if round(proposed_amount, 2) < minimum:
        return {
            "account_id": account.account_id,
            "eligible": False,
            "reason": f"proposed amount {proposed_amount} is below the policy minimum of {minimum}",
            "minimum_amount": minimum,
        }

    return {
        "account_id": account.account_id,
        "eligible": True,
        "accepted_amount": round(proposed_amount, 2),
        "minimum_amount": minimum,
    }


@mcp.tool
def calculate_hypothetical(account_id: str, restructuring_type: str, extra_months: int | None = None) -> dict:
    """Calculate what a restructuring option would look like without
    committing to it. restructuring_type is 'extend_tenure' (pass
    extra_months, capped by policy) or 'one_time_settlement' (a discounted
    lump sum that closes the loan early). Remaining principal is
    approximated as emi_amount * months_remaining -- a simplification, not a
    real amortization schedule."""
    account = store.get_account_or_raise(account_id)
    if restructuring_type not in RESTRUCTURING_TYPES:
        raise ValueError(f"restructuring_type must be one of {RESTRUCTURING_TYPES}, got {restructuring_type!r}")

    block_reason = _blocked_from_automated_restructuring(account)
    if block_reason:
        return {"account_id": account.account_id, "eligible": False, "reason": block_reason}

    remaining_principal = account.emi_amount * account.months_remaining

    if restructuring_type == "extend_tenure":
        if extra_months is None or not (0 < extra_months <= MAX_RESTRUCTURING_EXTENSION_MONTHS):
            raise ValueError(
                f"extra_months must be between 1 and {MAX_RESTRUCTURING_EXTENSION_MONTHS}, got {extra_months!r}"
            )
        new_tenure = account.months_remaining + extra_months
        new_emi = round(remaining_principal / new_tenure, 2)
        return {
            "account_id": account.account_id,
            "restructuring_type": restructuring_type,
            "extra_months": extra_months,
            "new_months_remaining": new_tenure,
            "new_emi_amount": new_emi,
        }

    # one_time_settlement
    settlement_amount = round(remaining_principal * (1 - SETTLEMENT_DISCOUNT_PCT), 2)
    return {
        "account_id": account.account_id,
        "restructuring_type": restructuring_type,
        "remaining_principal": remaining_principal,
        "settlement_amount": settlement_amount,
        "discount_pct": SETTLEMENT_DISCOUNT_PCT,
    }
