"""Tools that touch money -- no real payment gateway exists anywhere in
this project. generate_payment_link's link is still real in every sense
that matters for this demo, though: it's a genuine, single-use, expiring
token (accounts.store.create_payment_token) that a real confirm page
(channels/browser_api.py's /pay/{token}) can actually redeem, which
really does call store.record_payment and really does move
months_remaining/emi_due_date/payment_history forward -- there's just no
real bank or card processor sitting behind the "confirm" click."""

import ast
import operator
import os

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


def _chat_app_base_url() -> str:
    # No real payment processor exists, so there's no "return URL" a real
    # gateway would redirect back to -- this is just this project's own
    # borrower-facing app, wherever it's actually reachable (localhost in
    # dev, the real deployed domain in production). Defaults to localhost
    # so a fresh checkout works before anyone sets this.
    return os.environ.get("CHAT_APP_BASE_URL", "http://localhost:8000")


@mcp.tool
def generate_payment_link(account_id: str, amount: float) -> dict:
    """Generate a payment link for the given amount -- a real, single-use,
    expiring link (see accounts.store.create_payment_token) that actually
    records a real payment when confirmed on the page it points to. No
    real bank/card processor sits behind it -- confirming just runs this
    project's own record_payment, the same as a real gateway's webhook
    would, minus the gateway."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    token = store.create_payment_token(account_id, amount)
    return {
        "account_id": account_id,
        "amount": amount,
        "payment_link": f"{_chat_app_base_url()}/pay/{token}",
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


# guardrail/grounding.py's check_grounding only trusts a number that
# already appears in a real tool result or the borrower's own words --
# by design, it can't verify arbitrary arithmetic on those numbers (its
# own module docstring documents this as a deliberate trade-off, not an
# oversight: checking derived math for real would need actual
# computation, and doing that INSIDE the guardrail risks its own bugs).
# Found live, repeatedly: the model doing that math in its own reply
# text instead -- "you'd pay roughly 95% of ₹1,084,741.92, which is
# about ₹1,031,504.84" -- got blocked outright, leaving the borrower with
# no number at all for a completely reasonable question. compute() closes
# this the right way: the arithmetic happens in a real tool call, so its
# result is a genuine tool message like any other, grounded through the
# exact same mechanism as every other real figure -- no special-casing
# arithmetic in the guardrail at all, and no way for the model to slip an
# invented number through it either, since the result is computed here,
# not asserted by the model.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval_arithmetic(node: ast.AST) -> float:
    """Walks a parsed expression, only ever evaluating plain numeric
    literals combined with +, -, *, / and unary +/-. Deliberately an AST
    walk with an explicit allowlist, not a regex-filtered eval() -- no
    name lookup, attribute access, subscripting, or call node is ever a
    reachable branch here, so there is no code-execution surface to
    reason about, unlike restricting eval()'s input text ever could
    fully guarantee."""
    if isinstance(node, ast.Expression):
        return _safe_eval_arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_arithmetic(node.left)
        right = _safe_eval_arithmetic(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_arithmetic(node.operand))
    raise ValueError(f"expression contains something other than plain arithmetic: {ast.dump(node)}")


@mcp.tool
def compute(expression: str) -> dict:
    """Evaluate a plain arithmetic expression -- numbers and + - * / ( )
    only -- and return the exact result to state verbatim. Call this
    whenever you need to state a number that's DERIVED from other real
    figures already established this conversation (a percentage of a
    real balance, a sum, a difference) and no other tool already returns
    it directly. Never do this arithmetic yourself in your reply text:
    the grounding guardrail only trusts a number that came from a real
    tool result, and will block your own mental math even when the
    inputs were real and the arithmetic was correct.

    Example: the borrower's real outstanding balance is ₹1,084,741.92
    (from get_payment_status) and they ask what a 5% discount would look
    like -- call compute("1084741.92 * 0.95"), then state ITS result, not
    a number you calculated yourself.

    Raises ValueError for anything that isn't plain arithmetic (a name,
    a function call, anything resembling code) -- this is a calculator,
    not a code execution tool."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_arithmetic(tree)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"{expression!r} isn't a plain arithmetic expression: {e}") from e
    return {"expression": expression, "result": round(result, 2)}
