"""A second, narrower mechanical check alongside grounding.py's --
that one verifies a stated URL/₹ amount traces back to something real;
this one verifies a specific kind of CLAIM (a restructuring/partial-
payment request is accepted, blocked, or has some outcome) was actually
checked via a real tool call this turn, not asserted from memory.

Found live via eval/realistic_conversation_benchmark.py's
many_operations_same_account_en scenario, round 4: on a dispute-blocked
account, deep into a long conversation, the borrower proposed a
concrete new amount ("can i at least pay like 20000 now") and the
model stated the block directly from established context -- correct in
substance, but never called propose_partial_payment for THIS specific
number, so grounding.py's check_grounding had nothing to catch (no
stray URL or ungrounded ₹ figure -- ₹20,000 came from the borrower's
own message). Two separate prompt refinements aimed at this exact
pattern didn't fix it; this mechanical check is the more reliable
alternative check_grounding's own design already argues for elsewhere.

Deliberately narrow, same spirit as grounding.py: only fires when BOTH
a concrete number AND explicit reduced/restructured-payment language
appear in the borrower's own message this turn -- a vague ask ("can
you lower my payment?") has nothing concrete to verify yet, and
skipping straight to stating a known block is legitimate there (see
agent/client.py's _CHECK_DISPUTE_BLOCK_FIRST).
"""

import re

from businessflow.guardrail.grounding import extract_plain_numbers, extract_rupee_amounts, extract_shorthand_amounts

_RESTRUCTURING_INTENT_RE = re.compile(
    r"\b(instead of|partial|only pay|just pay|can i pay|at least|less than|lower|reduce|stretch|extend|settle)\b",
    re.IGNORECASE,
)

# Any one of these actually checks a concrete proposal against real
# policy/account state -- if one of these ran this turn, the claim is
# verified regardless of what specific number it involved.
_VERIFYING_TOOLS = {"propose_partial_payment", "calculate_hypothetical", "escalate_to_human"}


class UnverifiedRestructuringClaim:
    def __init__(self, proposed_amount: float | None):
        self.proposed_amount = proposed_amount

    def __bool__(self) -> bool:
        return self.proposed_amount is not None

    def describe(self) -> str:
        return (
            f"borrower proposed a concrete amount (₹{self.proposed_amount:,.0f}) for a reduced/"
            "restructured payment this turn, but no propose_partial_payment/calculate_hypothetical/"
            "escalate_to_human call happened -- the reply's stated outcome is unverified"
        )


def check_unverified_restructuring_claim(user_message: str, tools_called_this_turn: set[str]) -> UnverifiedRestructuringClaim:
    if tools_called_this_turn & _VERIFYING_TOOLS:
        return UnverifiedRestructuringClaim(None)
    if not _RESTRUCTURING_INTENT_RE.search(user_message):
        return UnverifiedRestructuringClaim(None)

    amounts = (
        extract_rupee_amounts(user_message)
        | extract_plain_numbers(user_message)
        | extract_shorthand_amounts(user_message)
    )
    if not amounts:
        return UnverifiedRestructuringClaim(None)

    return UnverifiedRestructuringClaim(max(amounts))
