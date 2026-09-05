"""The tool that hands a conversation off to a human."""

import logging

from businessflow.accounts import store
from businessflow.tools.payment_tools import calculate_hypothetical
from businessflow.tools.server import mcp

logger = logging.getLogger(__name__)


@mcp.tool(
    description=(
        "Hand this account off to a human agent, with the reason so they don't start "
        "cold. Used whenever policy requires a human (open dispute, repeated broken "
        "promises) or the agent is otherwise unsure."
    )
)
def escalate_to_human(account_id: str, reason: str) -> dict:
    """Hand this account off to a human agent, with the reason so they don't
    start cold. Used whenever policy requires a human (open dispute,
    repeated broken promises) or the agent is otherwise unsure."""
    store.get_account_or_raise(account_id)  # raises if the account doesn't exist
    escalation_id = store.create_escalation(account_id, reason)

    logger.info("escalation created id=%s account_id=%s reason=%s", escalation_id, account_id, reason)

    return {
        "escalation_id": escalation_id,
        "account_id": account_id,
        "reason": reason,
        "status": "queued_for_human",
    }


# There is no document-generation or email/delivery capability anywhere in
# this system (see agent/client.py's _NO_FABRICATED_ACTIONS) -- so a loan-
# closure-certificate/NOC request, a common ask once months_remaining
# reaches 0, was previously unhandled by any tool at all. This checks real
# eligibility and, when eligible, only queues a human to issue the actual
# document -- it never produces or sends one itself, the same reason
# escalate_to_human above exists rather than this system acting alone.
_CLOSURE_CERTIFICATE_REASON = (
    "Borrower requesting loan closure certificate/NOC -- fully repaid, "
    "needs human to issue the actual document"
)


@mcp.tool(
    description=(
        "A borrower asking for a loan closure certificate / NOC / proof their loan is "
        "fully paid off. If months_remaining is 0 (fully repaid), queues a human to "
        "issue the actual document and returns eligible=True -- this system cannot "
        "generate or send the certificate itself. If months_remaining is still > 0, "
        "returns eligible=False with the real value and does not escalate."
    )
)
def request_closure_certificate(account_id: str) -> dict:
    """A borrower asking for a loan closure certificate / NOC / proof their
    loan is fully paid off. Checks the account's real months_remaining:
    if it's 0 (fully repaid), queues a human to issue the actual document
    and returns eligible=True with the escalation details -- this system
    cannot generate or send the certificate itself, only hand it to a
    human who can. If months_remaining is still greater than 0, returns
    eligible=False with the real months_remaining value and does NOT
    create an escalation -- there's nothing yet for a human to act on."""
    account = store.get_account_or_raise(account_id)
    if account.months_remaining > 0:
        return {
            "account_id": account.account_id,
            "eligible": False,
            "months_remaining": account.months_remaining,
        }

    escalation_id = store.create_escalation(account.account_id, _CLOSURE_CERTIFICATE_REASON)
    logger.info(
        "closure certificate escalation created id=%s account_id=%s", escalation_id, account.account_id
    )
    return {
        "account_id": account.account_id,
        "eligible": True,
        "months_remaining": account.months_remaining,
        "escalation_id": escalation_id,
        "reason": _CLOSURE_CERTIFICATE_REASON,
        "status": "queued_for_human",
    }


# Found live via a real Telegram conversation: after calculate_hypothetical
# correctly computed a 3-month extension's real numbers, the model's reply
# described them as already applied ("the loan now has 17 months left")
# -- calculate_hypothetical's own docstring is explicit that it commits to
# nothing, and there is no tool anywhere in this system that writes a
# restructuring to the account. This tool is that missing "commit" step,
# and it still doesn't write the account directly -- it only queues the
# real proposed terms for a human to approve (see accounts/store.py's
# approve_restructuring / reject_restructuring, and ops/api.py's
# POST /escalations/{id}/approve|reject). Its return is always
# "pending_approval", never "done" -- so a reply built honestly from this
# result can't repeat that bug.
@mcp.tool(
    description=(
        "Once a borrower has explicitly agreed to a SPECIFIC extend-tenure proposal (the "
        "exact extra_months they were just quoted via calculate_hypothetical), call this "
        "to queue it for a human to approve. This tool itself never changes the account "
        "-- only a human approving it in the ops dashboard does. Returns status "
        "'pending_approval' with the real proposed terms if eligible, or the same "
        "ineligibility shape calculate_hypothetical returns if blocked."
    )
)
def propose_restructuring(account_id: str, extra_months: int) -> dict:
    """Once a borrower has explicitly agreed to a SPECIFIC extend-tenure
    proposal (the exact extra_months they were just quoted via
    calculate_hypothetical), call this to actually queue it for a human to
    approve. This tool itself never changes the account -- only a human
    approving it in the ops dashboard does. Returns status
    "pending_approval" with the real proposed terms if the account is
    eligible, or the same ineligibility shape calculate_hypothetical
    returns (eligible: False) if it's blocked (open dispute, etc)."""
    hypothetical = calculate_hypothetical(account_id, "extend_tenure", extra_months)
    if hypothetical.get("eligible") is False:
        return hypothetical

    proposed_changes = {
        "type": "extend_tenure",
        "extra_months": hypothetical["extra_months"],
        "new_months_remaining": hypothetical["new_months_remaining"],
        "new_emi_amount": hypothetical["new_emi_amount"],
    }
    reason = (
        f"Borrower agreed to extend tenure by {extra_months} month(s) -- "
        f"pending approval, new EMI would be ₹{hypothetical['new_emi_amount']:,.2f} "
        f"over {hypothetical['new_months_remaining']} months"
    )
    escalation_id = store.create_escalation(account_id, reason, proposed_changes=proposed_changes)
    logger.info(
        "restructuring proposal created id=%s account_id=%s extra_months=%s",
        escalation_id, account_id, extra_months,
    )
    return {
        "account_id": account_id,
        "escalation_id": escalation_id,
        "status": "pending_approval",
        **proposed_changes,
    }
