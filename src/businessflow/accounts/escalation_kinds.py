"""What kind of thing is this escalation?

Every escalation is one row with one free-text reason, and one Approve/Reject
pair in the ops dashboard -- so a fraud claim, a guardrail artifact, a
chronic-delinquency notice and a borrower's request for a call all looked the
same, were shown to the borrower in the same internal wording, and got the
same "Good news -- your request has been approved" message on approval.

This module is the single place that knows the *deterministic* reason strings
the code itself writes (so the producers and the classifier can't drift) and
turns a row into a kind. It deliberately does NOT try to understand
agent-written free text: that is written in the borrower's own language (a
Hindi grievance reason exists in the live queue), so anything it doesn't
recognise is just "request" -- shown neutrally and never guessed at. Making
kind a structured argument of the escalation tool would be the real fix
(it changes the tool's signature, so it is left for a deliberate decision).
"""

# Written by code, verbatim -- producers import these so the strings can't drift.
GUARDRAIL_REASON_PREFIX = "Guardrail blocked a reply:"
CHRONIC_DELINQUENCY_REASON = (
    "Chronically overdue -- past the mandatory escalation threshold with repeated "
    "reminders sent and no resolution"
)
BROKEN_PROMISE_PATTERN_PREFIX = "Broken promise pattern"
CLOSURE_CERTIFICATE_REASON = (
    "Borrower requesting loan closure certificate/NOC -- fully repaid, "
    "needs human to issue the actual document"
)
DEFAULT_HUMAN_AGENT_REASON = "Borrower requested a human agent"  # dashboard / Telegram /agent, with or without a typed reason
OPS_CALL_NUDGE_PREFIX = "Repeated failed contact attempt"  # raised by staff from the call-log nudge
FRAUD_REASON_PREFIX = "SUSPECTED FRAUD/IDENTITY CLAIM"  # the agent prompt requires this literal prefix

# kinds
RESTRUCTURING = "restructuring"
CLOSURE = "closure"
CALLBACK = "callback"
FRAUD = "fraud"
REQUEST = "request"  # anything else an agent/borrower wrote (any language)
SYSTEM = "system"  # raised by the system or by staff -- never a borrower's own request

_SYSTEM_PREFIXES = (
    GUARDRAIL_REASON_PREFIX,
    CHRONIC_DELINQUENCY_REASON,
    BROKEN_PROMISE_PATTERN_PREFIX,
    OPS_CALL_NUDGE_PREFIX,
)


def classify_escalation(reason: str, proposed_changes: dict | None = None) -> str:
    """Best-effort kind for display and notification wording only -- never
    for any decision that touches the account."""
    if proposed_changes:
        return RESTRUCTURING
    text = (reason or "").strip()
    if text.startswith(_SYSTEM_PREFIXES):
        return SYSTEM
    if text.startswith(FRAUD_REASON_PREFIX):
        return FRAUD
    if text == CLOSURE_CERTIFICATE_REASON:
        return CLOSURE
    if text.startswith(DEFAULT_HUMAN_AGENT_REASON):
        return CALLBACK
    return REQUEST


def is_system_ticket(reason: str, proposed_changes: dict | None = None) -> bool:
    return classify_escalation(reason, proposed_changes) == SYSTEM


# Triage order for the ops queue: lower sorts first. A fraud/identity claim is
# "categorically more urgent than an ordinary dispute" (agent/client.py), so it
# must not sit at the bottom just because it is the newest ticket.
URGENCY_RANK = {FRAUD: 0, REQUEST: 1, RESTRUCTURING: 2, CALLBACK: 2, CLOSURE: 3, SYSTEM: 4}
