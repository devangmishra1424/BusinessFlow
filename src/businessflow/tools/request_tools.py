"""What the borrower has asked of us, and the two ways they can ask more: the follow-up tools.

Added 2026-10-23 after an audit of every conversational surface against what the product can now do: a borrower could
raise a dispute, a payment claim or a restructuring request in chat, but never ask "what happened to it?"; the team
could message them, but a typed answer in chat went to the assistant, not the team; and a late fee became a real
charge only the team can waive, with no way to ask for that.

- get_my_requests: every request on the account (open first), what the team last said on each, and recent messages
  from the team that are not about a request.
- request_fee_waiver: asks the team to remove a late fee still owed -- a request in their inbox, never a waiver.
- reply_to_team: answers the team's most recent message (accounts/store.py reply_to_team_message), the same as the
  Reply box on the dashboard or a Telegram reply.
"""

from businessflow.accounts import cases, store
from businessflow.accounts.escalation_kinds import (
    CLOSURE,
    FEE_WAIVER_REASON_PREFIX,
    REPLY_REASON_PREFIX,
    RESTRUCTURING,
    SYSTEM,
    classify_escalation,
)
from businessflow.tools import validation
from businessflow.tools.server import mcp

_MAX_ITEMS = 10
_KIND_WORDS = {
    RESTRUCTURING: "request for more time on the EMI",
    CLOSURE: "loan closure certificate",
    cases.PAYMENT_CLAIM: "payment the borrower reported",
    cases.DISPUTE: "question about a charge or amount",
}


def _last_team_word(thread: list[dict]) -> str | None:
    team = [m for m in thread if m.get("author") == "ops"]
    return team[-1]["message"] if team else None


@mcp.tool(
    description=(
        "What has happened to everything this borrower asked us: each request (a payment they reported, a question "
        "about a charge, a request for more time, a closure certificate, a fee waiver, their reply to our message), "
        "whether it is open or closed, how it ended, and the last thing our team said on it -- plus recent messages "
        "from our team that are not about a request. Use it for 'did you check my payment?', 'what happened to my "
        "request?', 'did anyone reply?'. Quote what it returns; never guess a status or a decision."
    )
)
def get_my_requests(account_id: str) -> dict:
    """The borrower's own requests, open first then most recent, bounded to _MAX_ITEMS; system tickets (guardrail
    notices, staff-raised cases) are not theirs and are left out."""
    store.get_account_or_raise(account_id)
    threads = store.get_case_threads(account_id)
    items = []
    for d in store.get_disputes_for_account(account_id):
        ref = cases.dispute_case_ref(d["id"]) if d.get("id") is not None else None
        items.append({
            "ref": ref, "what": _KIND_WORDS.get(d.get("kind") or cases.DISPUTE, "request"),
            "their_words": (d["reason"] or "")[:300], "status": "open" if d["status"] == "open" else "closed",
            "opened": d["opened_at"].date().isoformat(), "outcome": d.get("resolution_note"),
            "last_from_our_team": _last_team_word(threads.get(ref, [])) if ref else None,
        })
    for e in store.get_escalations_for_account(account_id):
        kind = classify_escalation(e.reason, e.proposed_changes)
        if kind == SYSTEM:
            continue
        what = ("fee waiver request" if e.reason.startswith(FEE_WAIVER_REASON_PREFIX)
                else "reply to our message" if e.reason.startswith(REPLY_REASON_PREFIX)
                else _KIND_WORDS.get(kind, "request to our team"))
        items.append({
            "ref": e.escalation_id, "what": what, "their_words": e.reason[:300],
            "status": "open" if e.status == "queued_for_human" else ("declined" if e.status == "rejected" else "closed"),
            "opened": e.created_at.date().isoformat(), "outcome": e.resolution_reason,
            "last_from_our_team": _last_team_word(threads.get(e.escalation_id, [])),
        })
    open_items = sorted((i for i in items if i["status"] == "open"), key=lambda i: i["opened"])
    closed = sorted((i for i in items if i["status"] != "open"), key=lambda i: i["opened"], reverse=True)
    messages = [
        {"sent": m["created_at"].date().isoformat(), "message": m["message"][:400],
         "you_replied_as": m.get("replied_case_ref")}
        for m in store.get_borrower_messages(account_id) if m["kind"] == "message"
    ][:5]
    return {
        "account_id": account_id,
        "open_requests": open_items[:_MAX_ITEMS],
        "recently_closed": closed[: max(0, _MAX_ITEMS - len(open_items))],
        "messages_from_our_team": messages,
    }


@mcp.tool(
    description=(
        "The borrower asks us to remove (waive) a late fee still owed. This only sends the request to a person on our "
        "team with the borrower's reason -- it never removes the fee. Say exactly that: the team will look at it and "
        "reply; until then the fee is still owed. Returns no_fee_owed=true if there is no late fee to waive."
    )
)
def request_fee_waiver(account_id: str, reason: str) -> dict:
    """Opens (or, for the same reason, returns the existing) request in the ops Inbox, naming the fees owed; the team
    waives from the account page (ops/api.py waive_account_charge) with their name and a reason the borrower is sent."""
    account = store.get_account_or_raise(account_id)
    reason = validation.free_text(reason, "reason")
    owed = account.open_charges()
    if not owed:
        return {"account_id": account_id, "no_fee_owed": True, "requested": False}
    fees = ", ".join(f"₹{c.outstanding:,.0f} for the {c.for_due_date.isoformat()} EMI" for c in owed)
    case_ref = store.create_escalation(account_id, f"{FEE_WAIVER_REASON_PREFIX} ({fees}): {reason}")
    return {
        "account_id": account_id, "requested": True, "case_ref": case_ref, "fees_owed": fees,
        "what_happens_next": "A person on our team decides; their answer comes to the borrower. The fee is still owed until then.",
    }


@mcp.tool(
    description=(
        "The borrower's answer to the latest message our team sent them (e.g. we asked why the EMI is late and they "
        "explain). Sends their words to the team as a reply -- it becomes, or joins, a request the team answers. Use "
        "it when they are clearly answering our message; their own new question is not a reply. Returns "
        "no_team_message=true if our team has sent them nothing to reply to."
    )
)
def reply_to_team(account_id: str, message: str) -> dict:
    store.get_account_or_raise(account_id)
    message = validation.free_text(message, "message")
    latest = next((m for m in store.get_borrower_messages(account_id) if m["kind"] == "message"), None)
    if latest is None:
        return {"account_id": account_id, "no_team_message": True}
    result = store.reply_to_team_message(account_id, latest["id"], message, via="assistant")
    return {"account_id": account_id, "case_ref": result["case_ref"], "new_request": result["opened"],
            "replying_to": latest["message"][:200]}
