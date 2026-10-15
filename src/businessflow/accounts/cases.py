"""One case, one conversation.

A "case" is anything a borrower raised that a person has to deal with: a dispute, a payment they say they
already made (a payment claim), or an escalation (a request for more time, a call-back, a closure
certificate, a system notice). Before, each lived in its own place: the borrower read requests in one panel
and our replies in another, ops read disputes on one tab and wrote messages on a different one, and a reply
from ops had no link to the thing it answered.

Now every message about a case carries that case's reference in its event details (the events table is
append-only; nothing here needs a new table):

    disputes            "D-<disputes.id>"        e.g. "D-12"
    escalations         the escalation id        e.g. "ESC-0012"

and both dashboards show a case as one thread: what the borrower said, then every message about it, then
how it ended. Pure functions only (no database), so the thread logic is testable on its own.
"""

from __future__ import annotations

import re
from datetime import datetime

# disputes.kind
DISPUTE = "dispute"
PAYMENT_CLAIM = "payment_claim"  # "I've already paid" -- ops checks the bank record, then confirms or closes it
DISPUTE_KINDS = (DISPUTE, PAYMENT_CLAIM)

# A message written about one case, by ops (delivered over Telegram when linked) or by the borrower.
CASE_MESSAGE_EVENT = "case_message"

# Every event type that can belong to a case's thread. The two notification types are the outcome messages ops
# already sent (approve/reject, dispute resolved); they carry a case_ref from now on.
THREAD_EVENT_TYPES = (CASE_MESSAGE_EVENT, "restructuring_decision_notified", "dispute_resolution_notified")

AUTHORS = ("borrower", "ops")

_DISPUTE_REF = re.compile(r"^D-([1-9][0-9]{0,17})$")
_ESCALATION_REF = re.compile(r"^ESC-[0-9]{4,12}$")


def dispute_case_ref(dispute_id: int) -> str:
    return f"D-{dispute_id}"


def parse_case_ref(case_ref: str) -> tuple[str, int | str] | None:
    """("dispute", <id>) or ("escalation", "ESC-0012"); None for anything else. Every case_ref that comes
    from a request goes through this before it reaches a query."""
    text = (case_ref or "").strip()
    match = _DISPUTE_REF.match(text)
    if match:
        return "dispute", int(match.group(1))
    if _ESCALATION_REF.match(text):
        return "escalation", text
    return None


def thread_entry(event_type: str, details: dict, created_at: datetime) -> dict | None:
    """One thread message from one event row; None when the row isn't a case message (no ref, no text)."""
    case_ref = details.get("case_ref")
    message = (details.get("message") or "").strip()
    if not case_ref or not message:
        return None
    author = details.get("author")
    if event_type != CASE_MESSAGE_EVENT or author not in AUTHORS:
        author = "ops"  # an outcome notice is always from the team
    return {
        "case_ref": case_ref,
        "author": author,
        "message": message,
        "created_at": created_at,
        "delivered_via_telegram": bool(details.get("delivered_via_telegram", False)),
    }


def group_threads(rows: list[dict]) -> dict[str, list[dict]]:
    """{case_ref: [message, ...]} oldest first, from event rows ({event_type, details, created_at}) in any order."""
    threads: dict[str, list[dict]] = {}
    for row in rows:
        entry = thread_entry(row["event_type"], row["details"] or {}, row["created_at"])
        if entry is not None:
            threads.setdefault(entry["case_ref"], []).append(entry)
    for messages in threads.values():
        messages.sort(key=lambda m: m["created_at"])
    return threads
