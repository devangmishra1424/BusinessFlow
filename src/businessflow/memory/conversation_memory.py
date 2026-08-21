"""Cross-session memory: lets a NEW conversation for a returning borrower
start with a short recap of what happened last time, instead of starting
blind on every single call. Built entirely on the existing events table
(no new schema, no migration) -- every conversation turn and tool call is
already logged there via accounts.store.log_event.

Deliberately a short, structured digest, not a raw transcript dump: after
the Groq daily-token-quota exhaustion hit during eval runs this session,
padding every new conversation's context with unbounded history is
exactly the kind of thing that would make that worse in production.
"""

from businessflow.accounts import store
from businessflow.accounts.db import get_connection

_RECAP_EVENT_TYPES = ("user_message", "assistant_message", "tool_called")
_MAX_RECAP_EVENTS = 6


def log_turn(account_id: str | None, role: str, content: str) -> None:
    """Persists one conversation turn as an event. role is "user" or
    "assistant". No-op without an account_id -- there's nothing to key
    cross-session memory to for an anonymous, no-account conversation."""
    if account_id is None:
        return
    store.log_event(account_id, f"{role}_message", {"content": content})


def recent_context_recap(account_id: str | None) -> str | None:
    """A short digest of the last few relevant events for this account
    (messages and tool calls, oldest first) -- for seeding a NEW
    conversation. None if there's no account_id, or no prior history
    (a genuine first contact)."""
    if account_id is None:
        return None

    rows = get_connection().execute(
        "select event_type, details from events "
        "where account_id = %s and event_type = any(%s) "
        "order by created_at desc limit %s",
        (account_id, list(_RECAP_EVENT_TYPES), _MAX_RECAP_EVENTS),
    ).fetchall()
    if not rows:
        return None

    lines = []
    for row in reversed(rows):  # oldest first -- matches how it actually happened
        event_type, details = row["event_type"], row["details"]
        if event_type == "user_message":
            lines.append(f"Borrower said: {details['content']}")
        elif event_type == "assistant_message":
            lines.append(f"You said: {details['content']}")
        elif event_type == "tool_called":
            lines.append(f"You called {details['tool']}({details['arguments']})")

    return "Recap of your last contact with this borrower:\n" + "\n".join(lines)
