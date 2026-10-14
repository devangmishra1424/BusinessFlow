"""Aggregate operational signal on top of the existing events table --
every tool call, tool failure, and (now) conversation turn is already
logged there via accounts.store.log_event. This turns that raw log into
answerable questions: how many escalations this week, how often are
tools failing, is anything stuck.

No new schema -- same reasoning as memory/conversation_memory.py. This is
also the natural foundation for a future ops dashboard: whatever view
gets built there would need exactly these kinds of aggregate queries.
"""

from datetime import datetime, timedelta, timezone

from businessflow.accounts.db import get_connection


def event_counts_since(since: datetime | None = None) -> dict[str, int]:
    """Count of events by type since the given time (default: last 24h).
    A sudden spike in tool_call_failed, or any escalated_manually events,
    is the kind of thing an operator should actually look at -- this is
    the raw number that would back that alert."""
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

    rows = get_connection().execute(
        "select event_type, count(*) as n from events where created_at >= %s group by event_type order by n desc",
        (since,),
    ).fetchall()
    return {r["event_type"]: r["n"] for r in rows}


def escalation_rate(since: datetime | None = None) -> float:
    """Fraction of tool-calling turns that ended in escalate_to_human --
    a rising rate over time is a real signal that the agent is
    under-equipped for what it's being asked, not just noise."""
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

    # Counted in the database: one row back, however many tool calls the window holds. It used to fetch every
    # tool_called event with its details to count them in Python, which grew with the window (30 days) and made the
    # ops dashboard's escalation-rate card its slowest request.
    row = get_connection().execute(
        "select count(*) as calls, count(*) filter (where details->>'tool' = 'escalate_to_human') as escalations "
        "from events where created_at >= %s and event_type = 'tool_called'",
        (since,),
    ).fetchone()
    if not row["calls"]:
        return 0.0
    return row["escalations"] / row["calls"]
