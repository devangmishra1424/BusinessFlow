"""Stage 1 of the report pipeline (blueprint's own §13: Gather -> Analyze
-> Write -> Accuracy check) -- pure Python, no LLM, no judgment: just the
real, countable facts every later stage builds on. Reuses ops/flags.py's
already-tested flag logic rather than recomputing it a second way.
"""

from businessflow.accounts import store
from businessflow.ops.flags import compute_flags


def gather_account_facts(account_ids: list[str] | None = None) -> list[dict]:
    """Real facts for every account (or a specific subset), as of today
    -- the same anchor (store.current_date()) every flag and days-past-
    due calculation elsewhere in this project already uses, so a report
    can never disagree with what a live tool call would say about the
    same account."""
    as_of = store.current_date()
    accounts = store.list_accounts() if account_ids is None else [store.get_account_or_raise(a) for a in account_ids]

    facts = []
    for account in accounts:
        flags = compute_flags(account, as_of)
        facts.append({
            "account_id": account.account_id,
            "borrower_name": account.borrower_name,
            "business_name": account.business_name,
            "days_past_due": account.days_past_due(as_of),
            "dispute_open": account.dispute_open,
            "broken_promise_count": account.broken_promise_count(),
            "flags": [{"label": f.label, "reason": f.reason} for f in flags],
            "open_escalations": [
                {"escalation_id": e.escalation_id, "reason": e.reason}
                for e in store.get_escalations_for_account(account.account_id)
                if e.resolved_at is None
            ],
        })
    return facts
