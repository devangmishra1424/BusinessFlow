"""Account store, backed by real Postgres (Supabase) tables instead of the
in-memory dict this started as. Public function signatures are unchanged
from the mock -- the 8 tools that call these never needed to change for
this swap, only the internals did.

DEMO_TODAY still anchors "today" for the seeded demo accounts, for the
same reason as before: days-past-due and grace-period state are computed
relative to a fixed date, not the real clock, so the demo can't silently
drift out of its intended scenario as real time passes.
"""

import json
import logging
from datetime import date, datetime

import psycopg

from businessflow.accounts.db import get_connection
from businessflow.accounts.models import Account, Escalation, PaymentRecord, PromiseToPay

logger = logging.getLogger(__name__)

DEMO_TODAY = date(2026, 8, 21)


def _load_payment_history(account_id: str) -> list[PaymentRecord]:
    rows = get_connection().execute(
        "select payment_date, amount, on_time from payment_history where account_id = %s order by payment_date",
        (account_id,),
    ).fetchall()
    return [PaymentRecord(date=r["payment_date"], amount=float(r["amount"]), on_time=r["on_time"]) for r in rows]


def _load_promises(account_id: str) -> list[PromiseToPay]:
    rows = get_connection().execute(
        "select made_on, promised_date, promised_amount, kept from promises where account_id = %s order by promised_date",
        (account_id,),
    ).fetchall()
    return [
        PromiseToPay(made_on=r["made_on"], promised_date=r["promised_date"], promised_amount=float(r["promised_amount"]), kept=r["kept"])
        for r in rows
    ]


def _row_to_account(row: dict) -> Account:
    return Account(
        account_id=row["account_id"],
        borrower_name=row["borrower_name"],
        business_name=row["business_name"],
        phone_number=row["phone_number"],
        language_preference=row["language_preference"],
        loan_type=row["loan_type"],
        principal_amount=float(row["principal_amount"]),
        emi_amount=float(row["emi_amount"]),
        tenure_months=row["tenure_months"],
        months_remaining=row["months_remaining"],
        emi_due_date=row["emi_due_date"],
        nach_mandate_active=row["nach_mandate_active"],
        dispute_open=row["dispute_open"],
        risk_tier=row["risk_tier"],
        payment_history=_load_payment_history(row["account_id"]),
        promises=_load_promises(row["account_id"]),
    )


def get_account(account_id: str) -> Account | None:
    row = get_connection().execute("select * from accounts where account_id = %s", (account_id,)).fetchone()
    return _row_to_account(row) if row else None


def get_account_or_raise(account_id: str) -> Account:
    account = get_account(account_id)
    if account is None:
        raise ValueError(f"No account found for account_id={account_id!r}")
    return account


def list_accounts() -> list[Account]:
    rows = get_connection().execute("select * from accounts order by account_id").fetchall()
    return [_row_to_account(r) for r in rows]


def get_account_by_phone(phone_number: str) -> Account | None:
    row = get_connection().execute("select * from accounts where phone_number = %s", (phone_number,)).fetchone()
    return _row_to_account(row) if row else None


def current_date() -> date:
    """The single seam tools call through for 'today'. Returns the fixed demo
    anchor for now; swap this one function for date.today() once this runs
    against real, non-seeded accounts instead of the demo data."""
    return DEMO_TODAY


def add_promise(account_id: str, made_on: date, promised_date: date, promised_amount: float) -> bool:
    """Returns True if a new promise row was inserted, False if an
    identical one (same account, made today, same promised date/amount)
    already existed -- a duplicate tool call (a retry, or the model
    calling this twice in one turn) logs the promise once, not twice.

    This is a plain check-then-act, not a DB-enforced constraint -- fine
    for the realistic threat here (one conversation's sequential tool
    calls), not a guarantee against genuinely concurrent writers racing
    on the exact same promise. A unique index would be the fuller fix
    if that ever becomes a real risk."""
    existing = get_connection().execute(
        "select id from promises where account_id = %s and made_on = %s "
        "and promised_date = %s and promised_amount = %s",
        (account_id, made_on, promised_date, promised_amount),
    ).fetchone()
    if existing:
        return False
    get_connection().execute(
        "insert into promises (account_id, made_on, promised_date, promised_amount) values (%s, %s, %s, %s)",
        (account_id, made_on, promised_date, promised_amount),
    )
    return True


def open_dispute(account_id: str, reason: str) -> bool:
    """Returns True if this call actually opened a new dispute, False if
    the account already had one open -- flagging an already-disputed
    account is a no-op, not a second dispute-log entry. Same
    check-then-act caveat as add_promise above."""
    conn = get_connection()
    row = conn.execute("select dispute_open from accounts where account_id = %s", (account_id,)).fetchone()
    if row and row["dispute_open"]:
        return False
    conn.execute("update accounts set dispute_open = true, updated_at = now() where account_id = %s", (account_id,))
    conn.execute("insert into disputes (account_id, reason) values (%s, %s)", (account_id, reason))
    return True


def create_escalation(account_id: str, reason: str) -> str:
    """Idempotent against an exact-duplicate call: if an unresolved
    escalation with this same account_id + reason already exists,
    returns its existing escalation_id instead of opening a second
    ticket for the same thing. A genuinely different reason still opens
    a new escalation -- this only collapses true repeats."""
    conn = get_connection()
    existing = conn.execute(
        "select escalation_id from escalations where account_id = %s and reason = %s and resolved_at is null",
        (account_id, reason),
    ).fetchone()
    if existing:
        return existing["escalation_id"]
    seq_number = conn.execute("select nextval('escalation_seq')").fetchone()["nextval"]
    escalation_id = f"ESC-{seq_number:04d}"
    conn.execute(
        "insert into escalations (escalation_id, account_id, reason) values (%s, %s, %s)",
        (escalation_id, account_id, reason),
    )
    return escalation_id


def _row_to_escalation(row: dict) -> Escalation:
    return Escalation(
        escalation_id=row["escalation_id"],
        account_id=row["account_id"],
        reason=row["reason"],
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )


def get_escalations_for_account(account_id: str) -> list[Escalation]:
    rows = get_connection().execute(
        "select escalation_id, account_id, reason, status, created_at, resolved_at "
        "from escalations where account_id = %s order by created_at desc",
        (account_id,),
    ).fetchall()
    return [_row_to_escalation(r) for r in rows]


def list_open_escalations() -> list[Escalation]:
    """The ops queue: every escalation still waiting on a human, oldest
    first -- the order a human should actually work through them in."""
    rows = get_connection().execute(
        "select escalation_id, account_id, reason, status, created_at, resolved_at "
        "from escalations where status = 'queued_for_human' order by created_at",
    ).fetchall()
    return [_row_to_escalation(r) for r in rows]


def count_recent_events(account_id: str, event_type: str, since: datetime) -> int:
    row = get_connection().execute(
        "select count(*) as n from events where account_id = %s and event_type = %s and created_at >= %s",
        (account_id, event_type, since),
    ).fetchone()
    return row["n"]


def has_recent_event_with_detail(account_id: str, event_type: str, since: datetime, detail_key: str, detail_value) -> bool:
    """True if a matching event already exists -- the idempotency check
    the proactive outbound pass uses to avoid re-sending the same
    reminder kind to the same account twice in one day."""
    row = get_connection().execute(
        "select 1 from events where account_id = %s and event_type = %s and created_at >= %s "
        "and details->>%s = %s limit 1",
        (account_id, event_type, since, detail_key, str(detail_value)),
    ).fetchone()
    return row is not None


def verify_account_key(account_id: str, access_key: str) -> bool:
    """Checks a borrower-supplied key against the account's fixed PIN
    (assigned at seed time -- there's no real sign-up flow here). Used
    once, at conversation start, to gate access to that account's data;
    never returns the key itself, only whether it matched."""
    row = get_connection().execute(
        "select access_key from accounts where account_id = %s", (account_id,)
    ).fetchone()
    return row is not None and row["access_key"] == access_key


def log_event(account_id: str | None, event_type: str, details: dict) -> None:
    """account_id is a foreign key -- a value that doesn't correspond to
    a real account (e.g. a hallucinated account_id the model passed to
    a tool that doesn't itself validate it, like check_policy) would
    otherwise crash this insert outright. A logging side-effect failing
    must never take down a tool call that already succeeded (or a
    failure event trying to record why a DIFFERENT call failed), so this
    falls back to logging with no specific borrower instead of losing
    the event entirely -- found live, when exactly this happened during
    eval/reasoning_accuracy.py's general-Q&A scenario."""
    payload = json.dumps(details, ensure_ascii=False, default=str)
    try:
        get_connection().execute(
            "insert into events (account_id, event_type, details) values (%s, %s, %s)",
            (account_id, event_type, payload),
        )
    except psycopg.errors.ForeignKeyViolation:
        logger.warning(
            "log_event: account_id=%r doesn't exist -- logging event_type=%r with no specific borrower instead",
            account_id, event_type,
        )
        get_connection().execute(
            "insert into events (account_id, event_type, details) values (%s, %s, %s)",
            (None, event_type, payload),
        )
