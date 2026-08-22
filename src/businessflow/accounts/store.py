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
from datetime import date

from businessflow.accounts.db import get_connection
from businessflow.accounts.models import Account, PaymentRecord, PromiseToPay

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


def add_promise(account_id: str, made_on: date, promised_date: date, promised_amount: float) -> None:
    get_connection().execute(
        "insert into promises (account_id, made_on, promised_date, promised_amount) values (%s, %s, %s, %s)",
        (account_id, made_on, promised_date, promised_amount),
    )


def open_dispute(account_id: str, reason: str) -> None:
    conn = get_connection()
    conn.execute("update accounts set dispute_open = true, updated_at = now() where account_id = %s", (account_id,))
    conn.execute("insert into disputes (account_id, reason) values (%s, %s)", (account_id, reason))


def create_escalation(account_id: str, reason: str) -> str:
    conn = get_connection()
    seq_number = conn.execute("select nextval('escalation_seq')").fetchone()["nextval"]
    escalation_id = f"ESC-{seq_number:04d}"
    conn.execute(
        "insert into escalations (escalation_id, account_id, reason) values (%s, %s, %s)",
        (escalation_id, account_id, reason),
    )
    return escalation_id


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
    get_connection().execute(
        "insert into events (account_id, event_type, details) values (%s, %s, %s)",
        (account_id, event_type, json.dumps(details, ensure_ascii=False, default=str)),
    )
