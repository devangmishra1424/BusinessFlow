"""Seeds the 4 demo accounts (and their payment history / promises) into
the real Postgres tables. Idempotent -- clears existing demo rows first,
so it's safe to re-run after editing the seed data below.

Every date below is computed relative to UTC "today" at the moment this
runs, not a fixed literal -- re-running this script (e.g. the morning of
a demo) always produces a fresh, internally consistent "today" the whole
dashboard agrees with (accounts.store.current_date() also anchors to
UTC, not the host's local timezone -- see that function's docstring for
a real bug this avoided: a naive date.today() disagrees with UTC for
part of every day in any timezone ahead of it, e.g. IST's 00:00-05:30
window, which would have made this seed data's own "3 days past due"
story off by a day during exactly that window). The *shape* of the
story (which account is how overdue, in what order payments/promises
happened) is preserved exactly via fixed day-offsets from each account's
emi_due_date -- only the anchor point moves.

Concurrency: this runs as a pytest fixture (tests/conftest.py's
reseed_accounts) before tests that touch these accounts, and in practice
that means it can run from multiple test processes against the same
shared Supabase database around the same time (observed live: two
workflow runs' test phases overlapping). accounts.db.get_connection()
checks out a NEW pooled connection per .execute() call, autocommitted --
so the delete-then-insert sequence below is NOT atomic across a single
pooled connection, and two overlapping runs could freely interleave:
one run's DELETE FROM accounts removing a row a concurrent run's INSERT
INTO payment_history still expects to exist (ForeignKeyViolation), or
two runs both trying to INSERT the same account_id (UniqueViolation) --
both actually seen, not hypothetical. Fixed here by opening one raw
connection for the whole operation, holding a transaction-scoped
Postgres advisory lock (auto-released on commit or rollback, so a crash
mid-run can't leave it stuck) so concurrent reseeds queue up instead of
interleaving, and upserting accounts instead of delete-then-insert so
the account row is never briefly absent for a child-table insert to
reference.

Run: python -m scripts.seed_accounts
"""

import os
from datetime import date, datetime, timedelta, timezone

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

# Explicit here rather than relying on an import of businessflow.accounts.*
# to load it as a side effect (this script no longer imports that package
# at all, now that current_date()/DEMO_TODAY are gone) -- run standalone
# (python -m scripts.seed_accounts) same as any other entry point.
load_dotenv()

_ACCOUNT_IDS = ["BF-1001", "BF-1002", "BF-1003", "BF-1004"]

# Arbitrary, fixed key for the advisory lock -- just needs to be the same
# across every process seeding these same demo accounts, so it doesn't
# matter what the number is, only that it's consistent.
_SEED_LOCK_KEY = 8_401_002

# Days past due, as of whenever this seed actually runs -- the one place
# the "story" (who's how overdue) is authored; every other date below is
# derived from these via fixed day-offsets, never a literal date.
_DAYS_PAST_DUE = {
    "BF-1001": 3,
    "BF-1002": 11,
    "BF-1003": 20,
    "BF-1004": 12,
}


def _build_accounts(today: date) -> list[dict]:
    """Fixed, known PINs for the demo accounts -- there's no real sign-up
    flow here, so the key has to be assigned at seed time instead, and
    handed to whoever's demoing this rather than generated fresh per
    "sign up". Only emi_due_date is date.today()-relative; everything
    else about each account is static."""
    return [
        {
            "account_id": "BF-1001",
            "borrower_name": "Priya Sharma",
            "business_name": "Cotton Threads Boutique",
            "phone_number": "+919812345001",
            "language_preference": "hinglish",
            "loan_type": "Working Capital Loan",
            "principal_amount": 250_000,
            "emi_amount": 12_500,
            "tenure_months": 24,
            "months_remaining": 14,
            "emi_due_date": (today - timedelta(days=_DAYS_PAST_DUE["BF-1001"])).isoformat(),
            "nach_mandate_active": True,
            "dispute_open": False,
            "risk_tier": "low",
            "access_key": "482913",
        },
        {
            "account_id": "BF-1002",
            "borrower_name": "Arjun Mehta",
            "business_name": "Mehta Hardware & Electricals",
            "phone_number": "+919812345002",
            "language_preference": "hi",
            "loan_type": "Equipment Loan",
            "principal_amount": 500_000,
            "emi_amount": 22_000,
            "tenure_months": 36,
            "months_remaining": 20,
            "emi_due_date": (today - timedelta(days=_DAYS_PAST_DUE["BF-1002"])).isoformat(),
            "nach_mandate_active": False,
            "dispute_open": False,
            "risk_tier": "medium",
            "access_key": "716044",
        },
        {
            "account_id": "BF-1003",
            "borrower_name": "Fatima Khan",
            "business_name": "Khan Textiles Wholesale",
            "phone_number": "+919812345003",
            "language_preference": "hinglish",
            "loan_type": "Working Capital Loan",
            "principal_amount": 800_000,
            "emi_amount": 35_000,
            "tenure_months": 30,
            "months_remaining": 11,
            "emi_due_date": (today - timedelta(days=_DAYS_PAST_DUE["BF-1003"])).isoformat(),
            "nach_mandate_active": False,
            "dispute_open": True,  # disputes an incorrectly applied late fee
            "risk_tier": "high",
            "access_key": "930571",
        },
        {
            "account_id": "BF-1004",
            "borrower_name": "Ravi Iyer",
            "business_name": "Iyer Auto Spares",
            "phone_number": "+919812345004",
            "language_preference": "en",
            "loan_type": "Business Expansion Loan",
            "principal_amount": 600_000,
            "emi_amount": 28_000,
            "tenure_months": 30,
            "months_remaining": 22,
            "emi_due_date": (today - timedelta(days=_DAYS_PAST_DUE["BF-1004"])).isoformat(),
            "nach_mandate_active": True,
            "dispute_open": False,
            "risk_tier": "medium",
            "access_key": "205839",
        },
    ]


# (account_id, days before that account's emi_due_date, amount, on_time) --
# offsets computed once, by hand, from the original fixed-date seed data
# (e.g. BF-1001's three payments were exactly 92/61/31 days before its old
# 2026-08-18 due date), so the relative shape of the payment history is
# unchanged, only its anchor point moves with emi_due_date.
_PAYMENT_HISTORY_OFFSETS = [
    ("BF-1001", 92, 12_500, True),
    ("BF-1001", 61, 12_500, True),
    ("BF-1001", 31, 12_500, True),
    ("BF-1002", 92, 22_000, True),
    ("BF-1002", 58, 22_000, False),
    ("BF-1002", 22, 22_000, False),
    ("BF-1003", 61, 35_000, True),
    # July and August EMIs both missed entirely -- no records for them.
    ("BF-1004", 61, 28_000, True),
    ("BF-1004", 31, 28_000, True),
]

# (account_id, made_on days before due, promised_date days before due, amount, kept)
_PROMISE_OFFSETS = [
    ("BF-1002", 56, 51, 22_000, False),
    ("BF-1002", 31, 27, 22_000, True),
    ("BF-1003", 52, 47, 35_000, False),
    ("BF-1003", 27, 22, 35_000, False),
]


def _build_payment_history(accounts_by_id: dict[str, dict]) -> list[tuple]:
    rows = []
    for account_id, days_before_due, amount, on_time in _PAYMENT_HISTORY_OFFSETS:
        due = date.fromisoformat(accounts_by_id[account_id]["emi_due_date"])
        rows.append((account_id, (due - timedelta(days=days_before_due)).isoformat(), amount, on_time))
    return rows


def _build_promises(accounts_by_id: dict[str, dict]) -> list[tuple]:
    rows = []
    for account_id, made_on_offset, promised_offset, amount, kept in _PROMISE_OFFSETS:
        due = date.fromisoformat(accounts_by_id[account_id]["emi_due_date"])
        made_on = (due - timedelta(days=made_on_offset)).isoformat()
        promised_date = (due - timedelta(days=promised_offset)).isoformat()
        rows.append((account_id, made_on, promised_date, amount, kept))
    return rows


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")

    today = datetime.now(timezone.utc).date()
    accounts = _build_accounts(today)
    accounts_by_id = {a["account_id"]: a for a in accounts}
    payment_history = _build_payment_history(accounts_by_id)
    promises = _build_promises(accounts_by_id)

    # One connection, one transaction, for the whole operation -- unlike
    # accounts.db.get_connection(), which hands out a fresh autocommitted
    # connection per statement (fine for the app's normal read/write
    # calls, wrong for this all-or-nothing reseed).
    with psycopg.connect(database_url, row_factory=dict_row, autocommit=False) as conn:
        # Transaction-scoped: released automatically on commit OR rollback,
        # so a crash mid-seed can't leave it locked forever. A concurrent
        # reseed just blocks here until this one finishes, instead of
        # interleaving deletes/inserts with it.
        conn.execute("select pg_advisory_xact_lock(%s)", (_SEED_LOCK_KEY,))

        # Child tables first (nothing else references them), accounts via
        # upsert last among the writes that matter for this race: an
        # upsert never makes the account_id row briefly disappear the way
        # delete-then-insert did, so a concurrent seed's payment_history/
        # events inserts can never hit a missing parent row.
        conn.execute("delete from events where account_id = any(%s)", (_ACCOUNT_IDS,))
        conn.execute("delete from escalations where account_id = any(%s)", (_ACCOUNT_IDS,))
        conn.execute("delete from disputes where account_id = any(%s)", (_ACCOUNT_IDS,))
        conn.execute("delete from promises where account_id = any(%s)", (_ACCOUNT_IDS,))
        conn.execute("delete from payment_history where account_id = any(%s)", (_ACCOUNT_IDS,))
        conn.execute("delete from payment_tokens where account_id = any(%s)", (_ACCOUNT_IDS,))

        for a in accounts:
            conn.execute(
                """
                insert into accounts (
                    account_id, borrower_name, business_name, phone_number, language_preference,
                    loan_type, principal_amount, emi_amount, tenure_months, months_remaining,
                    emi_due_date, nach_mandate_active, dispute_open, risk_tier, access_key
                ) values (
                    %(account_id)s, %(borrower_name)s, %(business_name)s, %(phone_number)s, %(language_preference)s,
                    %(loan_type)s, %(principal_amount)s, %(emi_amount)s, %(tenure_months)s, %(months_remaining)s,
                    %(emi_due_date)s, %(nach_mandate_active)s, %(dispute_open)s, %(risk_tier)s, %(access_key)s
                )
                on conflict (account_id) do update set
                    borrower_name = excluded.borrower_name,
                    business_name = excluded.business_name,
                    phone_number = excluded.phone_number,
                    language_preference = excluded.language_preference,
                    loan_type = excluded.loan_type,
                    principal_amount = excluded.principal_amount,
                    emi_amount = excluded.emi_amount,
                    tenure_months = excluded.tenure_months,
                    months_remaining = excluded.months_remaining,
                    emi_due_date = excluded.emi_due_date,
                    nach_mandate_active = excluded.nach_mandate_active,
                    dispute_open = excluded.dispute_open,
                    risk_tier = excluded.risk_tier,
                    access_key = excluded.access_key,
                    telegram_chat_id = null,
                    updated_at = now()
                """,
                a,
            )

        # Found live: dispute_open=True was set directly on the account row
        # for every seeded account that needed it, but nothing ever wrote a
        # matching disputes row -- so ops/flags.py's "disputed" flag had a
        # real dispute_open flag but no real reason behind it to surface
        # (see accounts/store.py's get_latest_open_dispute_reason, and the
        # ops dashboard's own new "Disputes" section, both of which read
        # this table directly). Keeps the demo internally consistent: a
        # disputed seeded account now has the same real backing row a
        # dispute opened through flag_dispute/the client dashboard's
        # "Contest" action would.
        for a in accounts:
            if a["dispute_open"]:
                conn.execute(
                    "insert into disputes (account_id, reason) values (%s, %s)",
                    (a["account_id"], "An incorrectly applied late fee -- please review and correct it."),
                )

        for account_id, payment_date, amount, on_time in payment_history:
            conn.execute(
                "insert into payment_history (account_id, payment_date, amount, on_time) values (%s, %s, %s, %s)",
                (account_id, payment_date, amount, on_time),
            )

        for account_id, made_on, promised_date, promised_amount, kept in promises:
            conn.execute(
                "insert into promises (account_id, made_on, promised_date, promised_amount, kept) values (%s, %s, %s, %s, %s)",
                (account_id, made_on, promised_date, promised_amount, kept),
            )

        conn.commit()

    print(f"seeded {len(accounts)} accounts, {len(payment_history)} payment records, {len(promises)} promises")
    print(f"today anchor: {today}")
    print("\naccount_id + access_key pairs for the demo:")
    for a in accounts:
        print(f"  {a['account_id']}  key={a['access_key']}  ({a['borrower_name']})")


if __name__ == "__main__":
    main()
