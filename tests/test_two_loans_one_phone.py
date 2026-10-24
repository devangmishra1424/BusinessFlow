"""One borrower, several loans: two accounts may share a phone number (migrations/2026-10-28_phone_not_unique.sql).
A live database created before multi-loan refused the second loan with a unique-constraint error; schema.sql never had it.
"""

from datetime import date, timedelta

from businessflow.accounts import store
from businessflow.accounts.db import get_connection


def _open(ref: str, phone: str):
    return store.create_account("Test Borrower", "Test Shop", phone, "en", "Working Capital Loan", 100_000, 9_500, 12,
                                date.today() + timedelta(days=10), True, "low", external_ref=ref)[0]


def test_a_second_loan_on_the_same_phone_is_opened_and_both_stay_separate(reseed_accounts):
    conn = get_connection()
    conn.execute("delete from accounts where external_ref like 'PHONE-TEST-%'")
    try:
        a, b = _open("PHONE-TEST-1", "+919812349911"), _open("PHONE-TEST-2", "+919812349911")
        assert a.account_id != b.account_id
        assert store.accounts_by_external_ref(["PHONE-TEST-1", "PHONE-TEST-2"]) == {
            "PHONE-TEST-1": a.account_id, "PHONE-TEST-2": b.account_id}
    finally:
        conn.execute("delete from accounts where external_ref like 'PHONE-TEST-%'")


def test_the_migration_is_idempotent_and_leaves_no_phone_uniqueness(reseed_accounts):
    import pathlib

    conn = get_connection()
    sql = (pathlib.Path(__file__).resolve().parents[1] / "migrations" / "2026-10-28_phone_not_unique.sql").read_text(encoding="utf-8")
    conn.execute(sql)
    conn.execute(sql)  # twice: a database that already lost the rule is not an error
    unique = [r["conname"] for r in conn.execute(
        "select conname from pg_constraint where conrelid = 'accounts'::regclass and contype = 'u'").fetchall()]
    assert "accounts_phone_number_key" not in unique
