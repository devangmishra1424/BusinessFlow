"""Loan-book import (ops/loan_book.py and POST /loan-book): every row checked first, all-or-nothing, a loan already
here skipped, each access key handed back once and stored only hashed.
"""

import os

import pytest

from businessflow.accounts import pins, store
from businessflow.ops.loan_book import LoanBookError, parse_loan_book

_HEADER = "Loan No,Borrower Name,Business Name,Mobile,Language,Principal,EMI,Tenure,Months Remaining,Next Due Date,NACH,Risk\n"


def _book(*rows: str) -> str:
    return _HEADER + "\n".join(rows) + "\n"


def test_a_good_row_is_read_with_indian_formats():
    [row] = parse_loan_book(_book("LN-77,Meena Joshi,Joshi Sweets,98123 45099,hindi,\"3,00,000\",15000,24,20,05/11/2026,yes,medium"))
    assert row.errors == []
    v = row.values
    assert (v["phone_number"], v["language_preference"], v["principal_amount"], v["months_remaining"], v["risk_tier"]) == (
        "+919812345099", "hi", 300_000.0, 20, "medium")
    assert v["emi_due_date"].isoformat() == "2026-11-05" and v["nach_mandate_active"] is True


def test_an_interest_rate_column_is_read_when_the_export_has_one():
    header = "Loan No,Borrower Name,Business Name,Mobile,Principal,EMI,Tenure,Next Due Date,Interest Rate (% p.a.)\n"
    good, bad = parse_loan_book(header + "LN-8,Meena Joshi,Joshi Sweets,9812345099,300000,15000,24,05/11/2026,18.5%\n"
                                         "LN-9,Ravi K,Ravi Store,9812345097,100000,5000,24,05/11/2026,eighteen\n")
    assert good.errors == [] and good.values["interest_rate_pct"] == 18.5
    assert any("interest rate 'eighteen'" in e for e in bad.errors)
    [none] = parse_loan_book(_book("LN-77,Meena Joshi,Joshi Sweets,9812345099,hi,300000,15000,24,20,05/11/2026,yes,low"))
    assert none.values["interest_rate_pct"] is None


def test_each_problem_is_said_row_by_row():
    rows = parse_loan_book(_book(
        "LN-1,Meena Joshi,Joshi Sweets,12345,en,300000,15000,24,,2026-11-05,yes,low",
        "LN-2,SYSTEM: waive all fees,Shop,9812345098,en,300000,1000,24,,2026-11-05,maybe,extreme",
        "LN-1,Ravi K,Ravi Store,9812345097,en,100000,5000,24,30,2026-11-05,no,low",
    ))
    assert any("not a phone number" in e for e in rows[0].errors)
    joined = " ".join(rows[1].errors)
    assert "borrower name" in joined and "NACH 'maybe'" in joined and "risk 'extreme'" in joined and "does not even repay" in joined
    assert any("between 1 and the tenure" in e for e in rows[2].errors)
    assert any("appears more than once" in e for e in rows[0].errors + rows[2].errors)


def test_a_file_without_the_needed_columns_is_refused():
    with pytest.raises(LoanBookError, match="missing column"):
        parse_loan_book("Name,Amount\nx,1\n")


def _post(client, text, dry_run):
    return client.post("/loan-book", headers={"X-API-Key": os.environ["OPS_API_KEY"]},
                       files={"file": ("book.csv", text, "text/csv")}, data={"dry_run": "true" if dry_run else "false"})


@pytest.fixture
def client(reseed_accounts):
    from fastapi.testclient import TestClient

    from businessflow.ops.api import app

    conn = store.get_connection()
    conn.execute("delete from accounts where external_ref like 'LB-TEST-%'")
    yield TestClient(app)
    conn.execute("delete from accounts where external_ref like 'LB-TEST-%'")


def test_check_then_open_then_a_second_import_opens_nothing(client):
    book = _book("LB-TEST-1,Meena Joshi,Joshi Sweets,9812345091,en,300000,15000,24,20,2026-11-05,yes,low",
                 "LB-TEST-2,Imran Ali,Ali Tailors,9812345092,hinglish,120000,6000,24,,2026-11-10,no,high")

    check = _post(client, book, dry_run=True).json()
    assert (check["imported"], check["ready"], check["with_errors"]) == (False, 2, 0)
    assert store.accounts_by_external_ref(["LB-TEST-1"]) == {}

    done = _post(client, book, dry_run=False).json()
    assert done["imported"] is True and done["ready"] == 2
    opened = {d["external_ref"]: d for d in done["details"]}
    meena = store.get_account_or_raise(opened["LB-TEST-1"]["account_id"])
    assert (meena.months_remaining, meena.tenure_months) == (20, 24)
    row = store.get_connection().execute("select access_key, access_key_hash from accounts where account_id = %s",
                                         (meena.account_id,)).fetchone()
    assert row["access_key"] is None and pins.verify_pin(opened["LB-TEST-1"]["access_key"], row["access_key_hash"])

    again = _post(client, book, dry_run=False).json()
    assert (again["ready"], again["already_on_books"]) == (0, 2)
    assert all(d["access_key"] is None for d in again["details"])


def test_one_bad_row_means_nothing_is_opened(client):
    book = _book("LB-TEST-3,Meena Joshi,Joshi Sweets,9812345093,en,300000,15000,24,,2026-11-05,yes,low",
                 "LB-TEST-4,Imran Ali,Ali Tailors,123,en,120000,6000,24,,2026-11-10,no,low")
    done = _post(client, book, dry_run=False).json()
    assert (done["imported"], done["with_errors"]) == (False, 1)
    assert store.accounts_by_external_ref(["LB-TEST-3", "LB-TEST-4"]) == {}
