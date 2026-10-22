"""Bank-statement reconciliation (cash application). Reading and matching are pure (accounts/bank_statement.py); posting
runs against real Postgres through accounts/store.py and the ops API, same split as test_promise_plans.py.
"""

from datetime import date, timedelta

import pytest

from businessflow.accounts import store
from businessflow.accounts.bank_statement import (
    Candidate,
    StatementError,
    StatementLine,
    claim_reference_tokens,
    match_line,
    parse_statement,
)

# --- reading a statement --------------------------------------------------------------------------------------------

_HDFC_STYLE = """Account Statement,,,,,
Account No,XXXX1234,,,,
Date,Narration,Chq./Ref.No.,Value Dt,Withdrawal Amt.,Deposit Amt.
21/10/26,UPI-PRIYA SHARMA-9812345001@ybl-BF-1001,UTR1234567890,21/10/26,,"12,500.00"
21/10/26,NEFT DR-RENT OCTOBER,N123456,21/10/26,"40,000.00",
22/10/26,,R9,22/10/26,,abc
"""


def test_a_bank_export_with_lines_above_the_header_is_read():
    parsed = parse_statement(_HDFC_STYLE)

    [line] = parsed.lines
    assert (line.line_date, line.amount, line.reference) == (date(2026, 10, 21), 12_500.0, "UTR1234567890")
    assert parsed.debits_skipped == 1
    assert parsed.rejected == [{"row": 3, "reason": "could not convert string to float: 'abc'"}]


def test_a_signed_amount_column_keeps_only_credits():
    parsed = parse_statement("Txn Date,Description,Amount,Type\n2026-10-20,UPI BF-1002,22000,CR\n2026-10-20,Charges,-118,DR\n")
    assert [(l.amount, l.narration) for l in parsed.lines] == [(22_000.0, "UPI BF-1002")]
    assert parsed.debits_skipped == 1


@pytest.mark.parametrize("text", ["", "Name,Balance\nx,1\n"])
def test_a_file_without_a_date_and_amount_header_is_refused(text):
    with pytest.raises(StatementError):
        parse_statement(text)


def test_the_same_line_twice_in_one_file_stays_two_lines():
    parsed = parse_statement("Date,Narration,Credit\n01/10/2026,CASH DEPOSIT,500\n01/10/2026,CASH DEPOSIT,500\n")
    a, b = parsed.hashes()
    assert a != b and b.startswith(a)


# --- matching -------------------------------------------------------------------------------------------------------

_PRIYA = Candidate("BF-1001", "Priya Sharma", "Cotton Threads Boutique", "+919812345001", 12_500.0)
_ARJUN = Candidate("BF-1002", "Arjun Mehta", "Mehta Hardware & Electricals", "+919812345002", 22_500.0,
                   claim_refs={"412345678901": 77})
_CANDIDATES = [_PRIYA, _ARJUN]


def _line(narration: str, amount: float = 12_500.0, reference: str = "") -> StatementLine:
    return StatementLine(1, date(2026, 10, 21), amount, narration, reference)


def test_the_account_number_in_the_narration_posts_it():
    m = match_line(_line("UPI/BF1001/payment"), _CANDIDATES)
    assert (m.status, m.account_id) == ("auto", "BF-1001")


def test_the_reference_from_an_open_payment_claim_posts_it_and_settles_the_claim():
    m = match_line(_line("NEFT CR HDFC0001", 22_500, reference="412345678901"), _CANDIDATES)
    assert (m.status, m.account_id, m.dispute_id) == ("auto", "BF-1002", 77)


def test_the_phone_number_in_the_narration_posts_it():
    m = match_line(_line("UPI-9812345002@okaxis-PAYMENT"), _CANDIDATES)
    assert (m.status, m.account_id) == ("auto", "BF-1002")


def test_a_name_alone_is_only_ever_a_suggestion():
    m = match_line(_line("IMPS PRIYA SHARMA"), _CANDIDATES)
    assert (m.status, m.account_id) == ("suggested", "BF-1001")
    assert "amount equals what they owe now" in m.reason


def test_two_accounts_named_by_strong_identifiers_is_left_to_a_person():
    m = match_line(_line("BF-1001 for BF-1002"), _CANDIDATES)
    assert m.status == "suggested" and "more than one account" in m.reason


def test_a_line_naming_nobody_is_unmatched():
    assert match_line(_line("CASH DEPOSIT BRANCH 0042"), _CANDIDATES).status == "unmatched"


def test_claim_references_are_reference_like_tokens_only():
    assert claim_reference_tokens("Paid by UPI on 3rd, UTR 412345678901, ref abc") == {"412345678901"}


# --- posting, against real Postgres ---------------------------------------------------------------------------------

@pytest.fixture
def clean_bank_lines(reseed_accounts):
    store.get_connection().execute("delete from bank_lines")
    yield


def _statement(*rows: str) -> object:
    return parse_statement("Date,Narration,Reference,Credit\n" + "\n".join(rows) + "\n")


def test_a_strong_match_is_posted_once_even_when_the_statement_is_uploaded_twice(clean_bank_lines):
    day = (store.current_date() - timedelta(days=1)).strftime("%d/%m/%Y")
    before = len(store.get_account_or_raise("BF-1001").payment_history)
    statement = _statement(f"{day},UPI BF-1001 EMI,R-ONE-1,1000")

    first = store.import_bank_statement("oct.csv", statement)
    second = store.import_bank_statement("oct.csv", statement)

    assert (first["posted"], second["posted"], second["already_uploaded"]) == (1, 0, 1)
    account = store.get_account_or_raise("BF-1001")
    assert len(account.payment_history) == before + 1
    [line] = store.list_bank_lines("done")
    assert line["status"] == "posted" and line["payment_id"] is not None and line["decided_by"] == "bank statement (automatic)"


def test_a_line_that_settles_a_payment_claim_closes_the_claim(clean_bank_lines):
    claim = store.open_dispute("BF-1002", "Paid 1,000 by NEFT, UTR 998877665544", kind="payment_claim")
    day = store.current_date().strftime("%d/%m/%Y")

    result = store.import_bank_statement("oct.csv", _statement(f"{day},NEFT CR SOMEBANK,998877665544,1000"))

    assert result["posted"] == 1 and result["lines"][0]["dispute_id"] == claim["dispute_id"]
    assert store.get_dispute(claim["dispute_id"])["status"] == "resolved"
    assert store.get_account_or_raise("BF-1002").dispute_open is False


def test_a_strong_match_already_on_record_waits_instead_of_posting_twice(clean_bank_lines):
    today = store.current_date()
    store.record_payment("BF-1001", 1_234, payment_date=today, apply_extra_to_next=False)

    result = store.import_bank_statement("oct.csv", _statement(f"{today.strftime('%d/%m/%Y')},UPI BF-1001,R-DUP-1,1234"))

    [line] = result["lines"]
    assert (result["posted"], line["status"]) == (0, "suggested")
    assert "already on record" in line["match_reason"]


def test_a_person_posts_a_suggestion_once_and_a_second_click_is_refused(clean_bank_lines):
    day = store.current_date().strftime("%d/%m/%Y")
    [line] = store.import_bank_statement("oct.csv", _statement(f"{day},IMPS PRIYA SHARMA,R-NAME-1,700"))["lines"]
    assert (line["status"], line["account_id"]) == ("suggested", "BF-1001")

    posted = store.post_bank_line(line["id"], "BF-1001", "Asha")
    assert posted["line"]["status"] == "posted" and posted["line"]["decided_by"] == "Asha"
    with pytest.raises(store.BankLineAlreadyDecidedError):
        store.post_bank_line(line["id"], "BF-1001", "Asha")


def test_ignoring_a_line_keeps_who_and_why(clean_bank_lines):
    day = store.current_date().strftime("%d/%m/%Y")
    [line] = store.import_bank_statement("oct.csv", _statement(f"{day},INTEREST CREDIT,R-INT-1,42"))["lines"]
    assert line["status"] == "unmatched"

    ignored = store.ignore_bank_line(line["id"], "Asha", "Bank interest, not a loan payment")

    assert (ignored["status"], ignored["decided_by"], ignored["decided_note"]) == ("ignored", "Asha", "Bank interest, not a loan payment")
    assert store.list_bank_lines("open") == []
    with pytest.raises(store.BankLineAlreadyDecidedError):
        store.ignore_bank_line(line["id"], "Asha", "again")


def test_the_ops_endpoints_upload_post_and_refuse_a_bad_file(clean_bank_lines):
    import os

    from fastapi.testclient import TestClient

    from businessflow.ops.api import app

    client = TestClient(app)
    headers = {"X-API-Key": os.environ["OPS_API_KEY"]}
    day = store.current_date().strftime("%d/%m/%Y")
    csv_text = f"Date,Narration,Reference,Credit\n{day},UPI BF-1003,R-API-1,900\n{day},IMPS ARJUN MEHTA,R-API-2,800\n"

    r = client.post("/bank-statements", headers=headers, files={"file": ("oct.csv", csv_text, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["posted"], body["suggested"]) == (1, 1)

    [waiting] = client.get("/bank-lines", headers=headers).json()
    r = client.post(f"/bank-lines/{waiting['id']}/post", headers=headers, json={"account_id": "BF-1002", "decided_by": "Asha"})
    assert r.status_code == 200 and r.json()["status"] == "posted"
    r = client.post(f"/bank-lines/{waiting['id']}/post", headers=headers, json={"account_id": "BF-1002", "decided_by": "Asha"})
    assert r.status_code == 409

    r = client.post("/bank-statements", headers=headers, files={"file": ("x.csv", "Name,Balance\nx,1\n", "text/csv")})
    assert r.status_code == 400
