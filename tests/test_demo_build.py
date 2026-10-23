"""The demo builder's own logic (scripts/build_demo.py, scripts/demo_profiles.py), without servers: the loan book it
uploads passes the real importer's checks, the statements it uploads are read by the real statement parser, and each
borrower's story comes out as written whatever day it runs on.
"""

from datetime import date, timedelta

import pytest

from businessflow.accounts.bank_statement import parse_statement
from businessflow.ops.loan_book import parse_loan_book
from scripts import build_demo as b
from scripts.demo_profiles import LOANS, apr_pct, by_ref, due_dates, emi, schedule

IDS = {loan.ref: f"BF-{2001 + i}" for i, loan in enumerate(LOANS)}
# A day early in a month, a day late in a month, and a day in another season: the stories must hold on each.
DAYS = [date(2026, 10, 23), date(2026, 11, 3), date(2026, 12, 19)]


def test_every_schedule_repays_the_loan_exactly_and_the_apr_includes_the_fee():
    for loan in LOANS:
        rows = schedule(loan)
        assert rows[-1]["closing"] == 0 and len(rows) == loan.tenure
        assert round(sum(r["principal"] for r in rows), 2) == loan.principal
        assert apr_pct(loan) > loan.rate_pct  # the processing fee makes the real cost higher than the rate


@pytest.mark.parametrize("today", DAYS)
def test_the_loan_book_passes_the_importer(today):
    rows = parse_loan_book(b._loan_book_csv(today - timedelta(days=b.GO_LIVE_DAYS_AGO)))
    assert [r.errors for r in rows] == [[] for _ in rows]
    assert {r.values["external_ref"] for r in rows} == {loan.ref for loan in LOANS if not loan.opened_here}
    assert all(r.values["interest_rate_pct"] == by_ref(r.values["external_ref"]).rate_pct for r in rows)


@pytest.mark.parametrize("today", DAYS)
def test_each_story_comes_out_as_written(today):
    go_live = today - timedelta(days=b.GO_LIVE_DAYS_AGO)
    credits = b._credits(IDS, go_live, today)
    mine = lambda ref: [c for c in credits if c["ref"] == ref]  # noqa: E731

    assert all(c["date"] < today or c["kind"] == "upi_name" for c in credits)
    # Meena: one auto-debit on each due date since go-live, never late.
    meena = by_ref(b.MEENA)
    assert [c["date"] for c in mine(b.MEENA)] == [d for d in due_dates(meena) if go_live <= d < today]
    # Imran: the EMI and the fee together, nine days late, then nothing for two cycles.
    imran = mine(b.IMRAN)
    assert imran[-1]["amount"] == emi(by_ref(b.IMRAN)) + 500 and imran[-1]["kind"] == "upi_phone"
    last_due = b._latest_due(by_ref(b.IMRAN), today)
    assert imran[-1]["date"] == b._previous_due(by_ref(b.IMRAN), last_due, 2) + timedelta(days=9)
    # Lakshmi: the only credit with just a name, and it is this cycle's EMI.
    [waiting] = [c for c in credits if c["kind"] == "upi_name"]
    assert waiting["ref"] == b.LAKSHMI and waiting["date"] >= b._latest_due(by_ref(b.LAKSHMI), today)
    # Sunita: last cycle paid six days late by UPI, this cycle nothing.
    sunita = mine(b.SUNITA)
    prev = b._previous_due(by_ref(b.SUNITA), b._latest_due(by_ref(b.SUNITA), today))
    assert sunita[-1]["date"] == prev + timedelta(days=6) and sunita[-1]["kind"] == "upi_phone"
    # Ramesh's repaid loan: every EMI since go-live, to the last one.
    wc = by_ref(b.RAMESH_WC)
    assert [c["date"] for c in mine(b.RAMESH_WC)] == [d for d in due_dates(wc) if go_live <= d < today]


@pytest.mark.parametrize("today", DAYS)
def test_the_statements_are_read_by_the_real_parser_with_every_credit(today):
    go_live = today - timedelta(days=b.GO_LIVE_DAYS_AGO)
    credits = b._credits(IDS, go_live, today)
    month_start = today.replace(day=1)
    in_month = [c for c in credits if month_start <= c["date"] <= today]
    text, noise = b._statement_csv(month_start, today, in_month)
    parsed = parse_statement(text)
    assert parsed.rejected == []
    assert sorted(line.amount for line in parsed.lines) == sorted([c["amount"] for c in in_month]
                                                                  + [a for t, a, s, _ in b._NOISE if s == "credit"][:len(noise)])
    for c in in_month:
        if c["kind"] == "nach":
            assert IDS[c["ref"]] in c["narration"]  # names the loan account: the matcher posts it without anyone
