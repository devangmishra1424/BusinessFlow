"""Tests for the report-generation pipeline (gather -> analyze -> write
-> accuracy-check, blueprint's own §13). verify.py and analyze.py are
pure logic, tested directly with no DB or LLM. gather.py needs real
Postgres. generate.py's full pipeline needs both real Postgres and a
real Groq call for the write/retry stage -- no mocking, same convention
as the rest of this project.
"""

import os

import pytest

from businessflow.reports.analyze import analyze_accounts_needing_attention
from businessflow.reports.verify import check_report_accuracy

_pg_skip = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)
_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
)


def _account(account_id, days_past_due, flags):
    return {
        "account_id": account_id, "borrower_name": "Test", "business_name": "Test Biz",
        "days_past_due": days_past_due, "dispute_open": any(f["label"] == "disputed" for f in flags),
        "broken_promise_count": 2 if any(f["label"] == "broken_promises" for f in flags) else 0,
        "flags": flags, "open_escalations": [],
    }


class TestCheckReportAccuracy:
    def test_a_report_grounded_entirely_in_the_facts_passes(self):
        facts = [_account("BF-1001", 3, []), _account("BF-1003", 20, [{"label": "overdue", "reason": "..."}])]
        report = "BF-1003 is 20 days overdue and needs attention. BF-1001 is current."

        failure = check_report_accuracy(report, facts)

        assert not failure

    def test_a_hallucinated_account_id_is_caught(self):
        facts = [_account("BF-1001", 3, [])]
        report = "BF-9999 needs urgent attention."

        failure = check_report_accuracy(report, facts)

        assert failure
        assert "BF-9999" in failure.describe()

    def test_a_day_count_not_matching_any_real_account_is_caught(self):
        facts = [_account("BF-1003", 20, [{"label": "overdue", "reason": "..."}])]
        report = "BF-1003 is 45 days overdue."  # real account, but the wrong number

        failure = check_report_accuracy(report, facts)

        assert failure
        assert 45 in failure.unmatched_day_counts

    def test_zero_flagged_accounts_and_no_claims_passes_cleanly(self):
        facts = [_account("BF-1001", 3, [])]
        report = "No accounts currently need attention."

        assert not check_report_accuracy(report, facts)


class TestAnalyzeAccountsNeedingAttention:
    def test_only_flagged_accounts_count_as_needing_attention(self):
        facts = [
            _account("BF-1001", 3, []),
            _account("BF-1002", 10, [{"label": "overdue", "reason": "..."}]),
        ]

        result = analyze_accounts_needing_attention(facts)

        assert result["total_accounts"] == 2
        assert result["accounts_needing_attention"] == 1
        assert [a["account_id"] for a in result["flagged_accounts"]] == ["BF-1002"]

    def test_a_disputed_account_ranks_above_a_merely_overdue_one_regardless_of_days(self):
        facts = [
            _account("BF-1002", 60, [{"label": "overdue", "reason": "..."}]),  # very overdue, but only 1 flag type
            _account("BF-1003", 5, [
                {"label": "disputed", "reason": "..."}, {"label": "overdue", "reason": "..."},
            ]),  # fewer days, but 2 distinct flag types
        ]

        result = analyze_accounts_needing_attention(facts)

        assert [a["account_id"] for a in result["flagged_accounts"]] == ["BF-1003", "BF-1002"]

    def test_among_equal_flag_counts_more_overdue_ranks_first(self):
        facts = [
            _account("BF-1002", 5, [{"label": "overdue", "reason": "..."}]),
            _account("BF-1004", 25, [{"label": "overdue", "reason": "..."}]),
        ]

        result = analyze_accounts_needing_attention(facts)

        assert [a["account_id"] for a in result["flagged_accounts"]] == ["BF-1004", "BF-1002"]

    def test_no_accounts_at_all_is_handled_cleanly(self):
        result = analyze_accounts_needing_attention([])
        assert result == {"total_accounts": 0, "accounts_needing_attention": 0, "flagged_accounts": []}


@_pg_skip
def test_gather_account_facts_matches_real_seeded_data(reseed_accounts):
    from businessflow.reports.gather import gather_account_facts

    facts = gather_account_facts(["BF-1001", "BF-1003"])
    by_id = {f["account_id"]: f for f in facts}

    assert by_id["BF-1001"]["flags"] == []
    assert {f["label"] for f in by_id["BF-1003"]["flags"]} == {"overdue", "disputed", "broken_promises"}


@_pg_skip
@_groq_skip
def test_generate_report_produces_a_verified_report_about_real_accounts(reseed_accounts):
    from businessflow.reports.generate import generate_report

    report = generate_report("Which accounts need attention today?")

    assert report.verified is True
    assert report.analysis["accounts_needing_attention"] >= 1  # BF-1003 always qualifies in the seed data
    assert len(report.text) > 0
