"""Thin test for the check_policy MCP tool wrapper -- retrieval quality
itself is exhaustively covered by test_retriever.py; this just confirms
the tool correctly shapes DocumentRetriever's output as {query, results}.
Assumes scripts/seed_kb.py has already been run against the persistent
store this test session points at (same assumption as test_retriever.py).
"""

from businessflow.tools.policy_tools import check_policy


def test_check_policy_returns_query_and_grounded_results():
    result = check_policy(query="can I get a few more days to pay")

    assert result["query"] == "can I get a few more days to pay"
    assert len(result["results"]) > 0
    assert all("text" in r and "source_document" in r for r in result["results"])


def test_check_policy_with_no_matching_account_scoped_docs_still_returns_general_results():
    # account_id given but this borrower has no documents of their own
    # uploaded -- general policy docs must still come back, not an empty
    # result silently masking a real "found nothing" case.
    result = check_policy(query="what is the grace period", account_id="BF-1001")

    assert len(result["results"]) > 0
