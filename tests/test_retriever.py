"""Tests that hybrid retrieval actually surfaces the right chunk for
realistic borrower questions -- not just that it runs without crashing.
Assumes scripts/seed_kb.py has already been run against the persistent
store this test session points at.
"""

import pytest

from businessflow.rag.retriever import DocumentRetriever


@pytest.fixture(scope="module")
def retriever():
    return DocumentRetriever()


@pytest.mark.parametrize(
    "query,acceptable_headings",
    [
        ("can I get a few more days to pay", ["Grace period"]),
        ("I think my late fee is wrong", ["Dispute handling"]),
        # The "2+ broken promises -> escalate" fact is stated in both
        # escalation_policy.md and restructuring_options.md's "What blocks
        # an automated offer" section -- either is a genuinely correct
        # answer, so both are accepted rather than assuming one canonical
        # document per query.
        ("what happens if I break a promise to pay twice", ["Escalation policy", "What blocks an automated offer"]),
        ("can you lower my monthly payment for a while", ["Restructuring options"]),
        # Pure Devanagari -- "can I get a few more days without a penalty."
        # Verified via eval/retrieval_benchmark.py that this specific query
        # regressed to rank 3 (wrong chunk at #1) before the no-BM25-tokens
        # rerank-skip fix in retriever.py, since the English-only reranker
        # was demoting a chunk the multilingual embedding stage had already
        # ranked correctly.
        ("क्या मुझे कुछ और दिन मिल सकते हैं बिना जुर्माने के", ["Grace period"]),
    ],
)
def test_retrieves_the_right_document(retriever, query, acceptable_headings):
    results = retriever.retrieve(query, top_k=1)
    assert results, f"no results at all for {query!r}"
    assert any(h in results[0]["headings"] for h in acceptable_headings)


def test_general_docs_are_visible_regardless_of_account_id():
    retriever = DocumentRetriever()
    results = retriever.retrieve("can I get a few more days to pay", top_k=1, account_id="BF-1001")
    assert results
    assert results[0]["account_id"] == "general"


def test_a_query_with_no_bm25_tokens_still_returns_well_formed_results(retriever):
    # An empty _tokenize(query) (punctuation-only, or pure Devanagari --
    # the tokenizer regex is [a-z0-9]+) makes BM25Okapi.get_scores([])
    # return an all-zero-tie array; sorted() on ties is stable, so
    # without the empty-token guard in retriever.py this silently became
    # "whichever chunks were ingested first" rather than a real signal.
    # This doesn't assert a specific ranking (that's eval/retrieval_
    # benchmark.py's job) -- just that the guard doesn't crash or return
    # malformed output on the exact input that triggers it.
    ranked = retriever.retrieve("??? !!! ...", top_k=2)
    assert isinstance(ranked, list)
    assert len(ranked) <= 2
    assert all("headings" in r and "text" in r for r in ranked)


def test_one_borrowers_documents_are_not_visible_to_another():
    import os
    import tempfile

    from businessflow.rag.ingest import ingest_document
    from businessflow.rag.store import get_collection

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Priya's loan agreement\n\nA special one-time waiver clause applies to this loan.\n")
        temp_path = f.name

    try:
        ingest_document(temp_path, document_type="loan_agreement", account_id="BF-1001")
        retriever = DocumentRetriever()

        own_results = retriever.retrieve("special waiver clause", top_k=1, account_id="BF-1001")
        assert own_results and "waiver" in own_results[0]["text"].lower()

        other_results = retriever.retrieve("special waiver clause", top_k=1, account_id="BF-1002")
        assert not any("waiver" in r["text"].lower() for r in other_results)
    finally:
        # Clean up the persistent store, not just the temp file -- this
        # test would otherwise leave permanent test data mixed in with
        # real ingested documents.
        get_collection().delete(where={"source_document": temp_path})
        os.unlink(temp_path)
