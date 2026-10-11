"""DocumentRetriever as a self-contained snapshot, with no database and no models: the embedder and the reranker are
small fakes, so these tests are about the retriever's own logic -- scoping, candidate selection, what the reranker is
shown, what comes back -- and about the bug that motivated the snapshot design (a chunk the snapshot does not hold must
never be a crash). Ranking quality on the real models is tests/test_retrieval_quality.py and eval/retrieval_benchmark.py.
"""

import numpy as np
import pytest

from businessflow.rag import retriever as retr
from businessflow.rag.retriever import DocumentRetriever

AXES = {"late": 0, "grace": 1, "nach": 2, "contract": 3}


def _vec(axis: int) -> list[float]:
    v = [0.0] * 4
    v[axis] = 1.0
    return v


def _row(chunk_id, text, headings, source, account="general", document_type="policy", axis=0):
    return {"id": chunk_id, "document_text": text, "source_document": source, "document_type": document_type,
            "account_id": account, "chunk_index": 0, "headings": headings, "embedding": _vec(axis)}


ROWS = [
    _row("g-late", "A flat late fee of 500 applies after the grace period (see grace_period.md).", "Late fee policy", "data/kb/late_fee_policy.md", axis=AXES["late"]),
    _row("g-grace", "Every EMI has three days of grace before any fee.", "Grace period policy", "data/kb/grace_period.md", axis=AXES["grace"]),
    _row("g-nach", "The auto debit mandate collects the EMI from the bank.", "NACH mandate troubleshooting", "data/kb/nach_mandate_troubleshooting.md", axis=AXES["nach"]),
    _row("a-loan", "Principal 250,000 and monthly EMI 12,500.", "Loan Details", "data/documents/BF-1/priya_loan_agreement.md", "BF-1", "loan_agreement", AXES["contract"]),
    _row("b-loan", "Principal 900,000 and monthly EMI 40,000.", "Loan Details", "data/documents/BF-2/arjun_loan_agreement.md", "BF-2", "loan_agreement", AXES["contract"]),
]


class FakeReranker:
    """Scores a (query, text) pair by how many query words the text contains, and remembers what it was shown."""

    def __init__(self):
        self.shown: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.shown.extend(pairs)
        return [float(sum(w in text.lower() for w in q.lower().split())) for q, text in pairs]


@pytest.fixture
def reranker(monkeypatch):
    fake = FakeReranker()
    monkeypatch.setattr(retr, "_load_reranker", lambda: fake)

    def embed_query(text):
        words = text.lower()
        for word, axis in AXES.items():
            if word in words:
                return np.asarray(_vec(axis), dtype=np.float32)
        return np.asarray(_vec(AXES["late"]), dtype=np.float32)

    monkeypatch.setattr(retr, "embed_query", embed_query)
    monkeypatch.setattr(retr.query_llm, "translate_to_english", lambda q: q)
    return fake


def test_a_chunk_the_snapshot_does_not_hold_can_never_crash_retrieval(reranker):
    # The old retriever kept text in a snapshot but ran the embedding search live against the table: a chunk ingested
    # after the snapshot was returned by the search, was missing from the snapshot, and raised ValueError. Here the
    # embedding search covers exactly the rows the snapshot was built from.
    snapshot = DocumentRetriever(rows=[r for r in ROWS if r["id"] != "g-nach"])

    results = snapshot.retrieve("auto debit mandate nach", top_k=5)

    assert results
    assert {r["text"] for r in results}.isdisjoint({"The auto debit mandate collects the EMI from the bank."})


def test_an_empty_snapshot_answers_with_nothing_rather_than_failing(reranker):
    assert DocumentRetriever(rows=[]).retrieve("anything at all") == []


def test_a_borrower_sees_general_documents_and_their_own_never_another_borrowers(reranker):
    snapshot = DocumentRetriever(rows=ROWS)

    own = snapshot.retrieve("principal emi", top_k=5, candidate_pool=5, account_id="BF-1")
    other = snapshot.retrieve("principal emi", top_k=5, candidate_pool=5, account_id="BF-2")
    nobody = snapshot.retrieve("principal emi", top_k=5, candidate_pool=5)

    assert any(r["account_id"] == "BF-1" for r in own) and not any(r["account_id"] == "BF-2" for r in own)
    assert any(r["account_id"] == "BF-2" for r in other) and not any(r["account_id"] == "BF-1" for r in other)
    assert {r["account_id"] for r in nobody} == {"general"}


def test_the_document_type_filter_is_exact(reranker):
    snapshot = DocumentRetriever(rows=ROWS)

    assert {r["document_type"] for r in snapshot.retrieve("principal emi", top_k=5, account_id="BF-1", document_type="loan_agreement")} == {"loan_agreement"}
    assert snapshot.retrieve("late fee", top_k=5, document_type="regulatory") == []


def test_the_candidate_pool_is_never_smaller_than_top_k(reranker):
    # candidate_pool=1 with top_k=3 used to be able to return fewer than three results from a corpus that had them.
    results = DocumentRetriever(rows=ROWS).retrieve("late fee grace nach", top_k=3, candidate_pool=1)

    assert len(results) == 3


def test_a_chunk_sharing_no_word_with_the_query_is_not_a_keyword_candidate(reranker):
    snapshot = DocumentRetriever(rows=ROWS)

    snapshot.retrieve("late fee", top_k=2, candidate_pool=1)

    shown_texts = {text for _, text in reranker.shown}
    assert not any("auto debit mandate" in t for t in shown_texts), "a zero-score BM25 chunk was handed to the reranker"


def test_headings_are_shown_to_the_search_but_the_text_that_comes_back_is_unchanged(reranker):
    with_headings = DocumentRetriever(rows=ROWS, use_headings=True)
    results = with_headings.retrieve("late fee", top_k=1)

    assert any(text.startswith("Late fee policy\n") for _, text in reranker.shown)
    assert results[0]["text"] == ROWS[0]["document_text"]
    assert results[0]["headings"] == "Late fee policy"

    reranker.shown.clear()
    DocumentRetriever(rows=ROWS, use_headings=False).retrieve("late fee", top_k=1)
    assert not any(text.startswith("Late fee policy\n") for _, text in reranker.shown)


def test_results_say_what_kind_of_score_they_carry(reranker):
    snapshot = DocumentRetriever(rows=ROWS)

    assert snapshot.retrieve("late fee", top_k=1)[0]["score_kind"] == "rerank"
    # a query with no Latin letters or digits skips BM25 and the English reranker and is ranked by embedding alone
    devanagari = snapshot.retrieve("शुल्क", top_k=1)
    assert devanagari[0]["score_kind"] == "embedding"


def test_follow_references_pulls_in_the_document_a_result_cites(reranker):
    snapshot = DocumentRetriever(rows=ROWS)

    results = snapshot.retrieve("late fee", top_k=1, follow_references=True)

    assert [r["headings"] for r in results] == ["Late fee policy", "Grace period policy"]
    assert results[1]["referenced_from"] is True


def test_a_reference_to_a_document_outside_the_borrowers_scope_is_not_followed(reranker):
    rows = [dict(r) for r in ROWS]
    rows[1] = _row("g-grace-private", rows[1]["document_text"], "Grace period policy", "data/kb/grace_period.md", "BF-9", "policy", AXES["grace"])

    results = DocumentRetriever(rows=rows).retrieve("late fee", top_k=1, follow_references=True)

    assert [r["headings"] for r in results] == ["Late fee policy"]


def test_the_old_padding_of_the_keyword_pool_with_zero_score_chunks_can_be_switched_back_on(reranker):
    # Only r-kw contains the word "zzz", but the keyword pool has two slots: the old behaviour filled the spare slot with
    # the next chunk in ingestion order (r-pad1), which has nothing to do with the query. The embedding stage prefers
    # r-emb (and then r-kw by tie), so r-pad1 can only reach the reranker through that padding.
    src = "data/kb/x.md"
    rows = [
        _row("r-kw", "unique keyword zzz appears here", "A", src, axis=AXES["nach"]),
        _row("r-pad1", "first unrelated chunk", "B", src, axis=AXES["late"]),
        _row("r-pad2", "second unrelated chunk", "C", src, axis=AXES["late"]),
        _row("r-emb", "chunk the embedding likes", "D", src, axis=AXES["grace"]),
    ]
    query = "zzz grace"

    DocumentRetriever(rows=rows).retrieve(query, top_k=2, candidate_pool=2)
    assert not any("first unrelated chunk" in text for _, text in reranker.shown)

    reranker.shown.clear()
    DocumentRetriever(rows=rows, keyword_overlap_required=False).retrieve(query, top_k=2, candidate_pool=2)
    assert any("first unrelated chunk" in text for _, text in reranker.shown)


def test_an_unknown_fusion_is_refused_loudly(reranker):
    with pytest.raises(ValueError, match="fusion"):
        DocumentRetriever(rows=ROWS, fusion="average")


def test_rank_fusion_changes_the_order_but_keeps_each_results_cross_encoder_score(reranker):
    # RRF mixes the cross-encoder's order with the embedding's; the score a result carries stays the cross-encoder's,
    # which is the number any relevance check reads.
    plain = DocumentRetriever(rows=ROWS).retrieve("late fee grace", top_k=3, candidate_pool=3)
    fused = DocumentRetriever(rows=ROWS, fusion="rrf").retrieve("late fee grace", top_k=3, candidate_pool=3)

    assert {r["headings"] for r in plain} == {r["headings"] for r in fused}
    assert all(r["score_kind"] == "rerank" for r in fused)
    scores = {r["headings"]: r["relevance_score"] for r in plain}
    assert all(r["relevance_score"] == scores[r["headings"]] for r in fused)
