"""A regression gate on the retrieval that ships, using the benchmark in eval/retrieval_benchmark.py: real chunking, real
embeddings, the real cross-encoder, no database. It fails if a change to the retriever, the chunker, the embedding model
or the reranker makes real questions stop finding their answers, or lets one borrower's document reach another.

The floors are the benchmark's measured numbers for the shipped setting minus roughly one question's worth of room (see
eval/results/retrieval_benchmark.md for the numbers and their intervals): tight enough to catch a regression, loose
enough that a library upgrade that moves one borderline question does not fail the build. Raise them when the retriever
genuinely improves; never lower them to make a failing change pass.
"""

import inspect

import pytest

from businessflow.rag import ingest
from businessflow.rag.retriever import DocumentRetriever
from businessflow.tools import policy_tools
from eval import retrieval_benchmark as rb

# measured for the shipped setting on 2026-10-11: dev 0.76, held-out 0.94, loan 0.93 (Recall@2), isolation 0
FLOORS = {"dev": 0.68, "heldout": 0.85, "loan": 0.85}


def test_the_benchmarks_shipped_setting_is_what_production_runs():
    shipped = next(s for s in rb.SETTINGS if s.name == rb.SHIPPED)
    defaults = inspect.signature(DocumentRetriever.__init__).parameters

    assert shipped.pool == policy_tools._CANDIDATE_POOL
    assert shipped.headings_embed == ingest.EMBED_WITH_HEADINGS
    assert shipped.headings_search == defaults["use_headings"].default
    assert shipped.fusion == defaults["fusion"].default
    assert shipped.keyword_overlap_required == defaults["keyword_overlap_required"].default
    assert shipped.translate is True  # the retriever translates Hindi/Hinglish questions; there is no switch for it


@pytest.fixture(scope="module")
def shipped_results():
    rb.install_translator(rb.Translator(refresh=False))
    corpus = rb.build_corpus()
    shipped = next(s for s in rb.SETTINGS if s.name == rb.SHIPPED)
    dev = rb.load_fixture("dev_queries.json")["queries"]
    held = rb.load_fixture("heldout_queries.json")["queries"]
    negatives = rb.load_fixture("negative_queries.json")["queries"]
    return rb.evaluate(corpus, [shipped], dev, held, rb.loan_questions(corpus), negatives)[rb.SHIPPED]


@pytest.mark.parametrize("question_set", ["dev", "heldout", "loan"])
def test_the_deployed_view_holds_its_floor(shipped_results, question_set):
    recall_at_2 = shipped_results[question_set]["recall@2"]["mean"]
    assert recall_at_2 >= FLOORS[question_set], (
        f"Recall@2 on the {question_set} questions fell to {recall_at_2:.2f} (floor {FLOORS[question_set]}); "
        f"run `python -m eval.retrieval_benchmark` to see which questions moved"
    )


def test_no_borrower_ever_receives_another_borrowers_document(shipped_results):
    assert shipped_results["loan_isolation_leaks"] == 0


def test_the_top_score_still_separates_answerable_from_unanswerable_questions(shipped_results):
    assert shipped_results["answerable_vs_unanswerable"]["auc"] >= 0.80
