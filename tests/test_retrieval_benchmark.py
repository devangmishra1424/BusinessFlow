"""The retrieval benchmark's own arithmetic and fixtures, checked with known answers (no models, no network), the way
tests/test_tts_benchmark.py checks the voice benchmark: a benchmark nobody has checked is just a number generator."""

import json
import random

import pytest

from eval import retrieval_benchmark as rb
from eval.stats import bootstrap_ci, cluster_bootstrap_ci


def test_metric_values_read_recall_and_mrr_off_the_ranks():
    ranks = [1, 2, 3, 5, 6, None]
    assert rb.metric_values(ranks, "recall@1") == [1, 0, 0, 0, 0, 0]
    assert rb.metric_values(ranks, "recall@2") == [1, 1, 0, 0, 0, 0]
    assert rb.metric_values(ranks, "recall@5") == [1, 1, 1, 1, 0, 0]
    assert rb.metric_values(ranks, "mrr") == pytest.approx([1, 1 / 2, 1 / 3, 1 / 5, 1 / 6, 0])


def test_a_cluster_bootstrap_is_wider_than_a_plain_one_when_repeats_are_perfectly_correlated():
    # 6 distinct questions, each asked 9 times with the same outcome (as the loan set does across borrowers)
    outcomes = [1, 1, 1, 0, 1, 0]
    values = [o for o in outcomes for _ in range(9)]
    clusters = [f"q{i}" for i, _ in enumerate(outcomes) for _ in range(9)]

    _, plain_lo, plain_hi = bootstrap_ci(values)
    mean, lo, hi = cluster_bootstrap_ci(values, clusters)

    assert mean == pytest.approx(4 / 6)
    assert (hi - lo) > 2 * (plain_hi - plain_lo), "treating 54 repeats as 54 questions makes the interval far too narrow"


def test_without_clusters_the_cluster_bootstrap_agrees_with_the_plain_one():
    values = [random.Random(3).random() for _ in range(40)]
    clusters = [str(i) for i in range(40)]  # every value its own cluster
    plain = bootstrap_ci(values, seed=1)
    clustered = cluster_bootstrap_ci(values, clusters, seed=1)
    assert plain[0] == pytest.approx(clustered[0])
    assert abs((plain[2] - plain[1]) - (clustered[2] - clustered[1])) < 0.05


def test_the_cluster_bootstrap_of_nothing_is_nan():
    import math

    assert all(math.isnan(x) for x in cluster_bootstrap_ci([], []))


def _interval(diff, lo, hi):
    return {"diff": diff, "lo": lo, "hi": hi, "wins": 0, "losses": 0}


def _paired(**overrides):
    base = {s: {m: _interval(0.0, -0.05, 0.05) for m in rb.METRICS} for s in rb.SETS}
    for key, interval in overrides.items():
        set_name, metric = key.split("__")
        base[set_name][metric] = interval
    return base


def test_the_decision_rule_ships_only_a_change_that_helps_somewhere_and_hurts_nowhere():
    assert rb.verdict(_paired()).startswith("no measurable difference")
    assert rb.verdict(_paired(loan__mrr=_interval(0.04, 0.01, 0.07))) == "helps"
    assert rb.verdict(_paired(heldout__mrr=_interval(-0.06, -0.10, -0.01))).startswith("hurts")
    assert rb.verdict(_paired(loan__mrr=_interval(0.04, 0.01, 0.07), heldout__mrr=_interval(-0.06, -0.10, -0.01))).startswith("mixed")


def test_an_interval_touching_zero_is_not_a_finding():
    assert rb.verdict(_paired(dev__mrr=_interval(0.05, 0.0, 0.1))).startswith("no measurable difference")


def test_paired_difference_counts_wins_and_losses_on_the_same_questions():
    better = [1, 1, 2, None]
    worse = [1, 3, 5, 1]
    result = rb.paired_difference(better, worse, "recall@2")
    # recall@2: better = 1,1,1,0 ; worse = 1,0,0,1 -> diffs 0,1,1,-1
    assert result["wins"] == 2 and result["losses"] == 1
    assert result["diff"] == pytest.approx(0.25)


def test_auc_is_one_for_perfect_separation_and_half_for_none():
    assert rb.auc([5.0, 6.0], [1.0, 2.0]) == 1.0
    assert rb.auc([1.0, 2.0], [5.0, 6.0]) == 0.0
    assert rb.auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_every_setting_names_a_parent_that_exists_and_the_shipped_one_is_among_them():
    names = [s.name for s in rb.SETTINGS]
    assert len(names) == len(set(names))
    assert rb.SHIPPED in names
    for s in rb.SETTINGS:
        assert s.parent is None or s.parent in names
    assert sum(1 for s in rb.SETTINGS if s.parent is None) == 1


# ---------------------------------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------------------------------
def _kb_headings():
    from businessflow.rag.ingest import parse_and_chunk

    return [r.headings for path in sorted(rb.KB_DIR.glob("*.md")) for r in parse_and_chunk(str(path))]


@pytest.mark.parametrize("fixture", ["dev_queries.json", "heldout_queries.json"])
def test_every_acceptable_heading_is_a_real_heading_in_the_knowledge_base(fixture):
    headings = _kb_headings()
    for q in rb.load_fixture(fixture)["queries"]:
        assert q["acceptable_headings"], q["text"]
        for wanted in q["acceptable_headings"]:
            assert any(wanted in h for h in headings), f"{wanted!r} (for {q['text']!r}) matches no chunk heading"


def test_no_question_appears_in_two_sets():
    dev = {q["text"] for q in rb.load_fixture("dev_queries.json")["queries"]}
    held = {q["text"] for q in rb.load_fixture("heldout_queries.json")["queries"]}
    neg = {q["text"] for q in rb.load_fixture("negative_queries.json")["queries"]}
    assert not (dev & held) and not (dev & neg) and not (held & neg)
    assert len(dev) == 38 and len(held) == 47 and len(neg) == 30


def test_the_set_covers_every_style_and_language():
    for name in ("dev_queries.json", "heldout_queries.json"):
        queries = rb.load_fixture(name)["queries"]
        assert {q["language"] for q in queries} == {"en", "hinglish", "hi"}
        assert {"clean", "hinglish", "devanagari"} <= {q["style"] for q in queries}


def test_every_question_that_needs_translating_has_a_frozen_translation():
    from businessflow.rag import query_llm

    table = json.loads((rb.FIXTURES / "translations.json").read_text(encoding="utf-8"))
    texts = [q["text"] for n in ("dev_queries.json", "heldout_queries.json", "negative_queries.json") for q in rb.load_fixture(n)["queries"]]
    texts += [p for _, phrasings, _ in rb.FACTS for p in phrasings]
    missing = [t for t in texts if query_llm.needs_translation(t) and t not in table]
    assert not missing, f"run `python -m eval.retrieval_benchmark --refresh-translations`; no frozen translation for: {missing}"


def test_the_loan_questions_have_ground_truth_in_every_borrowers_own_agreement():
    # build_corpus() asserts this too; here it runs without the embedding models by rendering the agreements directly
    from datetime import date

    from scripts import seed_loan_agreements as seed

    accounts = rb.load_fixture("loan_accounts.json")["accounts"]
    template_by_id = {row[0]: row for row in seed._ACCOUNTS}
    assert len(accounts) == 9
    for account in accounts:
        t = template_by_id[account["account_id"]]
        text = seed._build_document(
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[10], float(account["principal"]), float(account["emi"]),
            account["tenure_months"], account["months_remaining"], account["loan_type"], date.fromisoformat(account["emi_due_date"]),
        )
        for fact, answer in rb._agreement_facts(account, t).items():
            assert answer.lower() in text.lower(), f"{account['account_id']} {fact}={answer!r}"
