"""Tests for retriever.py's cross-reference detection -- pure logic, no
DB needed (unlike test_retriever.py's real-Postgres tests, gated behind
DATABASE_URL). See retrieve()'s own follow_references docstring for the
real gap this closes: a query at the production top_k=1 default can
return a chunk whose own text says "(see X.md)" with no actual content
from X ever reaching the model.
"""

from businessflow.rag.retriever import _derive_title, _find_referenced_documents


def test_derive_title_from_a_policy_doc_filename():
    assert _derive_title("data/kb/grace_period.md") == "grace period"
    assert _derive_title("data/kb/late_fee_policy.md") == "late fee policy"


def test_derive_title_ignores_a_single_word_filename():
    # _MIN_TITLE_WORDS=2 -- a single generic word (e.g. a hypothetical
    # "faq.md") would substring-match almost anything, so it's excluded
    # rather than trusted as a real title.
    assert _derive_title("data/kb/faq.md") is None


def test_derive_title_from_a_real_uploaded_contract_filename():
    # Confirmed live: a real uploaded loan agreement's own chunk_index=0
    # heading is just letterhead (the lender's company name), not a
    # usable title -- filenames are reliable where headings aren't.
    assert _derive_title("data/documents/BF-1005/sunita_patil_loan_agreement.pdf") == "sunita patil loan agreement"


def test_find_referenced_documents_matches_the_explicit_see_convention():
    known = {"data/kb/grace_period.md": "grace period"}
    text = "An EMI paid after the 3-day grace period (see grace_period.md) is charged a late fee."

    found = _find_referenced_documents(text, known, exclude="data/kb/late_fee_policy.md")

    assert found == {"data/kb/grace_period.md"}


def test_find_referenced_documents_matches_a_title_mentioned_in_prose():
    # No "(see X)" citation at all -- this is the broader heuristic a real
    # uploaded contract (which won't use this project's own markdown
    # convention) needs: does the text just mention another document's
    # name.
    known = {"data/kb/late_fee_policy.md": "late fee policy"}
    text = "As per the late fee policy, a flat charge applies after the grace period."

    found = _find_referenced_documents(text, known, exclude="data/kb/grace_period.md")

    assert found == {"data/kb/late_fee_policy.md"}


def test_find_referenced_documents_never_references_its_own_document():
    known = {"data/kb/grace_period.md": "grace period"}
    text = "This IS the grace period policy -- every EMI has a 3-day grace period."

    found = _find_referenced_documents(text, known, exclude="data/kb/grace_period.md")

    assert found == set()


def test_find_referenced_documents_finds_multiple_real_references_in_one_chunk():
    known = {
        "data/kb/grace_period.md": "grace period",
        "data/kb/dispute_handling.md": "dispute handling",
    }
    text = (
        "An EMI paid after the 3-day grace period (see grace_period.md) is charged a flat late fee. "
        "A borrower who disputes the fee is treated as a dispute (see dispute_handling.md), not talked out of it."
    )

    found = _find_referenced_documents(text, known, exclude="data/kb/late_fee_policy.md")

    assert found == {"data/kb/grace_period.md", "data/kb/dispute_handling.md"}


def test_find_referenced_documents_returns_empty_for_unrelated_text():
    known = {"data/kb/grace_period.md": "grace period", "data/kb/dispute_handling.md": "dispute handling"}
    text = "The borrower's monthly income is verified against three months of bank statements."

    found = _find_referenced_documents(text, known, exclude="data/kb/late_fee_policy.md")

    assert found == set()
