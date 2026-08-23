"""Tests for extract_loan_terms.

The parse-failure/degrade-to-None paths are pure logic (no network) and
tested directly against a faked client, same convention as
test_query_llm.py's groq.APIConnectionError fakes. The happy-path
extraction itself needs a real Groq call, so it's gated on GROQ_API_KEY
exactly like every other real-LLM test in this project (see
test_query_llm.py's _groq_skip).
"""

import os

import pytest

from businessflow.rag import extraction
from businessflow.rag.extraction import extract_loan_terms

_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
)


def _fake_client_returning(content: str):
    """A minimal stand-in for groq.Groq exposing only the
    .chat.completions.create(...) surface extract_loan_terms actually
    calls, returning a fixed message content -- same fake-client shape
    test_query_llm.py already uses for its own client() monkeypatches."""
    class _Message:
        pass

    class _Choice:
        pass

    class _Completion:
        pass

    message = _Message()
    message.content = content
    choice = _Choice()
    choice.message = message
    completion = _Completion()
    completion.choices = [choice]

    return type("FakeClient", (), {"chat": type("Chat", (), {
        "completions": type("Completions", (), {"create": staticmethod(lambda **kwargs: completion)})()
    })()})()


def test_extract_loan_terms_returns_none_on_unparseable_json(monkeypatch):
    # The model failing to follow the strict-JSON instruction must
    # degrade to "no rate found," not raise and take the upload endpoint
    # down with it.
    monkeypatch.setattr(extraction, "client", lambda: _fake_client_returning("not json at all"))

    assert extract_loan_terms("irrelevant document text") == {"interest_rate_pct": None}


def test_extract_loan_terms_returns_none_when_the_key_is_missing(monkeypatch):
    monkeypatch.setattr(extraction, "client", lambda: _fake_client_returning('{"some_other_field": 1}'))

    assert extract_loan_terms("irrelevant document text") == {"interest_rate_pct": None}


def test_extract_loan_terms_returns_none_on_a_non_numeric_rate(monkeypatch):
    # A string, a bool, or anything else that isn't a real number must
    # be discarded rather than written to a numeric DB column.
    monkeypatch.setattr(extraction, "client", lambda: _fake_client_returning('{"interest_rate_pct": "high"}'))

    assert extract_loan_terms("irrelevant document text") == {"interest_rate_pct": None}


def test_extract_loan_terms_passes_through_a_null_rate(monkeypatch):
    monkeypatch.setattr(extraction, "client", lambda: _fake_client_returning('{"interest_rate_pct": null}'))

    assert extract_loan_terms("a document that never mentions an interest rate") == {"interest_rate_pct": None}


def test_extract_loan_terms_parses_a_real_looking_numeric_response(monkeypatch):
    monkeypatch.setattr(extraction, "client", lambda: _fake_client_returning('{"interest_rate_pct": 14.5}'))

    result = extract_loan_terms("the interest rate is 14.5% per annum")

    assert result == {"interest_rate_pct": 14.5}


@_groq_skip
def test_extract_loan_terms_finds_a_clearly_stated_interest_rate():
    text = (
        "LOAN AGREEMENT\n\n"
        "This agreement is between the Lender and the Borrower for a "
        "term loan of Rs. 250,000, repayable in 24 monthly installments. "
        "The interest rate is 14.5% per annum, calculated on the "
        "outstanding principal."
    )

    result = extract_loan_terms(text)

    assert result["interest_rate_pct"] is not None
    assert abs(result["interest_rate_pct"] - 14.5) < 0.5


@_groq_skip
def test_extract_loan_terms_returns_none_when_no_rate_is_stated_anywhere():
    # Must not infer a rate from the EMI/principal/tenure that ARE
    # stated -- only a genuinely-stated rate should ever come back
    # non-null.
    text = (
        "LOAN AGREEMENT\n\n"
        "This agreement is between the Lender and the Borrower for a "
        "term loan of Rs. 250,000, repayable in 24 monthly installments "
        "of Rs. 12,500 each. No other financial terms are specified in "
        "this document."
    )

    result = extract_loan_terms(text)

    assert result["interest_rate_pct"] is None
