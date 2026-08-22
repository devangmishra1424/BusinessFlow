"""Tests for query_llm.py -- the Hindi/Hinglish detection is pure logic
(no network), tested directly; translate_to_english/expand_query need a
real Groq call to test their happy path (same GROQ_API_KEY-gated
convention as test_agent_loop.py), but their failure path -- falling
back to the original query on a Groq error -- is tested by forcing that
error directly, since a bug there would mean a transient Groq outage
breaks retrieval outright instead of just degrading it.
"""

import os

import groq
import pytest

from businessflow.rag import query_llm

_groq_skip = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
)


@pytest.mark.parametrize("query,expected", [
    ("can I get a few more days to pay", False),
    ("yo i defo already paid tht EMI last wk trust me", False),
    # A real false positive found via a live Groq translation test: "the",
    # "main", "do", and "par" are common standalone English words that
    # used to be on the marker list, which meant a *correctly translated*
    # English sentence could still get flagged as needing translation.
    ("The late fee that has been applied to me is wrong, dude.", False),
    ("can you do the main calculation, at par with last time", False),
    ("क्या मुझे कुछ और दिन मिल सकते हैं", True),
    ("??? !!! ...", True),
    ("mera jo late fee laga hai na wo galat hai yaar", True),
    ("EMI kam karne ka koi tarika hai kya", True),
    ("is mahine thoda kam de doon to chalega kya", True),
])
def test_needs_translation_detects_hindi_hinglish_and_untokenizable_queries(query, expected):
    assert query_llm.needs_translation(query) is expected


def test_translate_to_english_falls_back_to_original_query_on_groq_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise groq.APIConnectionError(request=None)

    fake_client = type("FakeClient", (), {"chat": type("Chat", (), {
        "completions": type("Completions", (), {"create": staticmethod(_raise)})()
    })()})()
    monkeypatch.setattr(query_llm, "client", lambda: fake_client)

    result = query_llm.translate_to_english("क्या मुझे कुछ और दिन मिल सकते हैं")

    assert result == "क्या मुझे कुछ और दिन मिल सकते हैं"


def test_translate_to_english_falls_back_to_original_query_when_no_key_is_configured(monkeypatch):
    # Real bug found via CI (which has no GROQ_API_KEY at all):
    # client() itself raises a plain RuntimeError before ever reaching
    # a Groq API call -- a distinct failure point from groq.GroqError,
    # but needing the exact same graceful fallback. Without this, any
    # environment with no/misconfigured key would hard-crash retrieval
    # instead of just degrading it.
    def _raise():
        raise RuntimeError("GROQ_API_KEY is not set -- copy .env.example to .env and fill it in")

    monkeypatch.setattr(query_llm, "client", _raise)

    result = query_llm.translate_to_english("क्या मुझे कुछ और दिन मिल सकते हैं")

    assert result == "क्या मुझे कुछ और दिन मिल सकते हैं"


def test_expand_query_falls_back_to_original_only_on_groq_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise groq.APIConnectionError(request=None)

    fake_client = type("FakeClient", (), {"chat": type("Chat", (), {
        "completions": type("Completions", (), {"create": staticmethod(_raise)})()
    })()})()
    monkeypatch.setattr(query_llm, "client", lambda: fake_client)

    result = query_llm.expand_query("can I get a few more days to pay")

    assert result == ["can I get a few more days to pay"]


def test_expand_query_falls_back_to_original_when_no_key_is_configured(monkeypatch):
    def _raise():
        raise RuntimeError("GROQ_API_KEY is not set -- copy .env.example to .env and fill it in")

    monkeypatch.setattr(query_llm, "client", _raise)

    result = query_llm.expand_query("can I get a few more days to pay")

    assert result == ["can I get a few more days to pay"]


@_groq_skip
def test_translate_to_english_actually_translates_hinglish():
    result = query_llm.translate_to_english("mera jo late fee laga hai na wo galat hai yaar")
    assert result != "mera jo late fee laga hai na wo galat hai yaar"
    assert query_llm.needs_translation(result) is False


@_groq_skip
def test_expand_query_returns_original_plus_variants():
    result = query_llm.expand_query("can I get a few more days to pay", n_variants=2)
    assert result[0] == "can I get a few more days to pay"
    assert len(result) >= 2  # at least the original plus one real variant
    assert len(result) <= 3
