"""LLM-based query normalization for retrieval, reusing the same Groq
client/model as the main conversational agent (agent/client.py) rather
than standing up a second one.

Two independent, separately-measurable interventions:
  - translate_to_english(): normalizes a Hindi/Hinglish query to plain
    English before it reaches retriever.py's pipeline -- whose BM25
    tokenizer is ASCII-only and whose cross-encoder reranker is
    English-only (verified live: it can demote a chunk the multilingual
    embedding stage had already ranked correctly for a Devanagari query).
  - expand_query(): generates alternate phrasings of an already-English
    query, to widen recall when a borrower's wording doesn't overlap
    much with the KB doc that actually answers it -- a general
    RAG-quality lever, unrelated to language.

Both degrade to a safe no-op (return the input unchanged / a
single-item list) on any Groq failure -- a broken enhancement should
degrade retrieval quality, never break it outright.
"""

import logging

import groq

from businessflow.agent.client import MODEL, client
from businessflow.rag.tokenize import tokenize as _tokenize

logger = logging.getLogger(__name__)

# A deliberately simple, auditable word-list check -- not a language-ID
# model. _tokenize can't distinguish Hinglish from English at all (both
# are Latin-script and pass the same [a-z0-9]+ regex), so detecting
# code-switching needs a different signal than "has real tokens." The
# collections domain has a narrow, predictable vocabulary, so a curated
# list of common Hindi/Hinglish function words catches the overwhelming
# majority of real code-switched queries without a new ML dependency.
#
# Deliberately excludes a few transliterations that collide with common
# standalone English words -- "the" (the/the), "main" (main street),
# "do" (do you), "par" (par for the course) -- found via a real false
# positive in testing: translate_to_english() correctly translated a
# Hinglish sentence to "...is wrong, dude." and needs_translation() then
# flagged the ENGLISH RESULT as still needing translation, because "the"
# was on this list. A marker word that fires on ordinary English text
# defeats the entire point of the check.
_HINDI_MARKER_WORDS = frozenset({
    "hai", "hain", "ho", "hoga", "hogi", "tha", "thi",
    "nahi", "nahin", "kya", "kyun", "kyu", "kaise", "kab", "kaha", "kahan",
    "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski", "unka",
    "aap", "tum", "hum", "mai", "mujhe", "mujhko", "tumhe", "humein",
    "kar", "karo", "karna", "karke", "kiya", "kijiye", "karu", "karunga",
    "de", "diya", "dena", "denge", "milega", "milegi", "mil",
    "paisa", "paise", "rupaye", "rupiya",
    "yaar", "bhai", "bata", "bolo", "bola", "chahiye", "chalega", "chalegi",
    "abhi", "aaj", "kal", "tak", "se", "ka", "ki", "ke", "ko", "mein", "pe",
    "thoda", "zyada", "poora", "sab", "bas", "phir", "waala", "wala",
})


def needs_translation(query: str) -> bool:
    """True if the query has no Latin-alphanumeric content at all (pure
    Devanagari, punctuation-only -- retriever.py's BM25 stage gets zero
    signal either way), or contains recognizable Hindi/Hinglish marker
    words mixed into otherwise-Latin-script text."""
    tokens = _tokenize(query)
    if not tokens:
        return True
    return bool(set(tokens) & _HINDI_MARKER_WORDS)


def translate_to_english(query: str) -> str:
    """Best-effort translation via the same Groq model the main agent
    uses. Falls back to the original query, unchanged, on any Groq
    failure -- a failed translation should degrade retrieval quality,
    not break it outright."""
    try:
        completion = client().chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Translate the user's message into clear, natural English. "
                    "It may be in Hindi, Hinglish (Hindi written in Latin script, "
                    "possibly mixed with English), or already English. Respond "
                    "with ONLY the English translation -- no quotes, no "
                    "explanation, no preamble."
                )},
                {"role": "user", "content": query},
            ],
        )
        translated = completion.choices[0].message.content
        return translated.strip() if translated else query
    except groq.GroqError:
        logger.warning("query translation failed, retrieving with the original query instead", exc_info=True)
        return query


def expand_query(query: str, n_variants: int = 2) -> list[str]:
    """The original query plus up to n_variants alternate phrasings of
    the same underlying question -- widens recall for a query that uses
    different words than the KB doc that actually answers it. Falls
    back to [query] on any Groq failure.

    A/B'd against the full retrieval_benchmark.py suite: no clear
    aggregate win (a small recall@1 gain offset by a recall@5 loss), so
    check_policy doesn't enable this by default. But high per-query
    variance hides in that wash -- found live, a single hard, indirectly-
    phrased query ("is there someone I can talk to about a payment
    that's not showing up on my end", genuinely meant as a dispute) had
    the reranker score the correct doc a razor-thin -6.65 vs the wrong
    one's -6.85 without expansion (confirmed neither BM25 nor the
    embedding stage favored the correct doc either -- not a reranker-
    specific bug, a genuinely hard case for every stage), and a clean
    -3.92 vs -6.78 WITH expansion. A future improvement worth building:
    only expand when the top result's own score is ambiguous (near a
    threshold), rather than always-on or always-off."""
    try:
        completion = client().chat.completions.create(
            model=MODEL,
            temperature=0.7,
            messages=[
                {"role": "system", "content": (
                    f"Given a user's question, write {n_variants} alternate ways of "
                    "asking the same underlying question, using different words or "
                    "phrasing than the original -- the same intent, not a different "
                    "question. Respond with ONLY the alternate phrasings, one per "
                    "line, no numbering, no explanation."
                )},
                {"role": "user", "content": query},
            ],
        )
        content = completion.choices[0].message.content or ""
        variants = [line.strip() for line in content.splitlines() if line.strip()]
        return [query] + variants[:n_variants]
    except groq.GroqError:
        logger.warning("query expansion failed, retrieving with only the original query", exc_info=True)
        return [query]
