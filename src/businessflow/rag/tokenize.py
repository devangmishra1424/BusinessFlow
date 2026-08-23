"""Shared ASCII-alphanumeric tokenizer -- used both for BM25 indexing/
scoring in retriever.py and as the "has this query got any Latin-script
content at all" signal in query_llm.py. Split into its own module so
those two don't import each other.

Also home to dominant_script(), a second, complementary text-classification
signal: not "are there tokenizable Latin characters at all" but "which
script actually dominates this text." retriever.py needs this to detect
Devanagari-dominant CANDIDATE CHUNKS (as opposed to queries, which
needs_translation()/the empty-tokenize()-check already handle) so it can
route them around the English-only cross-encoder reranker the same way.

dominant_script's logic is deliberately identical to eval/script_metrics.py's
function of the same name (same two Unicode ranges, same 2x-margin rule) --
that one is not reused directly because eval/ already imports FROM
businessflow (eval/wer_benchmark.py does), and eval/ isn't part of the
installed src/ package (see pyproject.toml's packages.find, scoped to
src/ only) -- importing it from src/businessflow/rag would both invert
that dependency direction and add a hard dependency on a dev-only
package that may not exist in a production install.
"""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def dominant_script(text: str) -> str:
    """'devanagari' or 'latin' if one clearly dominates (more than 2x the
    other), 'mixed' if neither does, 'none' if there are no script
    characters at all (empty or punctuation-only). Simple Unicode
    character-range counting, not a language-ID model -- deliberately so;
    see eval/script_metrics.py's module docstring for why a transliteration
    library was tried and rejected for this same classification."""
    devanagari_count = len(_DEVANAGARI_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if devanagari_count == 0 and latin_count == 0:
        return "none"
    if devanagari_count > latin_count * 2:
        return "devanagari"
    if latin_count > devanagari_count * 2:
        return "latin"
    return "mixed"
