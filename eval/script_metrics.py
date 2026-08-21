"""Classifies text by its dominant script -- Devanagari vs Latin -- via
simple Unicode character-range counting.

Deliberately not using a transliteration library for this: a library like
indic-transliteration has real, verified coverage gaps (e.g. it leaves the
Devanagari character "ऑ", used for English loanwords, unconverted), which
would quietly corrupt a metric built on top of it. Classifying dominant
script by counting characters has no such gap -- it doesn't need to
understand the language, just count two fixed character ranges.
"""

import re

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def dominant_script(text: str) -> str:
    """'devanagari' or 'latin' if one clearly dominates (more than 2x the
    other), 'mixed' if neither does, 'none' if there are no script
    characters at all (empty or punctuation-only)."""
    devanagari_count = len(_DEVANAGARI_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if devanagari_count == 0 and latin_count == 0:
        return "none"
    if devanagari_count > latin_count * 2:
        return "devanagari"
    if latin_count > devanagari_count * 2:
        return "latin"
    return "mixed"
