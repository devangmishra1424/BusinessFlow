"""Shared ASCII-alphanumeric tokenizer -- used both for BM25 indexing/
scoring in retriever.py and as the "has this query got any Latin-script
content at all" signal in query_llm.py. Split into its own module so
those two don't import each other."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
