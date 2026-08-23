"""Structured extraction of precise, numeric loan-contract terms from an
uploaded document's raw text.

interest_rate_pct is the motivating case: it's a precise numeric contract
term, not something a RAG chunk-search over unstructured text should ever
have to approximate or infer from surrounding figures (EMI, principal).
The fix is to extract it once, at upload time, into a real structured
column (see schema.sql's accounts.interest_rate_pct) -- this module is
the extraction step, wired in by ops/api.py's upload_account_document.
"""

import json
import logging

from businessflow.agent.client import MODEL, client

logger = logging.getLogger(__name__)

# Strict JSON, one field only, and an explicit instruction to return null
# rather than infer a rate from the EMI/principal -- the same "never
# fabricate a number that wasn't actually stated" discipline
# agent/client.py's system prompt already enforces for the conversational
# agent, applied here to the extraction call instead.
_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured loan terms from the text of a signed loan "
    "agreement. Respond with strict JSON only, matching exactly this "
    'shape: {"interest_rate_pct": <number or null>}. interest_rate_pct '
    "is the loan's stated annual interest rate as a plain percentage "
    "number (e.g. 12.5 for 12.5% per annum), not a fraction and not a "
    "string. If no interest rate is stated anywhere in the text, "
    'respond with {"interest_rate_pct": null} -- never guess or infer '
    "one from the EMI amount, principal, or tenure. Only report a rate "
    "that the document actually states in words or figures."
)


def extract_loan_terms(document_text: str) -> dict:
    """Makes ONE real Groq call asking for interest_rate_pct only, as
    strict JSON (response_format={"type": "json_object"} -- supported by
    the installed groq SDK, verified via its chat.completions.create
    signature).

    A genuine call failure (a real Groq API error, e.g. a rate limit, or
    no key configured at all) is NOT caught here -- it propagates, same
    as query_llm.py's pattern of only catching what it can meaningfully
    degrade. The caller (ops/api.py's upload endpoint) is the one that
    decides a failed extraction shouldn't fail an otherwise-successful
    document upload, and logs it there.

    What IS handled here, by degrading to {"interest_rate_pct": None}
    rather than raising, is the model's response itself being unusable:
    not valid JSON, missing the key, or a non-numeric value. A failed
    parse must read as "we don't have it" -- exactly the existing
    _UNTRACKED_ACCOUNT_DATA prompt behavior -- never crash the upload
    endpoint or silently fabricate a plausible-looking rate.
    """
    completion = client().chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": document_text},
        ],
    )
    raw = completion.choices[0].message.content or ""

    try:
        parsed = json.loads(raw)
        rate = parsed["interest_rate_pct"]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning(
            "extract_loan_terms: model response wasn't parseable JSON with an "
            "interest_rate_pct key -- treating as 'no rate found'. raw=%r",
            raw,
        )
        return {"interest_rate_pct": None}

    if rate is None:
        return {"interest_rate_pct": None}
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        logger.warning(
            "extract_loan_terms: model returned a non-numeric interest_rate_pct=%r -- discarding",
            rate,
        )
        return {"interest_rate_pct": None}
    return {"interest_rate_pct": float(rate)}
