"""The Guardrail: a mechanical check that anything stated in a reply --
a URL, a currency amount -- actually traces back to a real tool result
or the borrower's own words somewhere in this conversation, not
something the model invented.

This is the architectural piece the original blueprint called for as a
distinct step before speaking ("ground before speaking... before
synthesis") that had been missing from the actual build -- its absence
is exactly what let the agent fabricate a fake payment link earlier this
session (see eval/realistic_conversation_benchmark.py's
settlement_then_backpedal_hi scenario: it invented
"https://payments.example.com/..." instead of calling generate_payment_link).

Deliberately narrow: checks URLs and rupee amounts specifically, not
every digit in the reply -- a blanket "every number must be grounded"
check would false-positive on ordinary descriptive numbers ("3 days",
"a 5% discount") and make the guardrail useless through over-triggering.
URLs and money are the two categories where an invented value is
genuinely dangerous.

Known, accepted trade-off: this checks literal values, not derived
arithmetic. Found live -- the model correctly computed "you'd save
₹22,000" from two real, grounded tool numbers (₹440,000 owed minus a
₹418,000 settlement), and got blocked anyway, since "22000" itself
never appears in any tool result. Verifying arbitrary arithmetic
correctness would need real computation, not a text-matching check, and
risks its own bugs -- favoring occasionally blocking a correct computed
number over letting a genuinely invented one through is the deliberate
choice here, not an oversight.
"""

import re

# Excludes [ and ] from the URL body entirely, not just as trailing
# punctuation to strip -- found live: the model wrote a Markdown link,
# "[https://...20000.0](https://...20000.0)", with no whitespace between
# the link text and the href, so a bare \S+ swallowed straight through
# the "](" into a second embedded URL, producing one long "URL" that
# matches nothing. Stopping at [ and ] correctly yields two separate,
# clean matches (the link text and the href) that a set naturally
# deduplicates once trailing punctuation like the closing ')' is stripped.
_URL_RE = re.compile(r"https?://[^\s\[\]]+")
# Allows a single embedded space between digit groups -- found live: the
# model wrote a Devanagari-numeral amount as "₹५ ८५,२००" (meant to be one
# number, ५,८५,२०० = 585200) with a stray space instead of a separator.
# Without this, [\d,]+ stops at the space and captures only "५" (5) as
# its own isolated "amount", which correctly matches nothing -- a false
# positive from the regex being too strict, not a real hallucination.
_RUPEE_AMOUNT_RE = re.compile(r"₹\s*([\d,]+(?: [\d,]+)*(?:\.\d+)?)")
_PLAIN_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Real borrowers write "15k" or "1.5L", not "15000" or "150000" -- found
# live: a user said "maybe 15k for now?", the model correctly asked
# "will you pay ₹15,000...?" (the exact hedge-handling behavior meant to
# happen), and the guardrail blocked its own correct question because
# "15k" and "15000" don't look alike as plain text.
_SHORTHAND_THOUSAND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[kK]\b")
_SHORTHAND_LAKH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:l|L|lakh|lakhs)\b")


def extract_urls(text: str) -> set[str]:
    # Trailing punctuation/markup isn't part of the URL -- strip it
    # before comparing, or the match never lines up with the clean URL
    # from a real tool result. Found live, one character at a time, each
    # a genuine case: '"' (a URL embedded in a tool result's own JSON
    # string, followed by a closing quote), '>' (the model wrapped a real
    # link in angle brackets, <https://...>). Covering the wider set of
    # common wrappers/sentence punctuation up front instead of continuing
    # to patch this one character at a time.
    return {u.rstrip("\">.,!?。）')]}*`;:") for u in _URL_RE.findall(text)}


def extract_rupee_amounts(text: str) -> set[float]:
    amounts = set()
    for m in _RUPEE_AMOUNT_RE.finditer(text):
        try:
            amounts.add(float(m.group(1).replace(",", "").replace(" ", "")))
        except ValueError:
            continue
    return amounts


def extract_shorthand_amounts(text: str) -> set[float]:
    amounts = set()
    for m in _SHORTHAND_THOUSAND_RE.finditer(text):
        try:
            amounts.add(float(m.group(1)) * 1_000)
        except ValueError:
            continue
    for m in _SHORTHAND_LAKH_RE.finditer(text):
        try:
            amounts.add(float(m.group(1)) * 100_000)
        except ValueError:
            continue
    return amounts


def extract_plain_numbers(text: str) -> set[float]:
    numbers = set()
    for m in _PLAIN_NUMBER_RE.finditer(text):
        try:
            numbers.add(float(m.group(0)))
        except ValueError:
            continue
    return numbers


class GroundingFailure:
    def __init__(self, ungrounded_urls: set[str], ungrounded_amounts: set[float]):
        self.ungrounded_urls = ungrounded_urls
        self.ungrounded_amounts = ungrounded_amounts

    def __bool__(self) -> bool:
        return bool(self.ungrounded_urls or self.ungrounded_amounts)

    def describe(self) -> str:
        parts = []
        if self.ungrounded_urls:
            parts.append(f"URL(s) not from any real tool result: {sorted(self.ungrounded_urls)}")
        if self.ungrounded_amounts:
            parts.append(f"amount(s) not from any real tool result or the borrower's own words: {sorted(self.ungrounded_amounts)}")
        return "; ".join(parts)


# Found live via the Telegram channel: asked to email the loan agreement,
# the model claimed to have "arranged to send" it, then two turns later
# restated it as settled fact ("I've already sent a copy") -- there is no
# email/SMS/document-delivery tool anywhere in this system at all, so
# unlike a rupee amount (which sometimes IS legitimately grounded),
# there is no scenario where this claim could ever be true right now.
# That makes it safe to check unconditionally, the same "deliberately
# narrow, but for a case where a blanket check can't false-positive"
# reasoning the URL/rupee checks above already rely on.
_FABRICATED_ACTION_RE = re.compile(
    r"\bi(?:'ve| have) (?:already )?(?:sent|emailed|mailed|forwarded)\b"
    r"|\barranged (?:for [^.]*? )?to be sent\b"
    r"|\barranged to send\b"
    r"|\b(?:document|agreement|contract|copy) i(?:'ve| have)? sent\b",
    re.IGNORECASE,
)


class FabricatedActionClaim:
    def __init__(self, matched_phrase: str | None):
        self.matched_phrase = matched_phrase

    def __bool__(self) -> bool:
        return self.matched_phrase is not None

    def describe(self) -> str:
        return (
            f"reply claims to have sent/emailed/delivered something ({self.matched_phrase!r}) -- "
            "no such capability exists in this system at all"
        )


def check_fabricated_action(reply_text: str) -> FabricatedActionClaim:
    """No conversation history needed, unlike check_grounding -- there is
    no tool call, ever, that could make this claim true, so it doesn't
    matter what happened earlier in the conversation."""
    match = _FABRICATED_ACTION_RE.search(reply_text)
    return FabricatedActionClaim(match.group(0) if match else None)


def check_grounding(reply_text: str, conversation: list[dict]) -> GroundingFailure:
    """conversation is the FULL conversation so far (not just this turn) --
    a value established two turns ago via a real tool call is still
    grounded now; scoping to "this turn only" would false-positive on
    ordinary multi-turn context reuse (e.g. restating an EMI amount
    fetched earlier without re-calling the tool)."""
    grounded_text_parts = [
        msg["content"] for msg in conversation
        if msg.get("role") in ("tool", "user") and isinstance(msg.get("content"), str)
    ]
    grounded_text = " ".join(grounded_text_parts)

    grounded_urls = extract_urls(grounded_text)
    grounded_amounts = (
        extract_rupee_amounts(grounded_text)
        | extract_plain_numbers(grounded_text)
        | extract_shorthand_amounts(grounded_text)
    )

    reply_urls = extract_urls(reply_text)
    reply_amounts = extract_rupee_amounts(reply_text)

    ungrounded_urls = {u for u in reply_urls if u not in grounded_urls}
    ungrounded_amounts = {a for a in reply_amounts if not any(abs(a - g) < 0.01 for g in grounded_amounts)}

    return GroundingFailure(ungrounded_urls, ungrounded_amounts)
