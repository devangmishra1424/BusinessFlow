"""The Groq client and system prompt shared by both the bare, tool-less
reply() (used to prove the voice I/O shell works end to end) and the real
tool-calling agent loop in agent/loop.py.
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-20b"

# Advances through as many fallback keys as are actually configured
# (ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ALTERNATE_GROQ_KEY3, ...),
# and reverts back to the primary on its own after _FALLBACK_COOLDOWN_SECONDS
# -- Groq's quota here is a daily (tokens-per-day) limit, so a fixed 24h
# cooldown before retrying the primary matches when it actually has a real
# chance of having cleared. Originally this was a one-way, permanent-for-
# the-process switch to a single hardcoded fallback; for a long-running
# server that's a real bug, not just an inefficiency -- it would stay on
# the fallback forever even after the primary's quota reset the next day,
# and it had nowhere to go once that one fallback also got rate-limited.
_FALLBACK_KEY_ENV_VAR = "ALTERNATE_GROQ_KEY"
_MAX_FALLBACK_KEY_SUFFIX = 20  # ALTERNATE_GROQ_KEY2..20 -- generous headroom; adding one more needs no code change

_current_key_index = 0  # 0 = primary (GROQ_API_KEY); N>0 = the Nth configured fallback
_switched_at: float | None = None
_FALLBACK_COOLDOWN_SECONDS = 24 * 60 * 60


def _fallback_env_var_names() -> list[str]:
    """ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ALTERNATE_GROQ_KEY3,
    ... -- as many as are actually set in the environment right now,
    discovered dynamically rather than hardcoded to a fixed count.
    Checked by suffix rather than stopped at the first gap, so removing
    one out of order doesn't silently hide the ones configured after it."""
    names = []
    if os.environ.get(_FALLBACK_KEY_ENV_VAR):
        names.append(_FALLBACK_KEY_ENV_VAR)
    for n in range(2, _MAX_FALLBACK_KEY_SUFFIX + 1):
        var = f"{_FALLBACK_KEY_ENV_VAR}{n}"
        if os.environ.get(var):
            names.append(var)
    return names


def _active_env_var() -> str:
    if _current_key_index == 0:
        return "GROQ_API_KEY"
    return _fallback_env_var_names()[_current_key_index - 1]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a collections agent for an Indian SMB lender, speaking to a "
    "borrower. {account_context}{language_instruction} Be direct, warm, "
    "and brief -- this is a spoken conversation, not a written one. "
    "{commitment_discipline} {dispute_handling} {read_only_tools} {no_fabricated_links} {ground_policy_claims} "
    "{check_dispute_block_first} {out_of_domain_legal}"
)

# Real borrowers hedge ("maybe 15k", "not sure which is better") instead of
# committing outright -- treating a hedge as a firm decision and acting on
# it (logging a promise, sending a payment link) is worse than asking one
# confirming question. Found via eval/realistic_conversation_benchmark.py:
# without this, a "maybe 15k, not sure" got logged as an accepted promise
# with a fabricated due date the borrower never gave.
_COMMITMENT_DISCIPLINE = (
    "Only call log_promise_to_pay, generate_payment_link, or "
    "propose_partial_payment when the borrower has given a specific, firm "
    "amount (and, for a promise, a specific date) -- if they're hedging "
    "('maybe', 'not sure', thinking out loud, or offering several options "
    "at once), ask one short confirming question instead of acting on it. "
    "Never invent a date or amount the borrower did not actually state."
)

# flag_dispute exists specifically to freeze automated collection action
# while a HUMAN verifies -- asking the borrower for "proof" the agent has
# no actual way to check just delays that freeze, it doesn't verify
# anything. Found via eval/realistic_conversation_benchmark.py: the same
# dispute claim, worded with typos, got met with "share your transaction
# reference" instead of an immediate flag -- noisy phrasing shouldn't
# change whether a dispute claim gets flagged.
_DISPUTE_HANDLING = (
    "If a borrower says they already paid, or disputes a charge, call "
    "flag_dispute right away, even if you can't verify it yourself -- a "
    "human will check the details afterward. Don't withhold the flag "
    "while asking for proof you have no way to actually verify."
)

# get_payment_status and calculate_hypothetical look up everything they
# need (balance, EMI, months remaining) from account_id alone -- neither
# needs the borrower to state a figure first. Found via repeated runs of
# eval/tool_calling_benchmark.py: on some runs, a plain "what would a
# settlement cost me" got met with "tell me your remaining balance"
# instead of just calling the tool, which already knows it.
_READ_ONLY_TOOLS = (
    "get_payment_status and calculate_hypothetical need only account_id "
    "-- call them directly to answer a status or 'what if' question, "
    "never ask the borrower to state a balance or figure the tool "
    "already looks up itself."
)

# Found live via eval/realistic_conversation_benchmark.py's
# many_operations_same_account_en scenario: asked to lower/stretch a
# payment on an account it already knew (this same session) had an open
# dispute, the model asked "how many extra months would you like?"
# instead of checking eligibility first -- a dead-end question, since the
# dispute was always going to block it regardless of the answer. The
# same model correctly went straight to calling propose_partial_payment
# (and got the same block) for a different restructuring ask two turns
# later in that same conversation -- so this is a real, specific
# inconsistency to close, not a hard model limitation.
_CHECK_DISPUTE_BLOCK_FIRST = (
    "If you already know (from get_payment_status or anything said "
    "earlier this session) that an account has an open dispute or "
    "repeated broken promises, don't ask for restructuring specifics "
    "(how many months, how much lower) before checking -- go straight "
    "to calculate_hypothetical or propose_partial_payment (or state the "
    "block directly and offer escalation) instead of gathering details "
    "for an offer you already know won't be approved. But once the "
    "borrower has already given a specific number or month count -- "
    "there's nothing left to gather -- always call calculate_hypothetical "
    "or propose_partial_payment with that real figure before stating any "
    "outcome. 'State the block directly' is for when nothing concrete has "
    "been proposed yet, never a substitute for checking a real number the "
    "borrower already gave you -- even late in a long conversation, after "
    "you've already discussed this account's dispute or broken promises "
    "several times, a NEW concrete number still needs its own real tool "
    "call. Don't rely on what you already know about the account instead "
    "of checking the specific new figure."
)

# Found live via eval/realistic_conversation_benchmark.py: after backing
# out of a settlement offer and asking for "the regular link", the model
# had already fetched the real EMI via get_payment_status but then wrote
# out its OWN made-up URL instead of calling generate_payment_link --
# presented as a real link, even though it isn't one that actually works.
_NO_FABRICATED_LINKS = (
    "Never write out a payment link yourself, not even as an example -- "
    "a URL you compose is not real and will not work. Always call "
    "generate_payment_link to get the actual link before mentioning one."
)

# Found live via eval/realistic_conversation_benchmark.py's
# many_operations_same_account_en scenario: asked how long a dispute
# "usually" takes to resolve, the model stated "5-7 business days" --
# a specific, confident-sounding number that appears nowhere in the real
# policy docs and was never checked (check_policy wasn't called that
# turn). The Guardrail (guardrail/grounding.py) only checks URLs and
# rupee amounts, so a plain-text claim like this slips through it
# entirely -- this has to be caught here, in the prompt, instead.
_GROUND_POLICY_CLAIMS = (
    "Never state a specific policy detail -- a fee, a timeline, a "
    "resolution window, how long something 'usually' takes -- from "
    "memory. Call check_policy first and ground the answer in what it "
    "actually returns; if it doesn't have an answer, say you're not "
    "sure and will find out, rather than giving a plausible-sounding "
    "number."
)

# Found via eval/red_team.py's out_of_domain_legal scenario: asked
# whether the loan agreement was "legally enforceable" and about "legal
# rights," the model didn't give a wrong legal opinion (good), but also
# didn't recognize the question was out of scope -- it said it would
# "check the policy documents and get back to you," a dead end, since
# check_policy's KB is internal collections policy, not legal counsel.
_OUT_OF_DOMAIN_LEGAL = (
    "If a borrower asks a genuinely legal question -- whether the loan "
    "agreement is enforceable, their legal rights, whether they should "
    "sue or be sued -- that is out of scope for check_policy (it only "
    "covers internal collections policy, not legal advice) and out of "
    "scope for you. Say plainly that you can't give legal advice and "
    "offer to escalate to a human, rather than saying you'll research "
    "it and follow up."
)

_NO_ACCOUNT_CONTEXT = (
    "You do not yet have access to any real account data or tools -- say "
    "so plainly if asked something you can't actually check. "
)

# The TTS engine for Hindi (Meta's MMS-TTS) only recognizes Devanagari
# characters -- its tokenizer returns an empty token sequence on Romanized
# Hindi ("Hinglish" in Latin script) and the model crashes downstream on
# that empty input. So each reply is single-script for now: full Devanagari
# Hindi, or full English -- not true intra-sentence code-switching, which
# would need the reply split into per-language runs and stitched across two
# TTS calls, deferred until the pipeline needs it.
_LANGUAGE_INSTRUCTIONS = {
    "en": "Reply in plain English.",
    "hi": (
        "Reply in Hindi, written entirely in Devanagari script (e.g. "
        "आपके), never in Romanized/Latin-script Hindi -- the "
        "text-to-speech engine cannot pronounce Romanized Hindi at all."
    ),
}


def client() -> Groq:
    global _current_key_index
    if _current_key_index > 0 and _switched_at is not None:
        if time.time() - _switched_at >= _FALLBACK_COOLDOWN_SECONDS:
            _current_key_index = 0  # cooldown elapsed -- give the primary another chance

    env_var = _active_env_var()
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set -- copy .env.example to .env and fill it in")
    return Groq(api_key=api_key)


def switch_to_fallback_key() -> bool:
    """Called when the currently active key's requests start failing
    with a rate limit. Advances to the next configured fallback
    (ALTERNATE_GROQ_KEY, then ALTERNATE_GROQ_KEY2, ...), if there's one
    left that hasn't been tried yet this round. Returns True if it
    actually advanced, False if every configured key is already in use
    -- the caller should let the original error propagate in that case."""
    global _current_key_index, _switched_at
    fallback_names = _fallback_env_var_names()
    if _current_key_index < len(fallback_names):
        _current_key_index += 1
        _switched_at = time.time()
        return True
    return False


def build_system_prompt(language: str = "en", account_id: str | None = None) -> str:
    """account_id, when given, tells the agent which borrower it's
    speaking with -- it's how the model knows what to pass as account_id
    when it calls a tool, the same way a real deployment would identify
    the caller by phone/WhatsApp number rather than asking them to state
    an internal account ID out loud."""
    account_context = (
        f"You are speaking with the borrower on account {account_id}. "
        if account_id
        else _NO_ACCOUNT_CONTEXT
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        account_context=account_context,
        language_instruction=_LANGUAGE_INSTRUCTIONS[language],
        commitment_discipline=_COMMITMENT_DISCIPLINE,
        dispute_handling=_DISPUTE_HANDLING,
        read_only_tools=_READ_ONLY_TOOLS,
        no_fabricated_links=_NO_FABRICATED_LINKS,
        ground_policy_claims=_GROUND_POLICY_CLAIMS,
        check_dispute_block_first=_CHECK_DISPUTE_BLOCK_FIRST,
        out_of_domain_legal=_OUT_OF_DOMAIN_LEGAL,
    )


def reply(user_message: str, language: str = "en") -> str:
    """A single-turn reply with no memory of prior turns and no tool
    access -- the placeholder reasoning step for the voice I/O shell.
    language is "en" or "hi" and controls which script the reply must use."""
    completion = client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(language)},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content
