"""The real agent: a multi-turn, tool-calling loop over Groq. This is what
actually connects the reasoning model to the 8 real tools (including
check_policy, which is the RAG retriever) -- replacing client.reply(),
which just talks without checking anything against real data.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import groq
import langfuse

from businessflow.accounts import store
from businessflow.accounts.escalation_kinds import GUARDRAIL_REASON_PREFIX
from businessflow.agent import prompt_versions
from businessflow.agent.client import MODEL, build_system_prompt, client, reset_to_primary_key, switch_to_fallback_key
from businessflow.guardrail import grounding
from businessflow.guardrail.unverified_restructuring import check_unverified_restructuring_claim
from businessflow.memory import conversation_memory
from businessflow.observability import trace
from businessflow.tools import mcp

logger = logging.getLogger(__name__)

# A hard cap -- never loop on tool calls forever. Was 5; found live that
# this is what let ONE confused turn (a badly garbled ASR transcript,
# which the model never settled an answer for) burn through every
# configured Groq key's independent per-minute budget by itself -- each
# round resends the full system prompt + all tool schemas (~5,000+
# tokens on its own), so 5 rounds plus the MAX_TOOL_ROUNDS-forced final
# answer is up to 6 full-price completion calls in one turn, enough to
# exhaust several 8,000-token/minute keys in sequence. Every real logged
# turn in eval/results/latency_benchmark.json -- including scenarios
# literally named "multi_tool_settlement" -- made at most 1 tool call
# (2 rounds) before answering; nothing in this project's actual usage
# has ever needed more. 3 keeps real headroom above that observed
# maximum while cutting the worst-case burn nearly in half.
MAX_TOOL_ROUNDS = 3

# Found live: a long-lived, never-/reset session (70+ real turns on one
# account across several days, via Telegram) resends its ENTIRE history on
# every single completion call within a turn -- including once per tool-
# calling round, not just once per turn -- and that's what actually burst
# a real conversation past Groq's per-minute token cap (8,000, much
# smaller and easier to hit than the 200k/day one), not sustained
# exhaustion. Safe to trim: every message and tool call is already
# durably persisted independently of this in-memory list (accounts.store.
# log_event for tool calls, memory/conversation_memory.log_turn for
# messages, both keyed to the account) -- this list's only job is being
# the model's own working context, not the system of record. 20 is
# comfortably above any real demo or eval conversation's real length
# (every scripted benchmark scenario is a handful of turns), so this
# never fires in normal use -- it only kicks in for a session that's
# actually run away unbounded, exactly the failure mode found live.
_MAX_CONVERSATION_TURNS = 20

# openai/gpt-oss-20b occasionally leaks its internal "harmony" format channel
# tags (e.g. "<|channel|>commentary") into the tool name it emits, which Groq
# rejects outright with a 400 before we ever see a tool_calls response --
# observed live: 'check_policy<|channel|>commentary'. This is a malformed
# *generation*, not a malformed request on our side -- retrying the identical
# request has a real chance of not reproducing it, since decoding is
# stochastic. Bounded to a couple of attempts; any other BadRequestError
# (e.g. a real bug in our own tool schema) is not this pattern and propagates
# immediately, per the project's "never retry a 4xx" rule for genuine
# client-side errors.
_MALFORMED_TOOL_CALL_RETRIES = 2

# The SDK's own retrying is switched off for this path (client(fail_fast=
# True), see its docstring: on a 429 the SDK slept for Retry-After before
# the key rotation below could run). That also drops its automatic retry of
# genuinely transient failures -- a dropped connection, a timeout, a 5xx --
# so those get a small, bounded budget of their own here instead.
_TRANSIENT_ERROR_RETRIES = 2
_TRANSIENT_ERROR_BACKOFF_SECONDS = 0.5

# Rotating to another key is instant, but when EVERY configured key is out of
# its per-minute tokens at once (measured: 8,000 tokens/min each, and a tool
# turn spends ~11k) there is nothing left to rotate to and the only thing
# that helps is waiting for a bucket to refill. The SDK used to do that wait
# for us, up to 60 s per attempt, before any rotation could happen; now the
# wait happens only in this saturated case, only for as long as the server
# says (Retry-After, capped), and only a bounded number of times.
_RATE_LIMIT_WAIT_ROUNDS = 2
_RATE_LIMIT_DEFAULT_WAIT_SECONDS = 10.0  # when a 429 carries no usable Retry-After
_RATE_LIMIT_MAX_WAIT_SECONDS = 30.0
# Patient mode, for the nightly evals only (BUSINESSFLOW_PATIENT_RATE_LIMITS=1, set by scripts/run_eval_monitor.py): the
# free tier allows 8,000 tokens a minute and one call carries ~5,000, so a suite firing calls back to back is
# rate-limited within two calls. A borrower must not wait minutes, but an eval at 2:30 am can: found 2026-10-23, every
# real-model suite had crashed on 429 every night since 14 Oct, leaving the Quality page on old numbers.
_PATIENT_WAIT_ROUNDS = 20
_PATIENT_MAX_WAIT_SECONDS = 65.0


def _patient() -> bool:
    return os.environ.get("BUSINESSFLOW_PATIENT_RATE_LIMITS") == "1"


def _retry_after_seconds(error: "groq.RateLimitError") -> float:
    """The server's own Retry-After from the 429 (seconds), clamped to
    [1, _RATE_LIMIT_MAX_WAIT_SECONDS]; a default when it is absent or isn't
    a plain number."""
    headers = error.response.headers if error.response is not None else {}
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        seconds = _RATE_LIMIT_DEFAULT_WAIT_SECONDS
    return min(max(seconds, 1.0), _PATIENT_MAX_WAIT_SECONDS if _patient() else _RATE_LIMIT_MAX_WAIT_SECONDS)


# What a reply that fails the guardrail (_finalize_reply) is replaced with. Named so evals can recognise it.
GUARDRAIL_SAFE_REPLY = "Let me connect you with someone who can confirm those exact details before we go further."


@langfuse.observe(name="groq_completion", as_type="generation")
def _create_completion(**kwargs):
    # Four independent retry budgets, deliberately not sharing one counter:
    # rotating through keys is bounded by however many real fallback keys
    # are configured (switch_to_fallback_key() itself returns False once
    # they're all tried), waiting out a fully saturated round is capped at
    # _RATE_LIMIT_WAIT_ROUNDS, while malformed-tool-call retries and
    # transient-error retries are each capped regardless of how many keys
    # got switched through along the way.
    malformed_tool_call_attempts = 0
    transient_error_attempts = 0
    rate_limit_waits = 0
    while True:
        groq_client, current_key_index = client(fail_fast=True)  # re-fetched each attempt -- picks up whichever key is currently active
        try:
            return groq_client.chat.completions.create(**kwargs)
        except (groq.APIConnectionError, groq.InternalServerError) as e:
            # APITimeoutError is a subclass of APIConnectionError. Same
            # key, short backoff -- a rate limit would be RateLimitError,
            # not these.
            if transient_error_attempts >= _TRANSIENT_ERROR_RETRIES:
                raise
            transient_error_attempts += 1
            trace.note(transient_retries=1)
            logger.warning(
                "transient Groq error (%s) -- retrying (attempt %d of %d)",
                type(e).__name__, transient_error_attempts, _TRANSIENT_ERROR_RETRIES,
            )
            time.sleep(_TRANSIENT_ERROR_BACKOFF_SECONDS * transient_error_attempts)
        except groq.RateLimitError as e:
            # This key is out of tokens for now -- advance to the next
            # configured fallback key, if any are left untried this round.
            if switch_to_fallback_key(current_key_index):
                logger.warning("Groq key rate-limited -- switched to the next configured fallback key")
                continue
            # Every configured key has been tried: wait for a bucket to
            # refill (bounded), then start over from the primary key.
            if rate_limit_waits >= (_PATIENT_WAIT_ROUNDS if _patient() else _RATE_LIMIT_WAIT_ROUNDS):
                raise
            rate_limit_waits += 1
            trace.note(rate_limit_waits=1)
            wait_seconds = _retry_after_seconds(e)
            logger.warning(
                "every configured Groq key is rate-limited -- waiting %.0fs for one to refill (round %d of %d)",
                wait_seconds, rate_limit_waits, _RATE_LIMIT_WAIT_ROUNDS,
            )
            time.sleep(wait_seconds)
            reset_to_primary_key()
        except groq.BadRequestError as e:
            code = (e.body or {}).get("error", {}).get("code") if isinstance(e.body, dict) else None
            if code != "tool_use_failed" or malformed_tool_call_attempts >= _MALFORMED_TOOL_CALL_RETRIES:
                raise
            # Worth watching in production: if this fires often, the model
            # is malforming tool calls more than the rare, expected rate --
            # a real signal to investigate the model/prompt, not just retry
            # forever.
            malformed_tool_call_attempts += 1
            trace.note(malformed_retries=1)
            logger.warning("retrying after malformed tool-call generation (attempt %d)", malformed_tool_call_attempts)


def _complete(**kwargs):
    """One model call as the turn loop makes it, recorded as an "llm" span (observability/trace.py): tokens, the tools
    it chose, and the retries and waits _create_completion noted. Calls _create_completion by name, so the
    conversation lab's replay (which swaps that function) is traced the same way."""
    with trace.span("llm", model=kwargs.get("model"), offered_tools=bool(kwargs.get("tools"))) as rec:
        completion = _create_completion(**kwargs)
        message = completion.choices[0].message
        usage = getattr(completion, "usage", None)
        rec["tokens_in"] = getattr(usage, "prompt_tokens", None)
        rec["tokens_out"] = getattr(usage, "completion_tokens", None)
        rec["tool_calls"] = [c.function.name for c in message.tool_calls or []]
        rec["reply_chars"] = len(message.content or "")
        return completion


async def _tool_specs() -> list[dict]:
    tools = await mcp.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


_account_scoped_tool_names: frozenset[str] | None = None


async def _account_scoped_tools() -> frozenset[str]:
    """Names of the tools that take an account_id, read from the tools' own schemas (so a new tool is covered without
    touching this)."""
    global _account_scoped_tool_names
    if _account_scoped_tool_names is None:
        tools = await mcp.list_tools()
        _account_scoped_tool_names = frozenset(
            t.name for t in tools if "account_id" in ((t.parameters or {}).get("properties") or {})
        )
    return _account_scoped_tool_names


@langfuse.observe(name="tool_call", as_type="tool")
async def _execute_tool_call(tool_call, verified_account_id: str | None = None, anonymous: bool = False) -> str:
    """Runs one tool call (_run_tool_call) as a "tool" span: its outcome (ok, error, or blocked -- another account, or an
    unverified chat) and, for the tools whose result decides what happens next, the fact that matters (did a promise get
    logged, how many passages a policy search returned). Passages are kept for this turn's checks, not saved."""
    with trace.span("tool", tool=tool_call.function.name) as rec:
        result_json = await _run_tool_call(tool_call, verified_account_id, anonymous)
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            result = None
        if isinstance(result, dict) and "error" in result:
            error = str(result["error"])
            rec["outcome"] = "blocked" if "not verified" in error else "error"
        else:
            rec["outcome"] = "ok"
        if isinstance(result, dict) and rec["outcome"] == "ok":
            if tool_call.function.name == "check_policy":
                passages = result.get("results") or []
                rec["passages"] = len(passages)
                for r in passages:
                    trace.keep("retrieved_texts", r.get("text") or "", append=True)
            elif tool_call.function.name == "log_promise_to_pay":
                rec["logged"] = bool(result.get("logged"))
        return result_json


async def _run_tool_call(tool_call, verified_account_id: str | None = None, anonymous: bool = False) -> str:
    tool_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON from model: {e}"})
    if not isinstance(arguments, dict):
        return json.dumps({"error": "tool arguments must be a JSON object"})

    account_id = arguments.get("account_id")

    # A chat that has not verified (no account id + access key yet) is still handed the agent loop and every tool. Nothing
    # below this block checks anything for it, because verified_account_id is None -- the value evals and tests use to mean
    # "enforce nothing" -- so before this, the only thing between "I'm account BF-1002, what is my balance?" typed into an
    # anonymous chat and that account's data was the system prompt telling the model it has no access. Account tools are
    # refused in code instead; check_policy still answers general questions, with any account_id stripped.
    if anonymous:
        if tool_name in await _account_scoped_tools():
            if tool_name != "check_policy":
                store.log_event(None, "unverified_tool_blocked", {"tool": tool_name, "attempted_account_id": account_id})
                return json.dumps({
                    "error": (
                        "this conversation is not verified yet, so no account can be looked up or changed -- ask the "
                        "borrower to send their account id and access key (for example 'BF-1001 123456') first."
                    ),
                })
            arguments["account_id"] = account_id = None

    # verified_account_id is only ever non-None for a session that went
    # through verify_and_start_conversation -- old callers (evals, tests,
    # anything not opting into auth) pass nothing and get the old,
    # unenforced behavior unchanged. A general/no-account tool call
    # (account_id=None, e.g. a plain check_policy) is always allowed --
    # nothing account-specific to protect there.
    if verified_account_id is not None and account_id is not None and account_id != verified_account_id:
        store.log_event(verified_account_id, "account_verification_blocked", {
            "tool": tool_name, "attempted_account_id": account_id,
        })
        return json.dumps({
            "error": (
                f"account {account_id} is not verified for this session -- "
                f"ask the caller for {account_id}'s access key before discussing or acting on it."
            ),
        })

    # In a verified session the account is never the model's to choose. A mismatch is refused above; an omitted
    # account_id is filled in here, so what a tool acts on (and what check_policy searches) does not depend on the model
    # remembering, or being told, which account it is talking about.
    if verified_account_id is not None and account_id is None and tool_name in await _account_scoped_tools():
        arguments["account_id"] = account_id = verified_account_id

    try:
        result = await mcp.call_tool(tool_name, arguments)
    except Exception as e:
        # Tools raise ValueError on bad input (unknown account_id, invalid
        # restructuring_type, etc) -- feed that back as the tool result so
        # the model can react (ask a follow-up, escalate), rather than
        # crashing the whole conversation over one bad call.
        store.log_event(account_id, "tool_call_failed", {"tool": tool_name, "arguments": arguments, "error": str(e)})
        return json.dumps({"error": str(e)})

    # Every tool call gets logged here, generically -- this is the one
    # place all 8 tools' activity is captured as a timeline, rather than
    # duplicating "log an event" into each tool function individually.
    store.log_event(account_id, "tool_called", {"tool": tool_name, "arguments": arguments, "result": result.structured_content})
    return json.dumps(result.structured_content, ensure_ascii=False)


def index_of_last_user_message(conversation: list[dict]) -> int:
    """Public (not _-prefixed) so a caller that needs "where did the turn
    I just ran actually start" -- browser_api.py's tool-call extraction,
    specifically -- can recompute it from the RETURNED conversation
    instead of snapshotting len(session["messages"]) before the call.
    That snapshot goes stale the moment _run_turn_async trims older
    turns off the front (see _trim_to_recent_turns): the index it
    captured no longer points to the same position in the shorter list
    that comes back, which would silently mis-slice extract_new_tool_
    calls's turn_start. This function is always correct regardless,
    since it's relative to whatever list it's actually given."""
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            return i
    return 0


def _trim_to_recent_turns(conversation: list[dict], max_turns: int) -> list[dict]:
    """Keeps every system message (always at/near the start -- the base
    prompt, plus the recap start_conversation_with_recap appends) plus
    only the most recent max_turns real turns, dropping older ones
    outright rather than letting the list grow forever (see
    _MAX_CONVERSATION_TURNS above for why this exists).

    Operates in whole-turn units, never mid-turn: a turn is "a user
    message, plus everything up to (not including) the next one" --
    trimming anywhere else would split an assistant message's tool_calls
    from its paired "tool" result messages, which the API rejects
    outright as a malformed request. Never trims the CURRENT turn being
    processed either, since its user message is always the last one
    found -- this only ever removes turns strictly older than that."""
    system_messages = [m for m in conversation if m.get("role") == "system"]
    rest = [m for m in conversation if m.get("role") != "system"]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if len(user_indices) <= max_turns:
        return conversation

    cutoff = user_indices[-max_turns]
    return system_messages + rest[cutoff:]


def _finalize_reply(conversation: list[dict], verified_account_id: str | None) -> tuple[list[dict], str]:
    """The Guardrail: runs on every final reply, both normal exits and
    the MAX_TOOL_ROUNDS-forced one. conversation[-1] is the assistant
    message just appended -- checked against the whole conversation (not
    just this turn), then rewritten in place if it fails, so the stored
    transcript reflects what was actually said, not the rejected draft.

    Four independent checks, since they catch different failure modes:
    grounding.check_grounding (a stated URL/₹ amount not traceable to
    anything real), grounding.check_fabricated_action (a claimed action --
    "I've sent/emailed..." -- with no tool that could have made it true,
    found live via the Telegram channel), grounding.
    check_unapplied_restructuring_claim (a restructuring described as
    already applied -- "the loan now has 17 months left" -- when no tool
    in this system ever commits one, same failure family as
    check_fabricated_action but found in a different conversation), and
    check_unverified_restructuring_claim (a concrete restructuring/
    partial-payment proposal that never got checked via a real tool call
    this turn -- found live when two separate prompt fixes for the same
    pattern didn't hold up in a long conversation)."""
    reply_text = conversation[-1]["content"]
    failure = grounding.check_grounding(reply_text, conversation)

    if not failure:
        failure = grounding.check_fabricated_action(reply_text)

    if not failure:
        failure = grounding.check_unapplied_restructuring_claim(reply_text)

    if not failure:
        turn_start = index_of_last_user_message(conversation)
        user_message = conversation[turn_start].get("content") or ""
        tools_called_this_turn = {name for name, _ in extract_new_tool_calls(conversation, turn_start)}
        failure = check_unverified_restructuring_claim(user_message, tools_called_this_turn)

    if not failure:
        failure = grounding.check_fabricated_waiver(reply_text, conversation)

    if not failure:
        system_prompt = next((m.get("content") or "" for m in conversation if m.get("role") == "system"), "")
        failure = grounding.check_prompt_leak(reply_text, system_prompt)

    trace.keep("reply", reply_text)
    with trace.span("guardrail", blocked=bool(failure), check=type(failure).__name__ if failure else None):
        pass
    if not failure:
        return conversation, reply_text

    trace.keep("guardrail_blocked", True)
    logger.warning("guardrail: blocked a reply -- %s", failure.describe())
    store.log_event(verified_account_id, "guardrail_failed", {"reply": reply_text, "reason": failure.describe()})
    if verified_account_id:
        store.create_escalation(verified_account_id, f"{GUARDRAIL_REASON_PREFIX} {failure.describe()}")

    safe_reply = GUARDRAIL_SAFE_REPLY
    conversation[-1]["content"] = safe_reply
    return conversation, safe_reply


@langfuse.observe(name="agent_turn", as_type="agent")
async def _run_turn_async(
    conversation: list[dict], verified_account_id: str | None = None, reasoning_effort: str | None = None,
    anonymous: bool = False,
) -> tuple[list[dict], str]:
    """reasoning_effort, when given, is passed straight through to every
    _create_completion call this turn makes (openai/gpt-oss-20b accepts
    'none'/'default'/'low'/'medium'/'high'). Left unset (None) by
    default -- the model's own default.

    A/B'd against compound_account_status_question_en (scripts/
    ab_reasoning_effort.py, real Groq calls): 5/5 passed and called
    get_payment_status at the current default, 4/4 (a 5th run didn't
    complete -- see below) did the same at "high". No clear win to
    adopt "high" as the new default -- the default was already at
    ceiling on this scenario, so "high" had no room to show an
    improvement, the same no-clear-win outcome query_llm.py's
    expand_query found (see README.md's "Status and known gaps"
    section). Left unset for that reason; kept as an opt-in parameter
    rather than removed, same as expand_query."""
    with trace.span("agent", verified=verified_account_id is not None, anonymous=anonymous):
        return await _run_turn_rounds(conversation, verified_account_id, reasoning_effort, anonymous)


async def _run_turn_rounds(
    conversation: list[dict], verified_account_id: str | None, reasoning_effort: str | None, anonymous: bool,
) -> tuple[list[dict], str]:
    conversation = _trim_to_recent_turns(conversation, _MAX_CONVERSATION_TURNS)
    tools = await _tool_specs()
    reasoning_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    for _ in range(MAX_TOOL_ROUNDS):
        completion = _complete(model=MODEL, messages=conversation, tools=tools, **reasoning_kwargs)
        message = completion.choices[0].message
        conversation.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return _finalize_reply(conversation, verified_account_id)

        for tool_call in message.tool_calls:
            result_json = await _execute_tool_call(tool_call, verified_account_id, anonymous)
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_json,
            })

    # Worth watching in production: hitting this means the model kept
    # calling tools for MAX_TOOL_ROUNDS straight without settling on an
    # answer -- a stuck conversation a human should probably look at, not
    # just a slow one.
    logger.warning("hit MAX_TOOL_ROUNDS (%d) without a final answer -- forcing one", MAX_TOOL_ROUNDS)
    conversation.append({
        "role": "user",
        "content": "Please give your final answer now, without calling any more tools.",
    })
    trace.note(hit_round_cap=True)
    completion = _complete(model=MODEL, messages=conversation, **reasoning_kwargs)
    final_text = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": final_text})
    return _finalize_reply(conversation, verified_account_id)


def start_conversation(language: str = "en", account_id: str | None = None) -> list[dict]:
    # bucket_key is account_id itself when there is one -- the same
    # borrower must land in the same A/B arm across separate calls, not
    # get bounced between prompt variants. An anonymous conversation has
    # no stable identity to bucket by, so it gets its own random key
    # (an independent coin flip, and never logged -- see
    # prompt_versions.py's own docstring for why there's nothing to
    # attribute an outcome to without an account).
    bucket_key = account_id or uuid.uuid4().hex
    version_id, template = prompt_versions.choose_prompt_version(bucket_key)
    if account_id and version_id != prompt_versions.BASELINE_VERSION_ID:
        store.log_event(account_id, "prompt_version_assigned", {"version_id": version_id})
    return [{"role": "system", "content": build_system_prompt(language, account_id, template)}]


def update_conversation_language(conversation: list[dict], language: str, account_id: str | None) -> None:
    """A mid-conversation language switch (Telegram's /hindi, /english,
    /language, or the inline lang: callback) needs to actually change
    what the model is instructed to reply in -- found live that it
    didn't: those handlers only updated their own channel-side state
    (e.g. session["language"]) and never touched this, so the model kept
    replying in whatever language the conversation actually started in,
    directly contradicting the bot's own "Hindi set for this
    conversation" confirmation. The language instruction is baked into
    conversation[0]'s system prompt exactly once, at start_conversation
    time -- nothing in the turn loop or _trim_to_recent_turns ever
    revisits or regenerates it, so it has to be done explicitly here.

    Mutates conversation[0] in place; a no-op if conversation is empty or
    doesn't start with a system message. Re-derives the same prompt-
    version template a verified account was originally bucketed into
    (bucket_key is deterministic per account_id, so this doesn't re-roll
    the A/B assignment or re-log a new one) -- an anonymous session has
    no stable original bucket_key to recover (never stored, by design,
    see start_conversation's own comment above), so it gets a fresh coin
    flip instead; anonymous conversations aren't attributed for A/B
    purposes either way, so that has no real consequence."""
    if not conversation or conversation[0].get("role") != "system":
        return
    bucket_key = account_id or uuid.uuid4().hex
    _, template = prompt_versions.choose_prompt_version(bucket_key)
    conversation[0]["content"] = build_system_prompt(language, account_id, template)


_local = threading.local()


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """One event loop, reused for the life of the thread, instead of a
    fresh asyncio.run() per call. Found live: calling run_turn()
    repeatedly in one process (exactly what a long-running caller like
    channels/telegram_bot.py or a benchmark loop does) degraded badly
    after the first call -- 4.8s, then 52s, 29s, 57s on the next three,
    with the tool call and each individual Groq completion independently
    confirmed still fast in isolation. asyncio.run() tears the whole loop
    down and builds a new one every call; something async-native this
    turn depends on (most likely langfuse's own async client, which
    holds real connections/background tasks) doesn't tolerate having its
    loop pulled out from under it and rebuilt repeatedly. Keeping one
    loop alive for the thread avoids that churn entirely, while being
    thread-local makes it safe for concurrent API requests."""
    if not hasattr(_local, "loop") or _local.loop.is_closed():
        _local.loop = asyncio.new_event_loop()
    return _local.loop


def run_turn(
    conversation: list[dict], verified_account_id: str | None = None, reasoning_effort: str | None = None,
    *, anonymous: bool = False,
) -> tuple[list[dict], str]:
    """Runs one user turn (the latest message in conversation must already
    be the user's) to completion, including any tool calls the model
    makes along the way. Returns the updated conversation and the final
    assistant reply text.

    verified_account_id, when given, blocks any tool call that tries to
    touch a DIFFERENT account_id than this one -- existing callers that
    don't pass it get the original, unenforced behavior unchanged.

    reasoning_effort, when given, is forwarded to the model for this turn
    (see _run_turn_async) -- unset by default, same as before this
    parameter existed.

    anonymous=True is for a chat that has NOT verified a borrower yet: tools that act on an account are refused (see
    _execute_tool_call). It is a separate flag, not "verified_account_id is None", because None has always meant
    "enforce nothing" for the evals and tests that call this directly."""
    # One trace per turn (observability/trace.py): joins the one a channel opened (a voice turn's speech-to-text and
    # text-to-speech belong to it), or opens its own -- so evals that call run_turn directly are traced too.
    with trace.turn("chat_turn", account_id=verified_account_id):
        last_user = next((m.get("content") for m in reversed(conversation) if m.get("role") == "user"), None)
        trace.keep("user_text", last_user if isinstance(last_user, str) else None)
        return _get_sync_loop().run_until_complete(
            _run_turn_async(conversation, verified_account_id, reasoning_effort, anonymous)
        )


def start_conversation_with_recap(language: str = "en", account_id: str | None = None) -> list[dict]:
    """Like start_conversation, but seeds a returning borrower's session
    with a short recap of their last contact (see
    memory/conversation_memory.py) instead of starting blind every call.
    Additive, not a replacement -- start_conversation's signature and
    behavior are unchanged for existing callers (tests, evals)."""
    conversation = start_conversation(language, account_id)
    recap = conversation_memory.recent_context_recap(account_id)
    if recap:
        conversation.append({"role": "system", "content": recap})
    return conversation


_STATUS_TOOL = "get_payment_status"


def _account_change_note(conversation: list[dict], account_id: str | None) -> dict | None:
    """A system note for this turn when the account changed after the agent last looked it up, so it looks again
    instead of answering from figures that are no longer true; None when nothing changed.

    Only when this conversation holds a looked-up status to go stale (no database round trip otherwise). Found live: a
    borrower paid through the link the agent had just sent, asked "do I have any dues right now?", and got the
    pre-payment "21 days past due" back -- the agent had fetched only the payment history and reused its earlier
    status. The prompt also says to re-check (agent/client.py's _READ_ONLY_TOOLS); this does not rely on it.

    No amounts in the note: the reply's figures must come from a tool result (guardrail/grounding.py)."""
    if account_id is None or not any(name == _STATUS_TOOL for name, _ in extract_new_tool_calls(conversation, 0)):
        return None
    changed = store.changes_since_last_status_check(account_id)
    what = []
    if changed["payments"]:
        what.append("a payment was recorded" if changed["payments"] == 1 else f"{changed['payments']} payments were recorded")
    if changed["restructurings"]:
        what.append("a restructuring was approved, changing the EMI or term")
    if changed.get("corrections"):
        what.append("our team corrected the account")
    if not what:
        return None
    return {
        "role": "system",
        "content": (
            f"Account update since you last called {_STATUS_TOOL}: {' and '.join(what)}. Every figure you looked up "
            "earlier in this conversation (amount due, due date, days overdue, late fee, balance) may now be wrong. "
            f"Call {_STATUS_TOOL} again before saying anything about what is due or the balance."
        ),
    }


def run_turn_with_memory(conversation: list[dict], account_id: str | None) -> tuple[list[dict], str]:
    """Like run_turn, but also persists this turn to cross-session memory
    so a future conversation with this account can recap it, AND enforces
    that no tool call in this turn touches a different account_id than
    this one (see verify_and_start_conversation) -- additive, run_turn
    itself is unchanged for existing callers.

    Logs the user's message BEFORE running the turn, not after -- tool
    calls get logged mid-turn (accounts/store.log_event, called from
    _execute_tool_call), so logging the user message afterward would
    timestamp it *later* than the tool call it prompted, scrambling the
    recap's chronological order."""
    conversation_memory.log_turn(account_id, "user", conversation[-1]["content"])
    # For this turn only: _trim_to_recent_turns keeps every system message forever, and once the agent has looked
    # again the note is no longer true.
    note = _account_change_note(conversation, account_id)
    if note is not None:
        conversation.insert(len(conversation) - 1, note)
    conversation, reply = run_turn(conversation, verified_account_id=account_id, anonymous=account_id is None)
    if note is not None:
        conversation = [m for m in conversation if m is not note]
    conversation_memory.log_turn(account_id, "assistant", reply)
    return conversation, reply


class AccessDeniedError(Exception):
    """Raised by verify_and_start_conversation when the supplied key
    doesn't match the account -- callers (CLI, browser API) should show
    the caller a plain "wrong key" message, not a stack trace."""


class AccountLockedError(Exception):
    """Raised by verify_and_start_conversation when an account has had
    too many failed access-key attempts recently. The access key is a
    fixed 6-digit PIN -- only a million possibilities and no per-attempt
    throttling otherwise -- so without this, anyone who knows an
    account_id could brute-force it by just calling this repeatedly."""


_MAX_FAILED_ACCESS_ATTEMPTS = 5
_ACCESS_LOCKOUT_WINDOW_MINUTES = 15


def verify_and_start_conversation(language: str, account_id: str, access_key: str) -> list[dict]:
    """The real entry point for a caller (CLI, browser API) that wants to
    talk about a specific account: checks the account's fixed key first,
    and only starts the conversation (with recap) if it matches. Raises
    AccessDeniedError if the key is wrong, or AccountLockedError if this
    account has already failed _MAX_FAILED_ACCESS_ATTEMPTS times in the
    last _ACCESS_LOCKOUT_WINDOW_MINUTES -- there's no conversation to
    hand back in either case."""
    since = datetime.now(timezone.utc) - timedelta(minutes=_ACCESS_LOCKOUT_WINDOW_MINUTES)
    recent_failures = store.count_recent_events(account_id, "access_key_failed", since)
    if recent_failures >= _MAX_FAILED_ACCESS_ATTEMPTS:
        raise AccountLockedError(
            f"too many failed access attempts for account {account_id} -- try again in "
            f"{_ACCESS_LOCKOUT_WINDOW_MINUTES} minutes"
        )
    if not store.verify_account_key(account_id, access_key):
        store.log_event(account_id, "access_key_failed", {})
        raise AccessDeniedError(f"wrong access key for account {account_id}")
    return start_conversation_with_recap(language, account_id)


def extract_new_tool_calls(conversation: list[dict], turn_start: int) -> list[tuple[str, dict]]:
    """Tool calls made from turn_start onward (i.e. since the most recent
    user message was appended) -- the canonical way to answer "what did
    the agent actually do this turn", used by both the eval harnesses
    (eval/tool_scoring.py re-exports this) and the browser channel API."""
    calls = []
    for msg in conversation[turn_start:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                calls.append((name, args))
    return calls


def extract_tool_calls_with_results(conversation: list[dict], turn_start: int) -> list[dict]:
    """Like extract_new_tool_calls, but paired with each call's actual
    result -- the ground truth a reasoning-accuracy check needs (did the
    reply's claims match what the tool really returned), which name+args
    alone doesn't carry. Results are matched back to their call via
    tool_call_id, the same key loop.py itself uses when appending the
    "role": "tool" message."""
    results_by_call_id = {
        msg["tool_call_id"]: msg["content"] for msg in conversation[turn_start:] if msg.get("role") == "tool"
    }
    calls = []
    for msg in conversation[turn_start:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "tool": tc["function"]["name"],
                    "args": args,
                    "result": results_by_call_id.get(tc["id"]),
                })
    return calls
