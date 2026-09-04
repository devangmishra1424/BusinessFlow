"""The real agent: a multi-turn, tool-calling loop over Groq. This is what
actually connects the reasoning model to the 8 real tools (including
check_policy, which is the RAG retriever) -- replacing client.reply(),
which just talks without checking anything against real data.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import groq
import langfuse

from businessflow.accounts import store
from businessflow.agent import prompt_versions
from businessflow.agent.client import MODEL, build_system_prompt, client, switch_to_fallback_key
from businessflow.guardrail import grounding
from businessflow.guardrail.unverified_restructuring import check_unverified_restructuring_claim
from businessflow.memory import conversation_memory
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


@langfuse.observe(name="groq_completion", as_type="generation")
def _create_completion(**kwargs):
    # Two independent retry budgets, deliberately not sharing one counter:
    # rate-limit switching is bounded by however many real fallback keys
    # are configured (switch_to_fallback_key() itself returns False once
    # they're all tried -- no separate cap needed here), while malformed-
    # tool-call retries are capped at _MALFORMED_TOOL_CALL_RETRIES
    # regardless of how many keys got switched through along the way.
    malformed_tool_call_attempts = 0
    while True:
        groq_client = client()  # re-fetched each attempt -- picks up whichever key is currently active
        try:
            return groq_client.chat.completions.create(**kwargs)
        except groq.RateLimitError:
            # This key's daily quota is exhausted (this is exactly what we
            # hit repeatedly during eval runs this session) -- advance to
            # the next configured fallback key, if any are left untried.
            if not switch_to_fallback_key():
                raise
            logger.warning("Groq key rate-limited -- switched to the next configured fallback key")
        except groq.BadRequestError as e:
            code = (e.body or {}).get("error", {}).get("code") if isinstance(e.body, dict) else None
            if code != "tool_use_failed" or malformed_tool_call_attempts >= _MALFORMED_TOOL_CALL_RETRIES:
                raise
            # Worth watching in production: if this fires often, the model
            # is malforming tool calls more than the rare, expected rate --
            # a real signal to investigate the model/prompt, not just retry
            # forever.
            malformed_tool_call_attempts += 1
            logger.warning("retrying after malformed tool-call generation (attempt %d)", malformed_tool_call_attempts)


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


@langfuse.observe(name="tool_call", as_type="tool")
async def _execute_tool_call(tool_call, verified_account_id: str | None = None) -> str:
    tool_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON from model: {e}"})

    account_id = arguments.get("account_id")

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
        return conversation, reply_text

    logger.warning("guardrail: blocked a reply -- %s", failure.describe())
    store.log_event(verified_account_id, "guardrail_failed", {"reply": reply_text, "reason": failure.describe()})
    if verified_account_id:
        store.create_escalation(verified_account_id, f"Guardrail blocked a reply: {failure.describe()}")

    safe_reply = "Let me connect you with someone who can confirm those exact details before we go further."
    conversation[-1]["content"] = safe_reply
    return conversation, safe_reply


@langfuse.observe(name="agent_turn", as_type="agent")
async def _run_turn_async(
    conversation: list[dict], verified_account_id: str | None = None, reasoning_effort: str | None = None,
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
    conversation = _trim_to_recent_turns(conversation, _MAX_CONVERSATION_TURNS)
    tools = await _tool_specs()
    reasoning_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    for _ in range(MAX_TOOL_ROUNDS):
        completion = _create_completion(model=MODEL, messages=conversation, tools=tools, **reasoning_kwargs)
        message = completion.choices[0].message
        conversation.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return _finalize_reply(conversation, verified_account_id)

        for tool_call in message.tool_calls:
            result_json = await _execute_tool_call(tool_call, verified_account_id)
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
    completion = _create_completion(model=MODEL, messages=conversation, **reasoning_kwargs)
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


_sync_loop: asyncio.AbstractEventLoop | None = None


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """One event loop, reused for the life of the process, instead of a
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
    loop alive for the whole process avoids that churn entirely."""
    global _sync_loop
    if _sync_loop is None or _sync_loop.is_closed():
        _sync_loop = asyncio.new_event_loop()
    return _sync_loop


def run_turn(
    conversation: list[dict], verified_account_id: str | None = None, reasoning_effort: str | None = None,
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
    parameter existed."""
    return _get_sync_loop().run_until_complete(_run_turn_async(conversation, verified_account_id, reasoning_effort))


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
    conversation, reply = run_turn(conversation, verified_account_id=account_id)
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
