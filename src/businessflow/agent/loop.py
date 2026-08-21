"""The real agent: a multi-turn, tool-calling loop over Groq. This is what
actually connects the reasoning model to the 8 real tools (including
check_policy, which is the RAG retriever) -- replacing client.reply(),
which just talks without checking anything against real data.
"""

import asyncio
import json
import logging

import groq
import langfuse

from businessflow.accounts import store
from businessflow.agent.client import MODEL, build_system_prompt, client, switch_to_fallback_key
from businessflow.memory import conversation_memory
from businessflow.tools import mcp

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5  # a hard cap -- never loop on tool calls forever

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
    switched_this_call = False
    for attempt in range(_MALFORMED_TOOL_CALL_RETRIES + 1):
        groq_client = client()  # re-fetched each attempt -- picks up the fallback key transparently once switched
        try:
            return groq_client.chat.completions.create(**kwargs)
        except groq.RateLimitError:
            # The primary key's daily quota is exhausted (this is exactly
            # what we hit repeatedly during eval runs this session) --
            # switch to the fallback key once, not part of the
            # malformed-tool-call retry budget below.
            if switched_this_call or not switch_to_fallback_key():
                raise
            switched_this_call = True
            logger.warning("primary Groq key rate-limited -- switched to the fallback key")
        except groq.BadRequestError as e:
            code = (e.body or {}).get("error", {}).get("code") if isinstance(e.body, dict) else None
            if code != "tool_use_failed" or attempt == _MALFORMED_TOOL_CALL_RETRIES:
                raise
            # Worth watching in production: if this fires often, the model
            # is malforming tool calls more than the rare, expected rate --
            # a real signal to investigate the model/prompt, not just retry
            # forever.
            logger.warning("retrying after malformed tool-call generation (attempt %d)", attempt + 1)


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
async def _execute_tool_call(tool_call) -> str:
    tool_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON from model: {e}"})

    account_id = arguments.get("account_id")

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


@langfuse.observe(name="agent_turn", as_type="agent")
async def _run_turn_async(conversation: list[dict]) -> tuple[list[dict], str]:
    tools = await _tool_specs()

    for _ in range(MAX_TOOL_ROUNDS):
        completion = _create_completion(model=MODEL, messages=conversation, tools=tools)
        message = completion.choices[0].message
        conversation.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return conversation, message.content

        for tool_call in message.tool_calls:
            result_json = await _execute_tool_call(tool_call)
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
    completion = _create_completion(model=MODEL, messages=conversation)
    final_text = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": final_text})
    return conversation, final_text


def start_conversation(language: str = "en", account_id: str | None = None) -> list[dict]:
    return [{"role": "system", "content": build_system_prompt(language, account_id)}]


def run_turn(conversation: list[dict]) -> tuple[list[dict], str]:
    """Runs one user turn (the latest message in conversation must already
    be the user's) to completion, including any tool calls the model
    makes along the way. Returns the updated conversation and the final
    assistant reply text."""
    return asyncio.run(_run_turn_async(conversation))


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
    so a future conversation with this account can recap it. Additive --
    run_turn itself is unchanged for existing callers.

    Logs the user's message BEFORE running the turn, not after -- tool
    calls get logged mid-turn (accounts/store.log_event, called from
    _execute_tool_call), so logging the user message afterward would
    timestamp it *later* than the tool call it prompted, scrambling the
    recap's chronological order."""
    conversation_memory.log_turn(account_id, "user", conversation[-1]["content"])
    conversation, reply = run_turn(conversation)
    conversation_memory.log_turn(account_id, "assistant", reply)
    return conversation, reply


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
