"""Tests for agent/loop.py's conversation-history trimming -- pure logic,
no Groq/Postgres needed (unlike test_agent_loop.py's real-API tests,
gated behind GROQ_API_KEY/DATABASE_URL). See _trim_to_recent_turns's own
docstring for the real bug this closes: a long-lived, never-/reset
session resent its entire history on every completion call within a
turn, which is what actually burst a real conversation past Groq's
per-minute token cap live, not sustained daily exhaustion.
"""

from businessflow.agent.loop import _trim_to_recent_turns, index_of_last_user_message


def _turn(n: int, with_tool_call: bool = False) -> list[dict]:
    """One user message plus its reply, optionally with a tool_calls/
    tool-result pair in between -- the atomic unit _trim_to_recent_turns
    must never split."""
    messages = [{"role": "user", "content": f"user msg {n}"}]
    if with_tool_call:
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"tc{n}", "function": {"name": "get_payment_status", "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"tc{n}", "content": "{}"})
    messages.append({"role": "assistant", "content": f"reply {n}"})
    return messages


def test_trim_leaves_a_conversation_under_the_limit_untouched():
    conversation = [{"role": "system", "content": "base prompt"}] + [m for n in range(5) for m in _turn(n)]

    assert _trim_to_recent_turns(conversation, max_turns=20) == conversation


def test_trim_keeps_every_system_message_and_only_the_most_recent_turns():
    system_messages = [{"role": "system", "content": "base prompt"}, {"role": "system", "content": "recap"}]
    conversation = system_messages + [m for n in range(25) for m in _turn(n)]

    trimmed = _trim_to_recent_turns(conversation, max_turns=20)

    assert trimmed[0] == system_messages[0]
    assert trimmed[1] == system_messages[1]
    user_contents = [m["content"] for m in trimmed if m["role"] == "user"]
    assert user_contents == [f"user msg {n}" for n in range(5, 25)]  # the last 20, oldest 5 dropped


def test_trim_never_orphans_a_tool_result_from_a_turn_that_survives():
    # The tool call/result pair lands in turn 5, which is inside the kept
    # window when trimming to the last 20 of 25 turns -- must survive
    # together, not get separated (a "tool" message with no preceding
    # tool_calls message is a malformed request the real API rejects).
    conversation = [{"role": "system", "content": "base prompt"}]
    conversation += [m for n in range(25) for m in _turn(n, with_tool_call=(n == 5))]

    trimmed = _trim_to_recent_turns(conversation, max_turns=20)

    tool_messages = [m for m in trimmed if m["role"] == "tool"]
    assert len(tool_messages) == 1
    calling_message = next(m for m in trimmed if m.get("role") == "assistant" and m.get("tool_calls"))
    assert calling_message["tool_calls"][0]["id"] == tool_messages[0]["tool_call_id"]


def test_trim_drops_a_tool_call_pair_entirely_when_its_whole_turn_is_dropped():
    # The tool call/result pair lands in turn 2, which falls OUTSIDE the
    # kept window when trimming to the last 5 of 10 turns -- must be
    # dropped as a whole unit, never left half-orphaned.
    conversation = [{"role": "system", "content": "base prompt"}]
    conversation += [m for n in range(10) for m in _turn(n, with_tool_call=(n == 2))]

    trimmed = _trim_to_recent_turns(conversation, max_turns=5)

    assert not any(m["role"] == "tool" for m in trimmed)
    assert not any(m.get("role") == "assistant" and m.get("tool_calls") for m in trimmed)
    user_contents = [m["content"] for m in trimmed if m["role"] == "user"]
    assert user_contents == [f"user msg {n}" for n in range(5, 10)]


def test_index_of_last_user_message_is_correct_on_a_trimmed_conversation():
    # The real reason this function needed to become public: a caller
    # that snapshots len(session["messages"]) BEFORE calling run_turn
    # gets a stale index once trimming shortens the list that comes
    # back -- this must be recomputed from the returned conversation,
    # which stays correct regardless of how many older turns were cut.
    conversation = [{"role": "system", "content": "base prompt"}]
    conversation += [m for n in range(25) for m in _turn(n)]
    trimmed = _trim_to_recent_turns(conversation, max_turns=20)

    assert trimmed[index_of_last_user_message(trimmed)]["content"] == "user msg 24"


def test_index_of_last_user_message_falls_back_to_zero_with_no_user_message():
    assert index_of_last_user_message([{"role": "system", "content": "base prompt"}]) == 0
