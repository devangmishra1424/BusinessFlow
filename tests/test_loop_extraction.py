"""Unit tests for the "what actually happened this turn" extraction
helpers in agent/loop.py -- pure logic over a hand-built conversation
list, no LLM or DB needed. Neither function had any test coverage
before this, despite being the canonical way both eval harnesses and
the browser API answer "what did the agent do."
"""

import json

from businessflow.agent.loop import extract_new_tool_calls, extract_tool_calls_with_results


def _assistant_tool_call_msg(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}],
    }


def _tool_result_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestExtractNewToolCalls:
    def test_extracts_name_and_parsed_args(self):
        conversation = [
            {"role": "user", "content": "what's my balance"},
            _assistant_tool_call_msg("call_1", "get_payment_status", {"account_id": "BF-1001"}),
            _tool_result_msg("call_1", '{"account_id": "BF-1001", "emi_amount": 12500}'),
        ]

        calls = extract_new_tool_calls(conversation, turn_start=0)

        assert calls == [("get_payment_status", {"account_id": "BF-1001"})]

    def test_only_looks_from_turn_start_onward(self):
        conversation = [
            {"role": "user", "content": "turn 1"},
            _assistant_tool_call_msg("call_1", "get_payment_status", {"account_id": "BF-1001"}),
            _tool_result_msg("call_1", "{}"),
            {"role": "assistant", "content": "here's your balance"},
            {"role": "user", "content": "turn 2"},
        ]
        turn_2_start = len(conversation)
        conversation.append(_assistant_tool_call_msg("call_2", "escalate_to_human", {"account_id": "BF-1001"}))

        calls = extract_new_tool_calls(conversation, turn_start=turn_2_start)

        assert calls == [("escalate_to_human", {"account_id": "BF-1001"})]

    def test_multiple_tool_calls_in_one_assistant_message(self):
        conversation = [{
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "get_payment_status", "arguments": '{"account_id": "BF-1001"}'}},
                {"id": "b", "function": {"name": "check_policy", "arguments": '{"query": "grace period"}'}},
            ],
        }]

        calls = extract_new_tool_calls(conversation, turn_start=0)

        assert calls == [
            ("get_payment_status", {"account_id": "BF-1001"}),
            ("check_policy", {"query": "grace period"}),
        ]

    def test_malformed_arguments_json_becomes_empty_dict_not_a_crash(self):
        conversation = [{
            "role": "assistant",
            "tool_calls": [{"id": "a", "function": {"name": "get_payment_status", "arguments": "{not valid json"}}],
        }]

        calls = extract_new_tool_calls(conversation, turn_start=0)

        assert calls == [("get_payment_status", {})]

    def test_no_tool_calls_returns_empty_list(self):
        conversation = [{"role": "assistant", "content": "just a plain reply, no tools"}]

        assert extract_new_tool_calls(conversation, turn_start=0) == []


class TestExtractToolCallsWithResults:
    def test_pairs_each_call_with_its_real_result_by_tool_call_id(self):
        conversation = [
            _assistant_tool_call_msg("call_1", "get_payment_status", {"account_id": "BF-1001"}),
            _tool_result_msg("call_1", '{"emi_amount": 12500.0, "days_past_due": 3}'),
        ]

        calls = extract_tool_calls_with_results(conversation, turn_start=0)

        assert calls == [{
            "tool": "get_payment_status",
            "args": {"account_id": "BF-1001"},
            "result": '{"emi_amount": 12500.0, "days_past_due": 3}',
        }]

    def test_results_are_matched_by_id_not_by_position(self):
        # Two calls in one assistant turn, with their results appended in
        # the SAME order below -- matching must key off tool_call_id, not
        # just "the next tool message in the list," or a reordering
        # anywhere in the pipeline would silently pair a call with the
        # wrong result.
        conversation = [{
            "role": "assistant",
            "tool_calls": [
                {"id": "call_a", "function": {"name": "get_payment_status", "arguments": '{"account_id": "BF-1001"}'}},
                {"id": "call_b", "function": {"name": "check_policy", "arguments": '{"query": "grace period"}'}},
            ],
        }, _tool_result_msg("call_b", '{"results": ["grace period doc"]}'),
           _tool_result_msg("call_a", '{"emi_amount": 12500.0}')]

        calls = extract_tool_calls_with_results(conversation, turn_start=0)

        by_tool = {c["tool"]: c["result"] for c in calls}
        assert by_tool["get_payment_status"] == '{"emi_amount": 12500.0}'
        assert by_tool["check_policy"] == '{"results": ["grace period doc"]}'

    def test_a_call_with_no_matching_result_yet_gets_none(self):
        conversation = [_assistant_tool_call_msg("call_1", "get_payment_status", {"account_id": "BF-1001"})]

        calls = extract_tool_calls_with_results(conversation, turn_start=0)

        assert calls == [{"tool": "get_payment_status", "args": {"account_id": "BF-1001"}, "result": None}]
