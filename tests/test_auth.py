"""Unit tests for account-key verification: the store-level key check,
the verify_and_start_conversation entry point, and the tool-call
enforcement that blocks a mismatched account_id mid-conversation. Real
Postgres, no LLM -- _execute_tool_call is exercised directly with a
minimal stand-in tool-call object (just the two attributes it actually
reads), not a mock of Groq or of any tool's own logic.
"""

import asyncio
import json
import os

import pytest

from businessflow.accounts import store
from businessflow.agent.loop import (
    AccessDeniedError,
    AccountLockedError,
    _execute_tool_call,
    verify_and_start_conversation,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


class _FakeFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict):
        self.function = _FakeFunction(name, arguments)
        self.id = "fake-tool-call-id"


def test_verify_account_key_accepts_the_real_seeded_key(reseed_accounts):
    assert store.verify_account_key("BF-1001", "482913") is True


def test_verify_account_key_rejects_wrong_key(reseed_accounts):
    assert store.verify_account_key("BF-1001", "000000") is False


def test_verify_account_key_rejects_unknown_account(reseed_accounts):
    assert store.verify_account_key("BF-9999", "482913") is False


def test_verify_and_start_conversation_succeeds_with_the_right_key(reseed_accounts):
    conversation = verify_and_start_conversation("en", "BF-1001", "482913")

    assert conversation[0]["role"] == "system"
    assert "BF-1001" in conversation[0]["content"]


def test_verify_and_start_conversation_raises_on_wrong_key(reseed_accounts):
    with pytest.raises(AccessDeniedError):
        verify_and_start_conversation("en", "BF-1001", "000000")


def test_verify_and_start_conversation_locks_out_after_repeated_wrong_keys(reseed_accounts):
    # The access key is a fixed 6-digit PIN with no other throttling --
    # without a lockout, an attacker who knows an account_id could just
    # try all million combinations against this one entry point.
    for _ in range(5):
        with pytest.raises(AccessDeniedError):
            verify_and_start_conversation("en", "BF-1001", "000000")

    with pytest.raises(AccountLockedError):
        verify_and_start_conversation("en", "BF-1001", "482913")  # even the REAL key is now blocked


def test_verify_and_start_conversation_does_not_lock_out_before_the_threshold(reseed_accounts):
    for _ in range(4):
        with pytest.raises(AccessDeniedError):
            verify_and_start_conversation("en", "BF-1001", "000000")

    conversation = verify_and_start_conversation("en", "BF-1001", "482913")
    assert conversation[0]["role"] == "system"


def test_lockout_is_scoped_to_the_specific_account(reseed_accounts):
    # Failed attempts against one account must not lock out a different
    # one -- a shared global counter would let one attacker's noise deny
    # service to every other borrower.
    for _ in range(5):
        with pytest.raises(AccessDeniedError):
            verify_and_start_conversation("en", "BF-1001", "000000")

    conversation = verify_and_start_conversation("en", "BF-1002", "716044")
    assert conversation[0]["role"] == "system"


def test_tool_call_blocked_when_account_id_does_not_match_verified_session(reseed_accounts):
    tool_call = _FakeToolCall("get_payment_status", {"account_id": "BF-1002"})

    result_json = asyncio.run(_execute_tool_call(tool_call, verified_account_id="BF-1001"))
    result = json.loads(result_json)

    assert "error" in result
    assert "BF-1002" in result["error"]
    assert "not verified" in result["error"]


def test_tool_call_allowed_when_account_id_matches_verified_session(reseed_accounts):
    tool_call = _FakeToolCall("get_payment_status", {"account_id": "BF-1001"})

    result_json = asyncio.run(_execute_tool_call(tool_call, verified_account_id="BF-1001"))
    result = json.loads(result_json)

    assert result.get("account_id") == "BF-1001"
    assert "error" not in result


def test_tool_call_unenforced_when_no_verified_account_id_given(reseed_accounts):
    # Backward compatibility: callers that don't pass verified_account_id
    # at all (evals, existing tests) get the original, unenforced
    # behavior -- this is what keeps 11+5 existing eval scenarios and 30+
    # unit tests from breaking when this feature was added.
    tool_call = _FakeToolCall("get_payment_status", {"account_id": "BF-1002"})

    result_json = asyncio.run(_execute_tool_call(tool_call))
    result = json.loads(result_json)

    assert result.get("account_id") == "BF-1002"
    assert "error" not in result


def test_general_tool_call_with_no_account_id_is_never_blocked(reseed_accounts):
    # check_policy with no account_id -- nothing account-specific to
    # protect, so this must go through even inside a verified session.
    tool_call = _FakeToolCall("check_policy", {"query": "what is the grace period"})

    result_json = asyncio.run(_execute_tool_call(tool_call, verified_account_id="BF-1001"))
    result = json.loads(result_json)

    assert "error" not in result
