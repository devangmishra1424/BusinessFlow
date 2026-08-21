"""Unit tests for cross-session memory -- built on the real events table,
real Postgres, no LLM involved."""

import os

import pytest

from businessflow.memory.conversation_memory import log_turn, recent_context_recap

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_log_turn_is_a_noop_without_an_account_id(reseed_accounts):
    log_turn(None, "user", "hello")  # must not raise, must not need an account to key to
    assert recent_context_recap(None) is None


def test_recap_is_none_with_no_prior_history(reseed_accounts):
    # reseed_accounts just cleared all events for the demo accounts.
    assert recent_context_recap("BF-1001") is None


def test_recap_builds_a_readable_digest_from_logged_turns(reseed_accounts):
    log_turn("BF-1001", "user", "Main is hafte de dunga")
    log_turn("BF-1001", "assistant", "Theek hai, main note kar leta hoon.")

    recap = recent_context_recap("BF-1001")

    assert recap is not None
    assert "Main is hafte de dunga" in recap
    assert "Theek hai, main note kar leta hoon." in recap
    assert recap.index("Main is hafte de dunga") < recap.index("Theek hai")  # oldest first


def test_recap_includes_recent_tool_calls(reseed_accounts):
    from businessflow.accounts import store

    store.log_event("BF-1001", "tool_called", {
        "tool": "log_promise_to_pay",
        "arguments": {"account_id": "BF-1001", "promised_date": "2026-08-25", "promised_amount": 12500},
        "result": {"logged": True},
    })

    recap = recent_context_recap("BF-1001")

    assert recap is not None
    assert "log_promise_to_pay" in recap


def test_recap_is_scoped_to_one_account(reseed_accounts):
    log_turn("BF-1001", "user", "message for account 1001 only")

    assert recent_context_recap("BF-1002") is None
