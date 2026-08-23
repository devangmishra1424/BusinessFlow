"""Tests that the Guardrail is actually wired into the agent loop's
finalize step -- not just that the grounding logic works in isolation
(see test_grounding.py). Real Postgres, no LLM needed: _finalize_reply
is a plain function over an already-built conversation list.
"""

import os

import pytest

from businessflow.accounts import store
from businessflow.agent.loop import _finalize_reply

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_ungrounded_reply_gets_rewritten_in_place(reseed_accounts):
    conversation = [
        {"role": "user", "content": "send me the link"},
        {"role": "tool", "content": '{"account_id": "BF-1001"}'},
        {"role": "assistant", "content": "Here: https://payments.example.com/pay?acc=BF-1001&amount=12500"},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id="BF-1001")

    assert "payments.example.com" not in reply
    assert "connect you" in reply
    assert updated_conversation[-1]["content"] == reply  # transcript reflects what was actually said


def test_ungrounded_reply_creates_a_real_escalation(reseed_accounts):
    conversation = [
        {"role": "user", "content": "send me the link"},
        {"role": "assistant", "content": "Here: https://payments.example.com/pay?acc=BF-1001&amount=12500"},
    ]

    _finalize_reply(conversation, verified_account_id="BF-1001")

    account = store.get_account_or_raise("BF-1001")
    # A real escalation row should now exist for this account (created via
    # store.create_escalation inside _finalize_reply's failure path).
    rows = store.get_connection().execute(
        "select count(*) as n from escalations where account_id = %s", ("BF-1001",)
    ).fetchone()
    assert rows["n"] >= 1
    assert account is not None  # sanity: account itself still intact


def test_grounded_reply_passes_through_unchanged(reseed_accounts):
    conversation = [
        {"role": "user", "content": "whats my balance"},
        {"role": "tool", "content": '{"emi_amount": 12500.0}'},
        {"role": "assistant", "content": "Your balance is ₹12,500."},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id="BF-1001")

    assert reply == "Your balance is ₹12,500."


def test_no_escalation_attempted_without_a_verified_account(reseed_accounts):
    # No account to escalate against -- must not raise trying to create
    # one (create_escalation requires a real account_id via FK).
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Here: https://payments.example.com/fake"},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id=None)

    assert "connect you" in reply


def test_unverified_restructuring_claim_gets_rewritten_in_place(reseed_accounts):
    # Real bug this catches: found live in a long conversation where the
    # model stated a dispute block for a NEW concrete amount without
    # ever calling propose_partial_payment/calculate_hypothetical this
    # turn -- grounding.check_grounding alone has nothing to catch here
    # (the amount, ₹20,000, came straight from the borrower's own words).
    conversation = [
        {"role": "user", "content": "can i at least pay like 20000 now instead of the full amount"},
        {"role": "assistant", "content": "Because of the open dispute, we can't accept a partial payment right now."},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id="BF-1001")

    assert "connect you" in reply
    assert updated_conversation[-1]["content"] == reply


def test_fabricated_action_claim_gets_rewritten_in_place(reseed_accounts):
    # Real bug found live via the Telegram channel: the model claimed to
    # have emailed the loan agreement -- there is no email tool anywhere
    # in this system, so no conversation history could ever make this
    # true. check_grounding alone has nothing to catch here (no URL, no
    # rupee amount in the reply).
    conversation = [
        {"role": "user", "content": "send me the contract on my email"},
        {"role": "assistant", "content": "I've already sent a copy of the full agreement to your email."},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id="BF-1001")

    assert "already sent" not in reply
    assert "connect you" in reply
    assert updated_conversation[-1]["content"] == reply


def test_restructuring_claim_with_a_real_verifying_tool_call_passes_through(reseed_accounts):
    conversation = [
        {"role": "user", "content": "can i at least pay like 20000 now instead of the full amount"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "function": {"name": "propose_partial_payment", "arguments": '{"account_id": "BF-1001", "proposed_amount": 20000}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"eligible": false, "reason": "open dispute"}'},
        {"role": "assistant", "content": "Because of the open dispute, we can't accept a partial payment right now."},
    ]

    updated_conversation, reply = _finalize_reply(conversation, verified_account_id="BF-1001")

    assert reply == "Because of the open dispute, we can't accept a partial payment right now."
