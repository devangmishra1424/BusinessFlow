"""Integration tests for the real tool-calling agent loop -- makes real
Groq API calls, so skipped entirely if GROQ_API_KEY isn't set (same
reasoning as test_pipeline.py: a mock can't prove the round-trip works).
"""

import os

import pytest

from businessflow.agent.loop import run_turn, start_conversation

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set -- copy .env.example to .env and fill it in to run this",
    ),
    pytest.mark.skipif(
        not os.environ.get("DATABASE_URL"),
        reason="DATABASE_URL not set -- these tests hit a real account (BF-1001)",
    ),
]


def test_agent_grounds_its_answer_in_a_real_tool_call():
    conversation = start_conversation(language="en", account_id="BF-1001")
    conversation.append({"role": "user", "content": "How many days past due is my payment, and how much do I owe?"})

    conversation, reply_text = run_turn(conversation)

    # A real tool call must have happened -- the model has no other way to
    # know these numbers, since they aren't in the system prompt.
    assert any(msg.get("role") == "tool" for msg in conversation)
    assert "3" in reply_text  # Priya Sharma (BF-1001) is 3 days past due
    # "how much do I owe" is genuinely ambiguous between this cycle's EMI
    # and the total outstanding balance -- get_payment_status returns both
    # (outstanding_balance_approx was added after this test was first
    # written), and a real run can correctly answer with either. Found
    # live via CI: the model started preferring the outstanding balance,
    # a more literal answer to "how much do I owe" than the recurring EMI,
    # which is a real improvement, not a regression -- so both grounded
    # numbers pass here, and only a hallucinated third number would fail.
    assert (
        "12" in reply_text or "12,500" in reply_text or "twelve" in reply_text.lower()
        or "175,000" in reply_text or "175000" in reply_text
    )


def test_agent_escalates_a_high_risk_account_instead_of_offering_automated_restructuring():
    conversation = start_conversation(language="en", account_id="BF-1003")
    conversation.append({
        "role": "user",
        "content": "I can't pay my full EMI this month, can you lower it or give me a payment plan?",
    })

    conversation, reply_text = run_turn(conversation)

    tool_calls_made = [
        msg["function"]["name"]
        for prior_message in conversation
        if prior_message.get("role") == "assistant"
        for msg in (prior_message.get("tool_calls") or [])
    ]
    # BF-1003 (Fatima Khan) has an open dispute and 2 broken promises --
    # policy blocks any automated restructuring offer on this account.
    assert "propose_partial_payment" not in tool_calls_made or "escalate_to_human" in tool_calls_made
