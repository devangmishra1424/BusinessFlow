"""Unit tests for the memory-aware wrappers in agent/loop.py
(start_conversation_with_recap) that don't require a live Groq call --
only start_conversation_with_recap is tested here, since building the
message list doesn't touch the model. run_turn_with_memory does call the
model (it wraps run_turn) and is covered by the GROQ-gated integration
tests instead.
"""

import os

import pytest

from businessflow.agent.loop import start_conversation, start_conversation_with_recap
from businessflow.memory.conversation_memory import log_turn

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set -- these tests hit real Postgres",
)


def test_start_conversation_with_recap_appends_recap_when_history_exists(reseed_accounts):
    log_turn("BF-1001", "user", "previous message from an earlier session")

    conversation = start_conversation_with_recap(language="en", account_id="BF-1001")

    assert len(conversation) == 2
    assert conversation[0]["role"] == "system"
    assert conversation[1]["role"] == "system"
    assert "previous message from an earlier session" in conversation[1]["content"]


def test_start_conversation_with_recap_matches_plain_version_with_no_history(reseed_accounts):
    plain = start_conversation(language="en", account_id="BF-1001")
    with_recap = start_conversation_with_recap(language="en", account_id="BF-1001")

    assert with_recap == plain  # no prior history -> nothing appended, identical to the plain version
