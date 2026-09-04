"""Tests for agent/loop.py's update_conversation_language -- pure logic,
no Groq/Postgres needed (choose_prompt_version's own get_active_variant
falls back to the baseline template when DATABASE_URL isn't set, by
design -- see its own docstring). See update_conversation_language's
docstring for the real bug this closes: a mid-conversation /hindi,
/english, or /language switch updated only the calling channel's own
tracking state, never conversation[0]'s actual system prompt -- so the
model kept replying in whatever language the conversation started in,
directly contradicting the bot's own "Hindi set for this conversation"
confirmation.
"""

from businessflow.agent.loop import start_conversation, update_conversation_language


def test_update_conversation_language_actually_changes_the_system_prompt():
    conversation = start_conversation(language="en", account_id="BF-1001")
    assert "Reply in plain English." in conversation[0]["content"]
    assert "Devanagari" not in conversation[0]["content"]

    update_conversation_language(conversation, "hi", "BF-1001")

    assert "Reply in plain English." not in conversation[0]["content"]
    assert "Devanagari" in conversation[0]["content"]


def test_update_conversation_language_preserves_the_account_context():
    # The switch must not accidentally revert to the no-account-context
    # template -- only the language instruction should change.
    conversation = start_conversation(language="en", account_id="BF-1001")

    update_conversation_language(conversation, "hi", "BF-1001")

    assert "account BF-1001" in conversation[0]["content"]


def test_update_conversation_language_only_touches_the_system_message():
    conversation = start_conversation(language="en", account_id="BF-1001")
    conversation.append({"role": "user", "content": "what's my balance"})
    conversation.append({"role": "assistant", "content": "checking now"})

    update_conversation_language(conversation, "hi", "BF-1001")

    assert conversation[1] == {"role": "user", "content": "what's my balance"}
    assert conversation[2] == {"role": "assistant", "content": "checking now"}


def test_update_conversation_language_is_a_noop_on_an_empty_conversation():
    conversation = []

    update_conversation_language(conversation, "hi", "BF-1001")

    assert conversation == []


def test_update_conversation_language_is_a_noop_when_the_first_message_is_not_system():
    conversation = [{"role": "user", "content": "hello"}]

    update_conversation_language(conversation, "hi", "BF-1001")

    assert conversation == [{"role": "user", "content": "hello"}]


def test_update_conversation_language_works_for_an_anonymous_conversation_too():
    conversation = start_conversation(language="en", account_id=None)

    update_conversation_language(conversation, "hi", None)

    assert "Devanagari" in conversation[0]["content"]
    assert "You do not yet have access to any real account data" in conversation[0]["content"]
