"""Unit tests for build_system_prompt -- pure string logic, no infra
needed. The one thing worth pinning down here is the Hindi-script
constraint: MMS-TTS crashes on Romanized Hindi (see audio/tts.py), so the
prompt telling the model to reply in Hindi MUST also tell it to use
Devanagari, every time, not just when someone happens to remember.
"""

import time

import pytest

from businessflow.agent import client as client_module
from businessflow.agent.client import build_system_prompt


def test_prompt_requires_checking_dispute_block_before_asking_restructuring_specifics():
    prompt = build_system_prompt(language="en")
    assert "open dispute" in prompt
    assert "restructuring specifics" in prompt


def test_prompt_requires_grounding_policy_claims_in_check_policy():
    prompt = build_system_prompt(language="en")
    assert "check_policy" in prompt
    assert "from memory" in prompt


def test_hindi_prompt_requires_devanagari_script():
    prompt = build_system_prompt(language="hi")

    assert "Devanagari" in prompt
    assert "आपके" in prompt  # the Devanagari example itself must render, not mojibake


def test_english_prompt_does_not_mention_devanagari():
    prompt = build_system_prompt(language="en")

    assert "Devanagari" not in prompt


def test_prompt_with_account_id_names_the_account():
    prompt = build_system_prompt(language="en", account_id="BF-1001")

    assert "BF-1001" in prompt
    assert "You do not yet have access to any real account data" not in prompt


def test_prompt_without_account_id_says_no_account_access():
    prompt = build_system_prompt(language="en", account_id=None)

    assert "You do not yet have access to any real account data" in prompt


@pytest.fixture
def reset_fallback_key_switch():
    """_using_fallback_key/_fallback_switched_at are process-wide, global
    state -- reset both after the test regardless of outcome, so nothing
    leaks into any other test that calls client()."""
    original_flag = client_module._using_fallback_key
    original_switched_at = client_module._fallback_switched_at
    yield
    client_module._using_fallback_key = original_flag
    client_module._fallback_switched_at = original_switched_at


def test_switch_to_fallback_key_changes_which_key_client_returns(reset_fallback_key_switch, monkeypatch):
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    client_module._using_fallback_key = False
    primary = client_module.client()

    switched = client_module.switch_to_fallback_key()

    assert switched is True
    assert client_module.client().api_key != primary.api_key
    assert client_module.client().api_key == "gsk_fake_fallback_key_for_this_test"


def test_switch_to_fallback_key_returns_false_when_none_configured(reset_fallback_key_switch, monkeypatch):
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._using_fallback_key = False

    assert client_module.switch_to_fallback_key() is False


def test_fallback_key_stays_active_before_cooldown_elapses(reset_fallback_key_switch, monkeypatch):
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    client_module.switch_to_fallback_key()
    # Barely any time has passed -- still well inside the cooldown window.
    client_module._fallback_switched_at = time.time() - 60

    assert client_module.client().api_key == "gsk_fake_fallback_key_for_this_test"


def test_fallback_key_reverts_to_primary_after_cooldown_elapses(reset_fallback_key_switch, monkeypatch):
    # Real bug this fixes: the switch used to be permanent for the
    # process's life -- a long-running server would stay on the fallback
    # forever even after the primary's daily quota reset the next day.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_primary_key_for_this_test")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    client_module.switch_to_fallback_key()
    client_module._fallback_switched_at = time.time() - client_module._FALLBACK_COOLDOWN_SECONDS - 1

    assert client_module.client().api_key == "gsk_fake_primary_key_for_this_test"
    assert client_module._using_fallback_key is False
