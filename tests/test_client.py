"""Unit tests for build_system_prompt -- pure string logic, no infra
needed. The one thing worth pinning down here is the Hindi-script
constraint: MMS-TTS crashes on Romanized Hindi (see audio/tts.py), so the
prompt telling the model to reply in Hindi MUST also tell it to use
Devanagari, every time, not just when someone happens to remember.
"""

import os
import time

import pytest

from businessflow.agent import client as client_module
from businessflow.agent.client import build_system_prompt


def test_prompt_requires_checking_dispute_block_before_asking_restructuring_specifics():
    prompt = build_system_prompt(language="en")
    assert "open dispute" in prompt
    assert "restructuring specifics" in prompt


def test_prompt_requires_checking_a_concrete_proposed_amount_rather_than_stating_the_block_from_memory():
    # Found live via eval/tool_calling_benchmark.py and eval/
    # realistic_conversation_benchmark.py: given a concrete proposed
    # amount on a dispute-blocked account, the model sometimes stated
    # the block directly (sometimes citing an ungrounded specific
    # number the Guardrail then had to block) instead of calling
    # propose_partial_payment to get the real, grounded reason/minimum.
    prompt = build_system_prompt(language="en")
    assert "never a substitute for checking a real number" in prompt


def test_prompt_requires_grounding_policy_claims_in_check_policy():
    prompt = build_system_prompt(language="en")
    assert "check_policy" in prompt
    assert "from memory" in prompt


def test_prompt_requires_grounding_account_facts_in_get_payment_status():
    # Found live via the Telegram channel: a compound question ("how much
    # is my loan, how many months left, what's my emi") got answered with
    # a wrong months-remaining figure and no tool call at all -- caught
    # neither by check_grounding (months isn't a rupee amount it checks)
    # nor by anything in the prompt before this fragment existed.
    prompt = build_system_prompt(language="en")
    assert "get_payment_status" in prompt
    assert "one get_payment_status call" in prompt  # the compound-question instruction specifically


def test_prompt_tells_the_model_to_check_interest_rate_pct_before_admitting_its_untracked():
    # interest_rate_pct is a real get_payment_status field now (populated
    # only once a loan agreement is uploaded and parsed) -- the prompt
    # must tell the model to check it first, and only fall back to "not
    # on file" when it's actually null for this account, not assume it's
    # always untracked the way it used to.
    prompt = build_system_prompt(language="en")
    assert "interest rate" in prompt
    assert "interest_rate_pct" in prompt
    assert "isn't on file" in prompt


def test_prompt_forbids_claiming_to_have_sent_or_emailed_anything():
    # Found live via the Telegram channel, the most serious issue found
    # there: asked to email the loan agreement, the model claimed to have
    # "arranged to send" it, then later restated it as settled fact
    # ("I've already sent a copy") -- there is no email/document-delivery
    # tool anywhere in this system. check_grounding only checks URLs and
    # rupee amounts, so a claimed ACTION like this needs an explicit
    # prompt rule, the same reason _NO_FABRICATED_LINKS exists.
    prompt = build_system_prompt(language="en")
    assert "sent, emailed, mailed, forwarded, or delivered" in prompt
    assert "no email, SMS, or document-delivery capability" in prompt


def test_prompt_grounds_nach_failure_answers_and_forbids_claiming_to_fix_a_mandate():
    # accounts.nach_mandate_active is a real seeded column and
    # faq_general.md frames this whole agent as existing because a
    # borrower's auto-debit failed -- but nothing told the model to check
    # that field (rather than guess a specific reason) or that it cannot
    # itself re-register a mandate.
    prompt = build_system_prompt(language="en")
    assert "nach_mandate_active" in prompt
    assert "nach_mandate_troubleshooting" in prompt
    assert "Never claim you can re-register, reactivate, or otherwise fix a NACH mandate yourself" in prompt


def test_prompt_declines_legal_advice_and_offers_escalation():
    prompt = build_system_prompt(language="en")
    assert "legal advice" in prompt
    assert "escalate" in prompt


def test_prompt_escalates_fraud_or_identity_claims_as_more_urgent_than_a_routine_dispute():
    # "this loan isn't mine" / "someone took this loan out in my name" is a
    # categorically different claim than an ordinary billing dispute -- both
    # currently could go through flag_dispute/escalate_to_human with no
    # distinction, so the reason string handed to escalate_to_human must
    # make the urgency unmistakable to whoever picks it up.
    prompt = build_system_prompt(language="en")
    assert "SUSPECTED FRAUD/IDENTITY CLAIM" in prompt
    assert "escalate_to_human" in prompt
    assert "ask them for 'proof'" in prompt


def test_prompt_refuses_to_fabricate_a_due_date_change_and_offers_escalation_instead():
    # This agent has no mechanism to actually move a due date -- it's tied
    # to the NACH mandate with the borrower's bank, not something this
    # system can self-serve. Same class of problem _NO_FABRICATED_ACTIONS
    # guards against, named explicitly here since it's common enough to ask.
    prompt = build_system_prompt(language="en")
    assert "can't do that directly" in prompt
    assert "NACH mandate" in prompt
    assert "Never agree to 'note', 'arrange', or 'put in' a due-date change" in prompt


def test_prompt_gives_only_general_credit_score_information_never_this_borrowers_specific_score():
    # The system has no access to any borrower's actual credit report/CIBIL
    # score. A general, factual answer is fine; a specific one about THIS
    # borrower's score or personalized credit advice is not, since there is
    # nothing to ground either in.
    prompt = build_system_prompt(language="en")
    assert "credit score" in prompt
    assert "never state anything specific about THIS borrower's actual score or credit report" in prompt
    assert "personalized financial advice" in prompt


def test_prompt_holds_the_settlement_discount_firm_against_negotiation_pressure():
    # SETTLEMENT_DISCOUNT_PCT is a fixed 5% policy constant -- an LLM under
    # social pressure to be "helpful" can drift toward implying flexibility
    # that doesn't exist. The prompt must require a real
    # calculate_hypothetical call and state the figure is fixed, not
    # negotiable.
    prompt = build_system_prompt(language="en")
    assert "currently 5%" in prompt
    assert "calculate_hypothetical" in prompt
    assert "not negotiable through this channel" in prompt


def test_prompt_treats_an_explicit_complaint_as_a_grievance_grounded_in_check_policy():
    # escalation_policy.md only ever covered handing an account to a human
    # HERE -- nothing told the model what to do when the borrower is
    # dissatisfied with HOW their case was handled, which is a distinct
    # thing from a routine dispute or an ordinary escalate_to_human call.
    prompt = build_system_prompt(language="en")
    assert "file a complaint" in prompt
    assert "grievance_redressal.md" in prompt
    assert "grievance about handling, not a routine account matter" in prompt


def test_prompt_never_states_ombudsman_contact_details_from_memory():
    # No Ombudsman phone number, address, or URL exists anywhere in this
    # system to ground one in -- stating one from memory would be exactly
    # the class of fabrication _NO_FABRICATED_LINKS/_NO_FABRICATED_ACTIONS
    # already guard against elsewhere in this prompt.
    prompt = build_system_prompt(language="en")
    assert "RBI Banking/NBFC Ombudsman" in prompt
    assert "never state a specific phone number, address, or URL for it from memory" in prompt


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
def reset_fallback_key_switch(monkeypatch):
    """_current_key_index/_switched_at are process-wide, global state --
    reset both after the test regardless of outcome, so nothing leaks
    into any other test that calls client(). Also clears every real
    ALTERNATE_GROQ_KEY{N} this process's actual .env may have loaded --
    without this, a test asserting "no more fallbacks configured" would
    silently see whatever real fallback keys happen to be in .env right
    now, not the clean slate it's written to expect.

    Ensures GROQ_API_KEY itself is set to a fake value if it isn't
    already -- found via CI (which has no real .env at all): a test
    reading client().api_key at index 0 raised RuntimeError there
    instead of asserting anything, since these tests only ever check
    which key was SELECTED, never make a real Groq call, so a fake
    primary key is exactly as good as a real one here."""
    for n in range(2, client_module._MAX_FALLBACK_KEY_SUFFIX + 1):
        monkeypatch.delenv(f"ALTERNATE_GROQ_KEY{n}", raising=False)
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    if not os.environ.get("GROQ_API_KEY"):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_primary_key_for_this_test")

    original_index = client_module._current_key_index
    original_switched_at = client_module._switched_at
    yield
    client_module._current_key_index = original_index
    client_module._switched_at = original_switched_at


def test_switch_to_fallback_key_changes_which_key_client_returns(reset_fallback_key_switch, monkeypatch):
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    client_module._current_key_index = 0
    primary = client_module.client()

    switched = client_module.switch_to_fallback_key()

    assert switched is True
    assert client_module.client().api_key != primary.api_key
    assert client_module.client().api_key == "gsk_fake_fallback_key_for_this_test"


def test_switch_to_fallback_key_returns_false_when_none_configured(reset_fallback_key_switch, monkeypatch):
    monkeypatch.delenv("ALTERNATE_GROQ_KEY", raising=False)
    client_module._current_key_index = 0

    assert client_module.switch_to_fallback_key() is False


def test_switch_to_fallback_key_advances_through_multiple_configured_fallbacks(reset_fallback_key_switch, monkeypatch):
    # Real scenario this fixes: with only ONE fallback slot, a second
    # rate limit in a row had nowhere left to go. With several
    # ALTERNATE_GROQ_KEY{N} configured, it should keep advancing.
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_1")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY2", "gsk_fake_fallback_2")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY3", "gsk_fake_fallback_3")
    client_module._current_key_index = 0

    assert client_module.switch_to_fallback_key() is True
    assert client_module.client().api_key == "gsk_fake_fallback_1"

    assert client_module.switch_to_fallback_key() is True
    assert client_module.client().api_key == "gsk_fake_fallback_2"

    assert client_module.switch_to_fallback_key() is True
    assert client_module.client().api_key == "gsk_fake_fallback_3"

    # All three configured fallbacks now tried -- nowhere left to go.
    assert client_module.switch_to_fallback_key() is False


def test_fallback_env_var_names_does_not_stop_at_the_first_gap(reset_fallback_key_switch, monkeypatch):
    # ALTERNATE_GROQ_KEY2 missing, but 3 and 4 ARE set -- a naive
    # "stop at the first unset one" scan would miss 3 and 4 entirely.
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_1")
    monkeypatch.delenv("ALTERNATE_GROQ_KEY2", raising=False)
    monkeypatch.setenv("ALTERNATE_GROQ_KEY3", "gsk_fake_fallback_3")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY4", "gsk_fake_fallback_4")

    names = client_module._fallback_env_var_names()

    assert names == ["ALTERNATE_GROQ_KEY", "ALTERNATE_GROQ_KEY3", "ALTERNATE_GROQ_KEY4"]


def test_fallback_key_stays_active_before_cooldown_elapses(reset_fallback_key_switch, monkeypatch):
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    # Explicit reset, not an assumption -- found via the full suite (not
    # this file in isolation): a real Groq call elsewhere that hits a
    # real rate limit advances this same global _current_key_index, and
    # nothing outside this file's own fixture resets it back down. This
    # test's premise (switch_to_fallback_key() advances index 0 -> 1)
    # only holds if it actually starts at 0.
    client_module._current_key_index = 0
    client_module.switch_to_fallback_key()
    # Barely any time has passed -- still well inside the cooldown window.
    client_module._switched_at = time.time() - 60

    assert client_module.client().api_key == "gsk_fake_fallback_key_for_this_test"


def test_fallback_key_reverts_to_primary_after_cooldown_elapses(reset_fallback_key_switch, monkeypatch):
    # Real bug this fixes: the switch used to be permanent for the
    # process's life -- a long-running server would stay on the fallback
    # forever even after the primary's daily quota reset the next day.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_primary_key_for_this_test")
    monkeypatch.setenv("ALTERNATE_GROQ_KEY", "gsk_fake_fallback_key_for_this_test")
    client_module._current_key_index = 0  # same explicit-reset reasoning as the test above
    client_module.switch_to_fallback_key()
    client_module._switched_at = time.time() - client_module._FALLBACK_COOLDOWN_SECONDS - 1

    assert client_module.client().api_key == "gsk_fake_primary_key_for_this_test"
    assert client_module._current_key_index == 0
