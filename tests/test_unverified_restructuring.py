"""Unit tests for guardrail/unverified_restructuring.py -- pure logic,
no LLM or DB. The real bug this exists to catch: a concrete
restructuring/partial-payment proposal stated without a verifying tool
call, found live in a long conversation where two prompt-only fixes
for the same pattern didn't hold up (see agent/client.py's
_CHECK_DISPUTE_BLOCK_FIRST and its own comments).
"""

import pytest

from businessflow.guardrail.unverified_restructuring import check_unverified_restructuring_claim


def test_concrete_proposal_with_no_verifying_tool_call_is_flagged():
    result = check_unverified_restructuring_claim(
        "can i at least pay like 20000 now instead of the full amount", tools_called_this_turn=set(),
    )

    assert result
    assert result.proposed_amount == 20000.0


def test_concrete_proposal_with_propose_partial_payment_called_is_not_flagged():
    result = check_unverified_restructuring_claim(
        "can i at least pay like 20000 now instead of the full amount",
        tools_called_this_turn={"propose_partial_payment"},
    )

    assert not result


def test_concrete_proposal_with_calculate_hypothetical_called_is_not_flagged():
    result = check_unverified_restructuring_claim(
        "can you lower it to 8000 instead", tools_called_this_turn={"calculate_hypothetical"},
    )

    assert not result


def test_concrete_proposal_with_propose_restructuring_called_is_not_flagged():
    # propose_restructuring calls calculate_hypothetical internally, but
    # the LLM tool-call trace this checks only ever sees the outer name.
    result = check_unverified_restructuring_claim(
        "can you extend it by 3 months instead", tools_called_this_turn={"propose_restructuring"},
    )

    assert not result


def test_concrete_proposal_with_escalate_to_human_called_is_not_flagged():
    # Escalating IS a legitimate verified outcome here -- a human will
    # actually check the real number, just not automatically.
    result = check_unverified_restructuring_claim(
        "i can only do 15000 instead of the full emi", tools_called_this_turn={"escalate_to_human"},
    )

    assert not result


def test_vague_ask_with_no_concrete_number_is_not_flagged():
    # Nothing concrete to verify yet -- per _CHECK_DISPUTE_BLOCK_FIRST,
    # stating a known block directly is legitimate here.
    result = check_unverified_restructuring_claim(
        "can you lower my monthly payment or stretch it out longer, im struggling here",
        tools_called_this_turn=set(),
    )

    assert not result


def test_a_number_with_no_restructuring_intent_language_is_not_flagged():
    # "20000" appears, but nothing about partial/lower/instead-of --
    # e.g. just answering an unrelated question with a number.
    result = check_unverified_restructuring_claim(
        "my business made 20000 in revenue last month", tools_called_this_turn=set(),
    )

    assert not result


def test_an_unrelated_tool_call_this_turn_does_not_satisfy_the_check():
    # get_payment_status happened, but that's not one of the tools that
    # actually verifies a restructuring/partial-payment proposal.
    result = check_unverified_restructuring_claim(
        "can i just pay 15000 instead of the full thing", tools_called_this_turn={"get_payment_status"},
    )

    assert result
    assert result.proposed_amount == 15000.0


@pytest.mark.parametrize("phrase", [
    "can i pay 12000 instead of the full amount",
    "i can only pay 12000 this month",
    "at least let me pay 12000 for now",
    "lower it to 12000 please",
    "can we settle this for 12000",
])
def test_various_real_phrasings_of_a_concrete_reduced_payment_ask_are_detected(phrase):
    result = check_unverified_restructuring_claim(phrase, tools_called_this_turn=set())
    assert result
