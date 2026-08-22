"""Unit test for realistic_conversation_benchmark.py's own scoring
override -- pure logic, no LLM or DB. Written after the same lesson
tool_scoring.py's own test suite exists for: a benchmark's scoring
logic is exactly as unscrutinized as anything else if it has zero
coverage, and this file's main() has always been run manually rather
than under pytest.
"""

from eval.realistic_conversation_benchmark import Turn, _apply_guardrail_intervention_override, _GUARDRAIL_SAFE_REPLY
from eval.tool_scoring import ToolExpectation, score_turn


def _missed_partial_payment_scored() -> dict:
    return score_turn(
        actual_calls=[],
        required=[ToolExpectation("propose_partial_payment", {"account_id": "BF-1003", "proposed_amount": 20000})],
        forbidden_tools=set(),
    )


def test_guardrail_intervention_turns_a_miss_into_a_pass_when_the_flag_is_set():
    turn = Turn("...", required=[ToolExpectation("propose_partial_payment")], accept_guardrail_intervention=True)
    scored = _missed_partial_payment_scored()
    assert scored["passed"] is False  # sanity: genuinely missed before the override

    overridden = _apply_guardrail_intervention_override(scored, turn, _GUARDRAIL_SAFE_REPLY)

    assert overridden["passed"] is True
    assert overridden["missed_required"] == []
    assert overridden["guardrail_intervened"] is True


def test_override_does_nothing_when_the_flag_is_not_set():
    turn = Turn("...", required=[ToolExpectation("propose_partial_payment")], accept_guardrail_intervention=False)
    scored = _missed_partial_payment_scored()

    overridden = _apply_guardrail_intervention_override(scored, turn, _GUARDRAIL_SAFE_REPLY)

    assert overridden == scored  # unchanged -- flag wasn't set, even though the reply matches


def test_override_does_nothing_when_the_reply_is_not_the_safe_deflection():
    # The flag is set, but the reply is a normal, non-guardrail-triggered
    # response -- a genuine miss here must stay a genuine miss.
    turn = Turn("...", required=[ToolExpectation("propose_partial_payment")], accept_guardrail_intervention=True)
    scored = _missed_partial_payment_scored()

    overridden = _apply_guardrail_intervention_override(scored, turn, "Sure, let me check that for you.")

    assert overridden["passed"] is False
    assert overridden["missed_required"] == ["propose_partial_payment"]


def test_escalate_to_human_satisfies_required_any_without_needing_the_guardrail_override():
    # Found live: the model sometimes escalates directly (with the real
    # amount captured in the reason) instead of calling
    # propose_partial_payment -- an equally verified, real action per
    # guardrail/unverified_restructuring.py's own _VERIFYING_TOOLS, so
    # required_any (not the guardrail-intervention fallback) is what
    # should credit this.
    turn = Turn(
        "...",
        required_any=[ToolExpectation("propose_partial_payment"), ToolExpectation("escalate_to_human")],
        accept_guardrail_intervention=True,
    )
    scored = score_turn(
        actual_calls=[("escalate_to_human", {"account_id": "BF-1003", "reason": "..."})],
        required=[], forbidden_tools=set(),
        required_any=[ToolExpectation("propose_partial_payment"), ToolExpectation("escalate_to_human")],
    )
    assert scored["passed"] is True  # already true before the override even runs

    overridden = _apply_guardrail_intervention_override(scored, turn, "a normal, non-guardrail reply")

    assert overridden == scored  # untouched -- the override never needed to fire


def test_override_stays_a_pass_when_the_turn_actually_passed_on_its_own():
    turn = Turn("...", required=[ToolExpectation("propose_partial_payment")], accept_guardrail_intervention=True)
    scored = score_turn(
        actual_calls=[("propose_partial_payment", {"account_id": "BF-1003", "proposed_amount": 20000})],
        required=[ToolExpectation("propose_partial_payment", {"account_id": "BF-1003", "proposed_amount": 20000})],
        forbidden_tools=set(),
    )
    assert scored["passed"] is True

    overridden = _apply_guardrail_intervention_override(scored, turn, _GUARDRAIL_SAFE_REPLY)

    assert overridden["passed"] is True
    assert overridden["missed_required"] == []
    assert overridden["satisfied_required"] == scored["satisfied_required"]
