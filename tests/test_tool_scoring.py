"""Unit tests for eval/tool_scoring.py -- pure logic, no LLM, no DB, so
these run fast and exhaustively rather than relying on a live benchmark
run to exercise every branch. Written because this module had ZERO
test coverage despite being the thing both tool-calling benchmarks'
entire pass/fail and precision/recall numbers rest on.
"""

import json

import pytest

from eval.tool_scoring import (
    ToolExpectation,
    aggregate_results,
    args_match,
    print_regression_delta,
    record_run_history,
    score_turn,
)


class TestArgsMatch:
    def test_exact_match(self):
        assert args_match({"account_id": "BF-1001"}, {"account_id": "BF-1001"}) is True

    def test_extra_actual_args_are_ignored_subset_match(self):
        assert args_match({"account_id": "BF-1001", "reason": "whatever"}, {"account_id": "BF-1001"}) is True

    def test_missing_required_key_fails(self):
        assert args_match({}, {"account_id": "BF-1001"}) is False

    def test_mismatched_value_fails(self):
        assert args_match({"account_id": "BF-1002"}, {"account_id": "BF-1001"}) is False

    def test_numeric_values_use_float_tolerance(self):
        assert args_match({"proposed_amount": 20000.0000001}, {"proposed_amount": 20000}) is True

    def test_numeric_values_beyond_tolerance_fail(self):
        assert args_match({"proposed_amount": 20001}, {"proposed_amount": 20000}) is False


class TestScoreTurn:
    def test_all_required_satisfied_passes(self):
        result = score_turn(
            actual_calls=[("get_payment_status", {"account_id": "BF-1001"})],
            required=[ToolExpectation("get_payment_status", {"account_id": "BF-1001"})],
            forbidden_tools=set(),
        )
        assert result["passed"] is True
        assert result["satisfied_required_tools"] == ["get_payment_status"]
        assert result["missed_required_tools"] == []

    def test_missing_required_call_fails(self):
        result = score_turn(actual_calls=[], required=[ToolExpectation("escalate_to_human")], forbidden_tools=set())

        assert result["passed"] is False
        assert result["missed_required_tools"] == ["escalate_to_human"]

    def test_forbidden_tool_called_fails(self):
        result = score_turn(
            actual_calls=[("log_promise_to_pay", {"account_id": "BF-1001"})],
            required=[],
            forbidden_tools={"log_promise_to_pay"},
        )

        assert result["passed"] is False
        assert result["forbidden_violations"] == ["log_promise_to_pay"]

    def test_star_forbidden_expands_to_every_tool(self):
        result = score_turn(
            actual_calls=[("check_policy", {"query": "grace period"})], required=[], forbidden_tools={"*"},
        )
        assert result["forbidden_violations"] == ["check_policy"]

    def test_a_tool_that_is_both_required_and_forbidden_is_exempted(self):
        # e.g. forbidden_tools={"*"} on a turn that also requires escalate_to_human --
        # the required call shouldn't count as a forbidden violation against itself.
        result = score_turn(
            actual_calls=[("escalate_to_human", {"account_id": "BF-1001"})],
            required=[ToolExpectation("escalate_to_human", {"account_id": "BF-1001"})],
            forbidden_tools={"*"},
        )
        assert result["passed"] is True
        assert result["forbidden_violations"] == []

    def test_neutral_call_is_neither_credited_nor_penalized(self):
        result = score_turn(
            actual_calls=[
                ("get_payment_status", {"account_id": "BF-1001"}),
                ("escalate_to_human", {"account_id": "BF-1001"}),
            ],
            required=[ToolExpectation("escalate_to_human", {"account_id": "BF-1001"})],
            forbidden_tools=set(),
        )
        assert result["passed"] is True
        assert result["neutral_calls"] == ["get_payment_status"]

    def test_required_any_satisfied_by_either_alternative(self):
        result = score_turn(
            actual_calls=[("escalate_to_human", {"account_id": "BF-1003"})],
            required=[],
            forbidden_tools=set(),
            required_any=[
                ToolExpectation("calculate_hypothetical", {"account_id": "BF-1003"}),
                ToolExpectation("escalate_to_human", {"account_id": "BF-1003"}),
            ],
        )
        assert result["passed"] is True
        assert "escalate_to_human" in result["satisfied_required"][0]
        # required_any hits deliberately don't land in the per-tool-only field
        assert result["satisfied_required_tools"] == []

    def test_required_any_missed_when_neither_alternative_happens(self):
        result = score_turn(
            actual_calls=[],
            required=[],
            forbidden_tools=set(),
            required_any=[ToolExpectation("calculate_hypothetical"), ToolExpectation("escalate_to_human")],
        )
        assert result["passed"] is False
        assert len(result["missed_required"]) == 1


class TestAggregateResults:
    def test_precision_and_recall_over_multiple_turns(self):
        turns = [
            score_turn([("get_payment_status", {})], [ToolExpectation("get_payment_status")], set()),
            score_turn([], [ToolExpectation("escalate_to_human")], set()),  # missed
            score_turn([("flag_dispute", {})], [], {"flag_dispute"}),  # forbidden violation
        ]
        agg = aggregate_results(turns)

        assert agg["total_required_calls"] == 2
        assert agg["total_satisfied_calls"] == 1
        assert agg["total_forbidden_violations"] == 1
        assert agg["recall"] == 0.5
        assert agg["precision"] == pytest.approx(1 / 2)

    def test_per_tool_breakdown_only_includes_exercised_tools(self):
        turns = [
            score_turn([("check_policy", {})], [ToolExpectation("check_policy")], set()),
            score_turn([("check_policy", {})], [ToolExpectation("check_policy")], set()),
            score_turn([], [ToolExpectation("check_policy")], set()),
        ]
        agg = aggregate_results(turns)

        assert set(agg["per_tool"]) == {"check_policy"}
        assert agg["per_tool"]["check_policy"] == {
            "required_count": 3, "satisfied": 2, "missed": 1, "recall": round(2 / 3, 4), "forbidden_violations": 0,
        }

    def test_required_any_does_not_pollute_per_tool_stats(self):
        # A required_any miss must not count as a "missed" for either
        # candidate tool -- that would double-penalize tools that were
        # never really "the" expected one for this turn.
        turns = [score_turn(
            [], [], set(),
            required_any=[ToolExpectation("calculate_hypothetical"), ToolExpectation("escalate_to_human")],
        )]
        agg = aggregate_results(turns)

        assert agg["per_tool"] == {}
        assert agg["required_any_total"] == 1
        assert agg["required_any_satisfied"] == 0
        assert agg["required_any_missed"] == 1
        # but it DOES still count toward the overall recall denominator
        assert agg["total_required_calls"] == 1
        assert agg["recall"] == 0.0

    def test_neutral_calls_are_counted_but_not_scored(self):
        turns = [score_turn([("get_payment_status", {})], [], set())]
        agg = aggregate_results(turns)

        assert agg["total_neutral_calls"] == 1
        assert agg["recall"] == 1.0  # no required calls at all -> vacuously perfect, not zero


class TestRunHistory:
    def test_first_run_has_no_previous(self, tmp_path):
        previous = record_run_history("some_benchmark", {"precision": 0.9, "recall": 0.8}, tmp_path)
        assert previous is None

    def test_second_run_returns_the_first_runs_summary(self, tmp_path):
        record_run_history("some_benchmark", {"precision": 0.9, "recall": 0.8}, tmp_path)
        previous = record_run_history("some_benchmark", {"precision": 0.95, "recall": 0.85}, tmp_path)

        assert previous == {"precision": 0.9, "recall": 0.8}

    def test_history_file_has_one_valid_json_line_per_run(self, tmp_path):
        record_run_history("some_benchmark", {"precision": 0.9}, tmp_path)
        record_run_history("some_benchmark", {"precision": 0.95}, tmp_path)

        lines = (tmp_path / "some_benchmark_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert "recorded_at" in entry and "summary" in entry

    def test_different_benchmark_names_do_not_share_history(self, tmp_path):
        record_run_history("benchmark_a", {"precision": 0.5}, tmp_path)
        previous = record_run_history("benchmark_b", {"precision": 0.9}, tmp_path)

        assert previous is None  # benchmark_b's own history is still empty


def test_print_regression_delta_flags_a_real_drop(capsys):
    print_regression_delta({"precision": 0.9, "recall": 0.8}, {"precision": 0.9, "recall": 0.5})
    output = capsys.readouterr().out
    assert "REGRESSION" in output
    assert "recall" in output


def test_print_regression_delta_does_not_flag_a_small_improvement(capsys):
    print_regression_delta({"precision": 0.9}, {"precision": 0.92})
    output = capsys.readouterr().out
    assert "REGRESSION" not in output


def test_print_regression_delta_handles_no_previous_run(capsys):
    print_regression_delta(None, {"precision": 0.9})
    output = capsys.readouterr().out
    assert "first" in output
