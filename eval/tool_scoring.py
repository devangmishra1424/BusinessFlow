"""Shared scoring primitives for tool-calling evals -- used by both
tool_calling_benchmark.py (single-turn scenarios) and
realistic_conversation_benchmark.py (messy, multi-turn scenarios), so the
definition of "did the right tool call happen, with the right arguments"
is identical across both rather than drifting apart.
"""

from dataclasses import dataclass, field

from businessflow.agent.loop import extract_new_tool_calls as extract_tool_calls  # noqa: F401 -- re-exported

ALL_TOOL_NAMES = {
    "get_payment_status",
    "log_promise_to_pay",
    "flag_dispute",
    "generate_payment_link",
    "propose_partial_payment",
    "calculate_hypothetical",
    "escalate_to_human",
    "check_policy",
}


@dataclass
class ToolExpectation:
    tool_name: str
    required_args: dict = field(default_factory=dict)  # subset match


def args_match(actual: dict, required: dict) -> bool:
    for key, expected_value in required.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(float(actual_value) - float(expected_value)) > 1e-6:
                return False
        elif actual_value != expected_value:
            return False
    return True


def score_turn(
    actual_calls: list[tuple[str, dict]],
    required: list[ToolExpectation],
    forbidden_tools: set[str],
    required_any: list[ToolExpectation] = (),
) -> dict:
    """Scores one turn's actual tool calls against what was required/forbidden.
    Tools that are neither required nor forbidden (e.g. a reasonable
    context-gathering get_payment_status call) are neutral -- not
    penalized, not credited.

    required_any is for genuinely equivalent alternatives (e.g. "check
    eligibility, OR just escalate directly given a known policy block" --
    both are correct, and requiring one specific tool when the prompt
    itself explicitly permits either is a scoring bug, not an agent one.
    At least one must be satisfied; default empty changes nothing for
    existing callers."""
    forbidden = ALL_TOOL_NAMES if "*" in forbidden_tools else forbidden_tools
    required_tool_names = {exp.tool_name for exp in required} | {exp.tool_name for exp in required_any}

    satisfied, missed = [], []
    for exp in required:
        hit = any(name == exp.tool_name and args_match(args, exp.required_args) for name, args in actual_calls)
        (satisfied if hit else missed).append(exp.tool_name)

    if required_any:
        any_hits = [
            exp.tool_name for exp in required_any
            if any(name == exp.tool_name and args_match(args, exp.required_args) for name, args in actual_calls)
        ]
        if any_hits:
            satisfied.append(f"one of {[e.tool_name for e in required_any]} -> {any_hits[0]}")
        else:
            missed.append(f"one of {[e.tool_name for e in required_any]}")

    forbidden_violations = [name for name, _ in actual_calls if name in forbidden and name not in required_tool_names]
    neutral_calls = [name for name, _ in actual_calls if name not in forbidden and name not in required_tool_names]

    return {
        "actual_tool_calls": [{"tool": n, "args": a} for n, a in actual_calls],
        "satisfied_required": satisfied,
        "missed_required": missed,
        "forbidden_violations": forbidden_violations,
        "neutral_calls": neutral_calls,
        "passed": not missed and not forbidden_violations,
    }
