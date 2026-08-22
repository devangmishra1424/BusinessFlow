"""Shared scoring primitives for tool-calling evals -- used by both
tool_calling_benchmark.py (single-turn scenarios) and
realistic_conversation_benchmark.py (messy, multi-turn scenarios), so the
definition of "did the right tool call happen, with the right arguments"
is identical across both rather than drifting apart.

aggregate_results()/record_run_history() below used to be near-identical
logic pasted into each benchmark's own main() -- moved here once, plus
two things neither version had: a per-tool breakdown (so "check_policy
recall dropped" is answerable, not just "recall dropped somewhere") and
a persisted run history (so a regression between runs is something this
framework can actually show, not just something a human might notice by
eye on two separately-printed JSON blobs).
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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
    satisfied_tools, missed_tools = [], []  # required only, NOT required_any -- see aggregate_results() docstring
    for exp in required:
        hit = any(name == exp.tool_name and args_match(args, exp.required_args) for name, args in actual_calls)
        (satisfied if hit else missed).append(exp.tool_name)
        (satisfied_tools if hit else missed_tools).append(exp.tool_name)

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
        "satisfied_required_tools": satisfied_tools,
        "missed_required_tools": missed_tools,
        "forbidden_violations": forbidden_violations,
        "neutral_calls": neutral_calls,
        "passed": not missed and not forbidden_violations,
    }


def aggregate_results(turn_results: list[dict]) -> dict:
    """Run-level metrics over a flat list of score_turn() outputs (every
    turn of every scenario in one benchmark run): overall precision/
    recall (as before), PLUS a per-tool breakdown and neutral-call
    visibility that neither benchmark script's own ad hoc version had.

    Per-tool recall is scoped to plain `required` expectations only, not
    required_any -- crediting a required_any miss to every one of its
    candidate tools would double-penalize tools that were never really
    "the" expected one for that turn. required_any is tracked as its own
    separate satisfaction rate instead, honestly rather than folded in."""
    total_satisfied = sum(len(t["satisfied_required_tools"]) for t in turn_results)
    total_missed = sum(len(t["missed_required_tools"]) for t in turn_results)
    total_required = total_satisfied + total_missed
    total_forbidden_violations = sum(len(t["forbidden_violations"]) for t in turn_results)
    total_neutral_calls = sum(len(t["neutral_calls"]) for t in turn_results)

    def _is_required_any_entry(s) -> bool:
        return isinstance(s, str) and s.startswith("one of ")

    required_any_turns = [t for t in turn_results if any(_is_required_any_entry(s) for s in t["satisfied_required"] + t["missed_required"])]
    required_any_satisfied = sum(1 for t in required_any_turns if any(_is_required_any_entry(s) for s in t["satisfied_required"]))

    # required_any's misses count toward the overall recall/precision
    # denominators too (a "one of [...]" miss is a real miss), just not
    # toward any single tool's per-tool stats.
    required_any_missed = len(required_any_turns) - required_any_satisfied
    total_required += len(required_any_turns)
    total_satisfied += required_any_satisfied

    recall = total_satisfied / total_required if total_required else 1.0
    denom = total_satisfied + total_forbidden_violations
    precision = total_satisfied / denom if denom else 1.0

    per_tool = {}
    for tool_name in sorted(ALL_TOOL_NAMES):
        tool_satisfied = sum(t["satisfied_required_tools"].count(tool_name) for t in turn_results)
        tool_missed = sum(t["missed_required_tools"].count(tool_name) for t in turn_results)
        tool_forbidden_hits = sum(t["forbidden_violations"].count(tool_name) for t in turn_results)
        tool_required_total = tool_satisfied + tool_missed
        if tool_required_total == 0 and tool_forbidden_hits == 0:
            continue  # never actually exercised as required or forbidden in this run
        per_tool[tool_name] = {
            "required_count": tool_required_total,
            "satisfied": tool_satisfied,
            "missed": tool_missed,
            "recall": round(tool_satisfied / tool_required_total, 4) if tool_required_total else None,
            "forbidden_violations": tool_forbidden_hits,
        }

    return {
        "turn_count": len(turn_results),
        "total_required_calls": total_required,
        "total_satisfied_calls": total_satisfied,
        "total_forbidden_violations": total_forbidden_violations,
        "total_neutral_calls": total_neutral_calls,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "required_any_total": len(required_any_turns),
        "required_any_satisfied": required_any_satisfied,
        "required_any_missed": required_any_missed,
        "per_tool": per_tool,
    }


def record_run_history(name: str, summary: dict, results_dir: Path) -> dict | None:
    """Appends this run's summary to results/<name>_history.jsonl, stamped
    with the real wall-clock time, and returns the immediately PRIOR
    run's summary (or None if this is the first) -- silent metric drift
    between runs is exactly what a benchmark with no history can't catch,
    since each run previously just overwrote the last one's JSON file."""
    history_path = results_dir / f"{name}_history.jsonl"
    previous = None
    if history_path.exists():
        lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous = json.loads(lines[-1])["summary"]

    results_dir.mkdir(exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"recorded_at": datetime.now(timezone.utc).isoformat(), "summary": summary}, ensure_ascii=False) + "\n")
    return previous


def print_regression_delta(previous: dict | None, current: dict, metrics: tuple[str, ...] = ("precision", "recall")) -> None:
    if previous is None:
        print("(no previous run recorded -- this is the first)")
        return
    for m in metrics:
        if m not in previous or m not in current:
            continue
        delta = current[m] - previous[m]
        flag = "  <-- REGRESSION" if delta < -0.01 else ""
        print(f"  {m}: {previous[m]} -> {current[m]} ({delta:+.4f}){flag}")
