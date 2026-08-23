"""One-off A/B test: does reasoning_effort="high" measurably improve
tool-calling reliability on compound_account_status_question_en (the
exact live bug this session found -- a compound multi-part account
question answered with NO get_payment_status call at all, "0 months
left" stated as fact when the real value was 14)?

Runs that one scenario 5x at the current default (reasoning_effort
unset -- whatever Groq/the model default to) and 5x with
reasoning_effort="high", via the real _run_turn_async path (loop.py's
reasoning_effort kwarg, added for this experiment), against the real
Groq API and real Postgres -- same requirements as the other eval
scripts, no mocks.

Mirrors scripts/verify_restructuring_guardrail.py's existing pattern
for running a single scenario directly instead of the full suite.

Run: python -m scripts.ab_reasoning_effort
"""

import contextlib
import io
import json
import sys
import time

from businessflow.agent.loop import extract_new_tool_calls, run_turn, start_conversation
from eval.realistic_conversation_benchmark import SCENARIOS
from eval.tool_scoring import score_turn
from scripts.seed_accounts import main as _reseed_demo_accounts

_SCENARIO = next(s for s in SCENARIOS if s.scenario_id == "compound_account_status_question_en")
_TURN = _SCENARIO.turns[0]  # the scenario has exactly one turn
_REPEATS = 5


def _run_once(reasoning_effort: str | None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()

    conversation = start_conversation(language=_SCENARIO.language, account_id=_SCENARIO.account_id)
    turn_start = len(conversation)
    conversation.append({"role": "user", "content": _TURN.user_message})

    started_at = time.monotonic()
    conversation, reply = run_turn(conversation, reasoning_effort=reasoning_effort)
    elapsed_seconds = time.monotonic() - started_at

    actual_calls = extract_new_tool_calls(conversation, turn_start)
    scored = score_turn(actual_calls, _TURN.required, _TURN.forbidden_tools, _TURN.required_any)
    called_get_payment_status = any(name == "get_payment_status" for name, _ in actual_calls)

    return {
        "reasoning_effort": reasoning_effort or "(unset/default)",
        "passed": scored["passed"],
        "called_get_payment_status": called_get_payment_status,
        "tool_calls": [name for name, _ in actual_calls],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "reply": reply,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = {"default": [], "high": []}

    print(f"=== {_REPEATS}x at current default (reasoning_effort unset) ===")
    for i in range(_REPEATS):
        r = _run_once(None)
        results["default"].append(r)
        print(f"  run {i + 1}: passed={r['passed']} get_payment_status_called={r['called_get_payment_status']} "
              f"elapsed={r['elapsed_seconds']}s tools={r['tool_calls']}")

    print(f"\n=== {_REPEATS}x at reasoning_effort='high' ===")
    for i in range(_REPEATS):
        r = _run_once("high")
        results["high"].append(r)
        print(f"  run {i + 1}: passed={r['passed']} get_payment_status_called={r['called_get_payment_status']} "
              f"elapsed={r['elapsed_seconds']}s tools={r['tool_calls']}")

    for label, rows in results.items():
        pass_rate = sum(1 for r in rows if r["passed"]) / len(rows) if rows else None
        mean_latency = sum(r["elapsed_seconds"] for r in rows) / len(rows) if rows else None
        print(f"\n{label}: pass_rate={pass_rate} mean_elapsed_seconds={round(mean_latency, 3) if mean_latency else None}")

    out_path = "scripts/ab_reasoning_effort_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out_path}")

    return results


if __name__ == "__main__":
    main()
