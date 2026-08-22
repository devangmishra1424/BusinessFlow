"""Measures how long a real conversation turn actually takes, end to
end (user message -> final reply), including every tool-call round
trip -- never measured before this. Reuses a representative spread of
scenarios (zero, one, and multi-tool-call cases, across languages) so
the numbers reflect real usage patterns, not one synthetic best case.

Voice (STT/TTS) is out of scope for now, so this measures the text/LLM/
tool-loop path only -- the part that's shared with voice once that's
built, and the part actually controllable by this project's own code
(STT/TTS latency is a separate, model-choice question).

Every scenario runs against the real Groq API, real Postgres, and the
real RAG index -- no mocks, same as the other eval scripts. Each
scenario runs _REPEATS_PER_SCENARIO times (real network/API latency
has genuine run-to-run variance) with a fresh reseed between runs.

Tracks agent.client's key-switch counter across each turn and reports
"clean" (zero switches) latency separately from the raw aggregate --
found necessary on the very first real run: a Groq daily-quota key
rotating out mid-turn adds a real failed-request round trip before the
next key succeeds, and one run genuinely took 63s against another,
otherwise-identical run of the same scenario at 3s. Folding those into
one aggregate would silently misrepresent what this system's real,
healthy-quota latency actually is.

Run from the project root: python -m eval.latency_benchmark
"""

import contextlib
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import businessflow.agent.client as client_module
from businessflow.agent.loop import extract_new_tool_calls, run_turn, start_conversation
from eval.tool_scoring import print_regression_delta, record_run_history
from scripts.seed_accounts import main as _reseed_demo_accounts

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REPEATS_PER_SCENARIO = 2


@dataclass
class Scenario:
    scenario_id: str
    account_id: str | None
    language: str
    user_message: str
    notes: str = ""


SCENARIOS = [
    Scenario("no_tool_greeting_en", None, "en", "Hi, who am I speaking with?", notes="Zero tool calls -- a plain reply."),
    Scenario("single_tool_balance_en", "BF-1001", "en", "What's my current balance and how many days late am I?",
              notes="One tool call (get_payment_status)."),
    Scenario("single_tool_promise_en", "BF-1001", "en", "I will pay ₹12500 by August 25 2026, I promise.",
              notes="One tool call (log_promise_to_pay)."),
    Scenario("multi_tool_settlement_hi", "BF-1004", "hi",
              "Agar main abhi ek baar mein poora loan settle kar doon, toh kitna paisa dena hoga?",
              notes="check_policy-adjacent reasoning + calculate_hypothetical -- likely 1-2 tool rounds, in Hindi."),
    Scenario("policy_question_en", None, "en", "What is your grace period policy before late fees kick in?",
              notes="A retrieval-backed reply (check_policy) with no account context."),
]


def run_scenario_once(scenario: Scenario) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _reseed_demo_accounts()

    conversation = start_conversation(language=scenario.language, account_id=scenario.account_id)
    turn_start = len(conversation)
    conversation.append({"role": "user", "content": scenario.user_message})

    key_index_before = client_module._current_key_index
    started_at = time.monotonic()
    conversation, reply = run_turn(conversation)
    elapsed_seconds = time.monotonic() - started_at
    key_switches_during_turn = client_module._current_key_index - key_index_before

    tool_calls = extract_new_tool_calls(conversation, turn_start)
    return {
        "scenario_id": scenario.scenario_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "tool_call_count": len(tool_calls),
        "tools_called": [name for name, _ in tool_calls],
        "key_switches_during_turn": key_switches_during_turn,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    results = [run_scenario_once(s) for s in SCENARIOS for _ in range(_REPEATS_PER_SCENARIO)]

    clean_results = [r for r in results if r["key_switches_during_turn"] == 0]
    degraded_count = len(results) - len(clean_results)

    def _stats(rows: list[dict]) -> dict:
        latencies = [r["elapsed_seconds"] for r in rows]
        if not latencies:
            return {"run_count": 0, "p50_seconds": None, "p90_seconds": None, "max_seconds": None, "mean_seconds": None}
        return {
            "run_count": len(rows),
            "p50_seconds": round(_percentile(latencies, 50), 3),
            "p90_seconds": round(_percentile(latencies, 90), 3),
            "max_seconds": round(max(latencies), 3),
            "mean_seconds": round(statistics.mean(latencies), 3),
        }

    by_tool_count: dict[int, list[float]] = {}
    for r in clean_results:
        by_tool_count.setdefault(r["tool_call_count"], []).append(r["elapsed_seconds"])

    summary = {
        "run_count": len(results),
        "degraded_run_count": degraded_count,  # runs where a Groq key rotated out mid-turn -- excluded from "clean"
        "clean": _stats(clean_results),
        "raw_all_runs": _stats(results),
        "clean_mean_seconds_by_tool_call_count": {
            str(n): round(statistics.mean(vals), 3) for n, vals in sorted(by_tool_count.items())
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in results:
        degraded_marker = "  [KEY ROTATED MID-TURN]" if r["key_switches_during_turn"] else ""
        print(f"  {r['elapsed_seconds']:6.3f}s  {r['scenario_id']:<28} tools={r['tools_called']}{degraded_marker}")

    print("\n=== vs previous run (clean runs only) ===")
    previous = record_run_history("latency_benchmark", summary, _RESULTS_DIR)
    previous_clean = previous.get("clean") if previous else None
    print_regression_delta(previous_clean, summary["clean"], metrics=("p50_seconds", "p90_seconds", "mean_seconds"), lower_is_better=True)

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "latency_benchmark.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
