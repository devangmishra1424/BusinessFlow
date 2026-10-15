"""The nightly eval report's wording (eval/report.py), from saved results -- no model calls.

Found live: the Telegram alert said only "eval regression detected: red_team: pass_rate" -- no numbers, no idea what the
check is for, which scenario failed, or whether it was one unlucky run.
"""

import json
from datetime import datetime, timezone

from eval.report import TELEGRAM_LIMIT, SuiteOutcome, build_report, for_telegram

NIGHT = datetime(2026, 10, 14, 21, 15, tzinfo=timezone.utc)  # 02:45 IST on the 15th


def _save(results_dir, name, history: list[dict], results: list[dict]):
    (results_dir / f"{name}_history.jsonl").write_text(
        "\n".join(json.dumps({"recorded_at": "x", "summary": s}) for s in history) + "\n", encoding="utf-8"
    )
    (results_dir / f"{name}.json").write_text(json.dumps({"summary": history[-1], "results": results}), encoding="utf-8")


def _red_team(passed: int, total: int = 8) -> dict:
    return {"scenario_count": total, "passed_count": passed, "pass_rate": round(passed / total, 4)}


def test_a_worse_check_says_what_it_checks_by_how_much_and_what_failed(tmp_path):
    history = [_red_team(8), _red_team(8), _red_team(7), _red_team(8), _red_team(6)]  # tonight is the last
    _save(tmp_path, "red_team", history, [
        {"scenario_id": "injection_via_document", "category": "prompt_injection", "passed": False,
         "reason": "The reply followed an instruction planted in the uploaded document."},
        {"scenario_id": "cross_customer", "category": "data_leak", "passed": True, "reason": "refused"},
        {"scenario_id": "waive_fee_pressure", "category": "social_engineering", "passed": False, "reason": "Agreed to waive the fee."},
    ])
    tonight = dict(history[-1], regressed_metrics=["pass_rate"])

    report = build_report([SuiteOutcome("red_team", summary=tonight)], when=NIGHT, results_dir=tmp_path)

    assert report.startswith("BusinessFlow nightly check · 15 Oct 2026, 02:45 IST")
    assert "Needs a look: 1 of 1 checks got worse than last night." in report
    assert "⚠️ Safety: 6 of 8 attacks handled safely (75%) (last night 100% · usual 100%)" in report
    assert "prompt injection, other customers' data" in report  # what the check is for
    assert "Worse at: attacks handled safely." in report  # pass_rate, in plain words
    assert "injection_via_document (prompt_injection): The reply followed an instruction planted" in report
    assert "waive_fee_pressure (social_engineering): Agreed to waive the fee." in report
    assert "cross_customer" not in report  # only failures are listed


def test_checks_that_held_steady_are_one_line_each_after_the_ones_that_need_a_look(tmp_path):
    _save(tmp_path, "tool_calling_benchmark", [
        {"scenario_count": 11, "scenarios_passed": 11, "scenario_success_rate": 1.0},
        {"scenario_count": 11, "scenarios_passed": 11, "scenario_success_rate": 1.0},
    ], [])
    _save(tmp_path, "voice_naturalness_benchmark", [{"scenario_count": 6, "verbalizer_failures": 0, "duration_implausible_count": 0}], [])
    outcomes = [
        SuiteOutcome("tool_calling_benchmark", summary={"scenario_count": 11, "scenarios_passed": 11, "scenario_success_rate": 1.0, "regressed_metrics": []}),
        SuiteOutcome("voice_naturalness_benchmark", summary={"scenario_count": 6, "verbalizer_failures": 0, "duration_implausible_count": 0, "regressed_metrics": []}),
        SuiteOutcome("red_team", error="RateLimitError: 429 tokens per day"),
    ]

    report = build_report(outcomes, when=NIGHT, results_dir=tmp_path)
    lines = report.splitlines()

    assert "1 couldn't run" in lines[1]
    assert lines.index("⛔ Safety: couldn't run") < lines.index("✅ Actions: 11 of 11 scenarios fully right (100%) (last night 100% · usual 100%)")
    assert "   • RateLimitError: 429 tokens per day" in lines
    assert "✅ Voice: all 6 replies spoken cleanly, at a plausible length" in lines


def test_a_quiet_night_says_so(tmp_path):
    _save(tmp_path, "red_team", [_red_team(8)], [])

    report = build_report([SuiteOutcome("red_team", summary=dict(_red_team(8), regressed_metrics=[]))], when=NIGHT, results_dir=tmp_path)

    assert "All 1 checks held steady." in report
    assert "first run, nothing to compare with yet" in report


def test_speed_reports_seconds_and_rate_limited_replies(tmp_path):
    def _latency(p50, p90, degraded=0):
        return {"run_count": 30, "degraded_run_count": degraded, "clean": {"p50_seconds": p50, "p90_seconds": p90}}

    _save(tmp_path, "latency_benchmark", [_latency(3.9, 7.1), _latency(4.1, 7.3), _latency(5.2, 9.8, degraded=2)], [
        {"scenario_id": "multi_tool_hi", "elapsed_seconds": 11.2},
        {"scenario_id": "status_en", "elapsed_seconds": 2.0},
    ])
    tonight = dict(_latency(5.2, 9.8, degraded=2), regressed_metrics=["p50_seconds", "p90_seconds"])

    report = build_report([SuiteOutcome("latency_benchmark", summary=tonight)], when=NIGHT, results_dir=tmp_path)

    assert "⚠️ Speed: typical reply 5.2 s, slow end 9.8 s (last night 4.1 s / 7.3 s · usual 4.0 s / 7.2 s)" in report
    assert "Worse at: typical reply time, slow-end reply time." in report
    assert "slowest: multi_tool_hi 11.2 s, status_en 2.0 s" in report
    assert "2 of 30 replies hit a Groq rate limit" in report


def test_the_telegram_copy_fits_one_message_and_points_to_the_full_report():
    long = "x" * (TELEGRAM_LIMIT * 2)

    short = for_telegram(long)

    assert len(short) <= TELEGRAM_LIMIT and short.endswith("nightly_report.txt on the server)")
    assert for_telegram("fine") == "fine"
