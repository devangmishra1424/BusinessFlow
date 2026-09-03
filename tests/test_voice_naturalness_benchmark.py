"""Tests for eval/voice_naturalness_benchmark.py.

_is_duration_plausible is pure logic, tested directly with no model
involved. run_scenario's real-engine tests need no gating -- TTS (Piper/
MMS-TTS) and torchaudio's SQUIM_OBJECTIVE are all uncredentialed,
freely-runnable dependencies (same reasoning test_tts.py's own real
speak_hindi tests need none), unlike the Groq/Postgres-backed eval
scripts elsewhere in this project.

main()'s own aggregation/regressed_metrics logic is tested with
run_scenario and the model itself monkeypatched to canned, deterministic
results -- isolating what's actually under test (the counting and
flagging logic) from real TTS/SQUIM's own genuine run-to-run noise (see
voice_naturalness_benchmark.py's own docstring on why that noise is
real and must not drive the regression signal).
"""

import eval.voice_naturalness_benchmark as vnb
from eval.voice_naturalness_benchmark import Scenario, _is_duration_plausible, run_scenario
from torchaudio.pipelines import SQUIM_OBJECTIVE


def test_is_duration_plausible_accepts_a_realistic_duration():
    assert _is_duration_plausible(duration_seconds=2.0, text_length=40) is True


def test_is_duration_plausible_rejects_near_zero_duration():
    # A crash/empty-audio case -- must never pass silently as "fine".
    assert _is_duration_plausible(duration_seconds=0.01, text_length=40) is False


def test_is_duration_plausible_rejects_a_runaway_duration():
    assert _is_duration_plausible(duration_seconds=100.0, text_length=10) is False


def test_is_duration_plausible_handles_empty_text_without_a_zero_division():
    assert _is_duration_plausible(duration_seconds=0.01, text_length=0) is False


def test_run_scenario_on_a_real_english_sentence_with_a_date_and_amount():
    scenario = Scenario("t1", "en", "Your payment of ₹12,500 is due on 2026-09-15.")
    model = SQUIM_OBJECTIVE.get_model()

    result = run_scenario(scenario, model)

    assert result["verbalizer_left_a_stray_pattern"] is False
    assert result["duration_plausible"] is True
    assert 0.0 <= result["stoi"] <= 1.0
    assert result["duration_seconds"] > 0


def test_run_scenario_on_a_real_hindi_sentence_with_a_date_and_amount():
    scenario = Scenario("t2", "hi", "आपकी ₹12,500 की किस्त 2026-09-15 को देय है।")
    model = SQUIM_OBJECTIVE.get_model()

    result = run_scenario(scenario, model)

    assert result["verbalizer_left_a_stray_pattern"] is False
    assert result["duration_plausible"] is True
    assert 0.0 <= result["stoi"] <= 1.0
    assert result["duration_seconds"] > 0


def test_main_flags_verbalizer_and_duration_failures_but_not_squim_score_noise(monkeypatch, tmp_path):
    # Two canned scenario results: one with a real correctness failure
    # (both flags tripped), one clean -- everything else about main()'s
    # own SQUIM-score bookkeeping still runs, but must not itself end up
    # in regressed_metrics (see the module's own docstring for why).
    # _RESULTS_DIR redirected to tmp_path -- main() writes real history/
    # results files, which must never be the genuine eval/results/ ones
    # this test's canned, fake data would otherwise corrupt.
    monkeypatch.setattr(vnb, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(vnb, "SCENARIOS", [
        Scenario("clean", "en", "Hi there."),
        Scenario("broken", "en", "Rs. 500 due 2026-09-15"),
    ])
    canned_results = {
        "clean": {
            "scenario_id": "clean", "language": "en", "text": "Hi there.", "verbalized_text": "Hi there.",
            "duration_seconds": 1.0, "duration_plausible": True, "verbalizer_left_a_stray_pattern": False,
            "stoi": 0.99, "pesq": 4.0, "sisdr": 30.0,
        },
        "broken": {
            "scenario_id": "broken", "language": "en", "text": "Rs. 500 due 2026-09-15", "verbalized_text": "Rs. 500 due 2026-09-15",
            "duration_seconds": 0.01, "duration_plausible": False, "verbalizer_left_a_stray_pattern": True,
            "stoi": 0.5, "pesq": 1.0, "sisdr": 1.0,
        },
    }
    monkeypatch.setattr(vnb, "run_scenario", lambda scenario, model: canned_results[scenario.scenario_id])
    monkeypatch.setattr(vnb.SQUIM_OBJECTIVE, "get_model", lambda: None)

    summary = vnb.main()

    assert summary["verbalizer_failures"] == 1
    assert summary["duration_implausible_count"] == 1
    assert sorted(summary["regressed_metrics"]) == ["duration_implausible_count", "verbalizer_failures"]


def test_main_reports_no_regressed_metrics_when_everything_is_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(vnb, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(vnb, "SCENARIOS", [Scenario("clean", "en", "Hi there.")])
    monkeypatch.setattr(vnb, "run_scenario", lambda scenario, model: {
        "scenario_id": "clean", "language": "en", "text": "Hi there.", "verbalized_text": "Hi there.",
        "duration_seconds": 1.0, "duration_plausible": True, "verbalizer_left_a_stray_pattern": False,
        "stoi": 0.99, "pesq": 4.0, "sisdr": 30.0,
    })
    monkeypatch.setattr(vnb.SQUIM_OBJECTIVE, "get_model", lambda: None)

    summary = vnb.main()

    assert summary["regressed_metrics"] == []
