"""The benchmark's own arithmetic, checked on synthetic data with known answers (no model, no network), the way
eval/analyze_results.py was verified: a benchmark nobody has checked is just a number generator."""

import json
import math
import re

import numpy as np
import pytest

from eval import tts_benchmark as tb


def test_norm_hi_ignores_spelling_variants_that_are_not_errors():
    assert tb.norm_hi("पाँच हज़ार रुपये।") == tb.norm_hi("पांच हजार रुपये")
    assert tb.norm_hi("UPI, से  भुगतान!") == "upi से भुगतान"


def test_rms_dbfs_of_known_signals():
    assert tb.rms_dbfs(np.ones(1000, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)
    assert tb.rms_dbfs(np.full(1000, 0.1, dtype=np.float32)) == pytest.approx(-20.0, abs=1e-3)
    assert tb.rms_dbfs(np.zeros(10, dtype=np.float32)) == -120.0
    assert tb.rms_dbfs(np.array([], dtype=np.float32)) == -120.0


def test_edge_silence_is_measured_at_both_ends():
    sr = 16000
    audio = np.concatenate([np.zeros(sr // 5), 0.5 * np.ones(sr), np.zeros(sr // 2)]).astype(np.float32)  # 200 ms, 1 s, 500 ms
    lead, trail = tb.edge_silence_ms(audio, sr)
    assert lead == pytest.approx(200, abs=10)
    assert trail == pytest.approx(500, abs=10)


def test_edge_silence_of_all_silence_is_the_whole_clip():
    lead, trail = tb.edge_silence_ms(np.zeros(16000, dtype=np.float32), 16000)
    assert lead == trail == 1000


def test_coefficient_of_variation():
    assert tb.coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert tb.coefficient_of_variation([1.0, 3.0]) == pytest.approx(math.sqrt(2) / 2)  # stdev sqrt(2), mean 2
    assert tb.coefficient_of_variation([5.0]) == 0.0
    assert tb.coefficient_of_variation([]) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mean, lo, hi = tb.bootstrap_ci(values)
    assert mean == pytest.approx(0.45)
    assert lo < mean < hi
    assert (mean, lo, hi) == tb.bootstrap_ci(values)  # same seed, same answer
    assert tb.bootstrap_ci([0.5] * 8) == (0.5, 0.5, 0.5)  # no variation, no width
    assert all(math.isnan(x) for x in tb.bootstrap_ci([]))


def test_paired_ci_counts_who_wins_where_lower_is_better():
    a = {"s1": 0.10, "s2": 0.20, "s3": 0.30, "s4": 0.30}
    b = {"s1": 0.20, "s2": 0.20, "s3": 0.10, "s4": 0.40, "only_b": 9.0}
    mean, lo, hi, better, tie, worse = tb.paired_ci(a, b)
    assert (better, tie, worse) == (2, 1, 1)  # s1 and s4 better, s2 tie, s3 worse; "only_b" is ignored
    assert mean == pytest.approx((-0.10 + 0.0 + 0.20 - 0.10) / 4)
    assert lo <= mean <= hi


def test_a_clearly_better_engine_has_a_paired_interval_below_zero():
    a = {f"s{i}": 0.05 for i in range(20)}
    b = {f"s{i}": 0.15 + (i % 3) * 0.01 for i in range(20)}
    mean, lo, hi, better, tie, worse = tb.paired_ci(a, b)
    assert hi < 0 and better == 20 and worse == 0


def test_the_fixture_is_well_formed():
    data = json.loads(tb.FIXTURE.read_text(encoding="utf-8"))["sentences"]
    ids = [s["id"] for s in data]
    assert len(ids) == len(set(ids)), "sentence ids must be unique"
    assert {s["category"] for s in data} == {*tb.CLARITY_CATEGORIES, "hinglish_roman"}
    for s in data:
        assert s["id"].startswith(s["category"]) and s["text"].strip()
        has_devanagari = bool(re.search(r"[ऀ-ॿ]", s["text"]))
        assert has_devanagari == (s["category"] != "hinglish_roman"), s["id"]
    assert all(sum(1 for s in data if s["category"] == c) >= 4 for c in tb.CLARITY_CATEGORIES)


def test_the_markdown_report_renders_a_synthetic_summary():
    ci = {"mean": 0.07, "lo": 0.05, "hi": 0.09}
    engine = {"clips": 10, "failed": 0, "empty": 0, "errors": [], "load_s": 5.0, "by_category": {c: {"clips": 3, "ok": 3, "cer": ci, "wer": ci} for c in tb.CLARITY_CATEGORIES},
              "pooled_cer": ci, "pooled_wer": ci, "duration_cv_mean": 0.03, "identical_transcript_share": 0.9, "rms_dbfs_std": 1.2, "clipped_clips": 0, "opus_over_clips": 0,
              "lead_ms_mean": 80.0, "trail_ms_mean": 120.0, "rtf_mean": 0.07, "sample_rates": [22050], "memory": {"growth_mb": 12}}
    summary = {"baseline_rev": "abc1234", "engines": {"new_piper": engine}, "leakage": {"old": {"digit_runs_left": 9, "latin_words_left": 7}, "new": {"digit_runs_left": 0, "latin_words_left": 7}},
               "criteria": [{"name": "C1 robustness", "passed": True, "detail": "ok"}, {"name": "C2 clarity", "passed": False, "detail": "worse"}], "all_passed": False}
    md = tb._markdown(summary)
    assert "C1 robustness**: PASS" in md and "C2 clarity**: FAIL" in md and "All criteria passed: False" in md
    assert "| new_piper |" in md
