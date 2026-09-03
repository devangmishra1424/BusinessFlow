"""Unit tests for the eval monitor's pure timing logic and regression
aggregation -- no real eval suites run here (that needs real Groq/
Postgres/RAG, exercised by eval/*.py's own manual runs), same reasoning
test_outbound_scheduler.py already applies to its own scheduler. The
loop itself (run_eval_monitor.main()) is a real, always-running process
by design and isn't exercised here either.
"""

from datetime import datetime, timezone

from scripts.run_eval_monitor import _RUN_HOUR_UTC, _next_run_time, run_all_suites


def test_next_run_time_before_the_run_hour_is_later_today():
    now = datetime(2026, 8, 24, 3, 0, tzinfo=timezone.utc)  # before 21:00 UTC

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 24, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)


def test_next_run_time_after_the_run_hour_is_tomorrow():
    now = datetime(2026, 8, 24, 23, 0, tzinfo=timezone.utc)  # after 21:00 UTC

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 25, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)


def test_next_run_time_exactly_at_the_run_hour_is_tomorrow():
    # <=, not < -- once the run hour has actually arrived, the next
    # scheduled run is tomorrow's, not "right now" again.
    now = datetime(2026, 8, 24, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 25, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)


def test_run_all_suites_collects_regressions_only_from_suites_that_have_any(monkeypatch):
    import scripts.run_eval_monitor as monitor

    monkeypatch.setattr(monitor, "_SUITES", [
        ("clean_suite", lambda: {"regressed_metrics": []}),
        ("regressed_suite", lambda: {"regressed_metrics": ["recall"]}),
        ("no_key_suite", lambda: {}),  # a suite whose main() predates regressed_metrics -- must not KeyError
    ])

    regressions = run_all_suites()

    assert regressions == {"regressed_suite": ["recall"]}


def test_run_all_suites_skips_a_suite_that_raises_without_losing_the_others(monkeypatch):
    import scripts.run_eval_monitor as monitor

    def _boom():
        raise RuntimeError("transient API blip")

    monkeypatch.setattr(monitor, "_SUITES", [
        ("broken_suite", _boom),
        ("healthy_suite", lambda: {"regressed_metrics": ["pass_rate"]}),
    ])

    regressions = run_all_suites()

    assert regressions == {"healthy_suite": ["pass_rate"]}


def test_run_all_suites_returns_empty_dict_when_nothing_regressed(monkeypatch):
    import scripts.run_eval_monitor as monitor

    monkeypatch.setattr(monitor, "_SUITES", [
        ("a", lambda: {"regressed_metrics": []}),
        ("b", lambda: {"regressed_metrics": []}),
    ])

    assert run_all_suites() == {}
