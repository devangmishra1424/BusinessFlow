"""Unit tests for the eval monitor's regression aggregation and one-shot
main(). No real eval suites run here (that needs real Groq/Postgres/RAG,
exercised by eval/*.py's own manual runs) -- _SUITES and send_ops_alert
are monkeypatched so these test only run_all_suites'/main's own logic.

Scheduling ("when does this run") moved entirely to systemd
(businessflow-eval-monitor.timer, OnCalendar=*-*-* 21:00:00 UTC) -- there
is no more in-Python "next run time" logic to test, unlike
run_outbound_scheduler.py's own resident loop.
"""

import scripts.run_eval_monitor as monitor
from scripts.run_eval_monitor import run_all_suites


def test_run_all_suites_collects_regressions_only_from_suites_that_have_any(monkeypatch):
    monkeypatch.setattr(monitor, "_SUITES", [
        ("clean_suite", lambda: {"regressed_metrics": []}),
        ("regressed_suite", lambda: {"regressed_metrics": ["recall"]}),
        ("no_key_suite", lambda: {}),  # a suite whose main() predates regressed_metrics -- must not KeyError
    ])

    regressions = run_all_suites()

    assert regressions == {"regressed_suite": ["recall"]}


def test_run_all_suites_skips_a_suite_that_raises_without_losing_the_others(monkeypatch):
    def _boom():
        raise RuntimeError("transient API blip")

    monkeypatch.setattr(monitor, "_SUITES", [
        ("broken_suite", _boom),
        ("healthy_suite", lambda: {"regressed_metrics": ["pass_rate"]}),
    ])

    regressions = run_all_suites()

    assert regressions == {"healthy_suite": ["pass_rate"]}


def test_run_all_suites_returns_empty_dict_when_nothing_regressed(monkeypatch):
    monkeypatch.setattr(monitor, "_SUITES", [
        ("a", lambda: {"regressed_metrics": []}),
        ("b", lambda: {"regressed_metrics": []}),
    ])

    assert run_all_suites() == {}


def test_main_alerts_when_a_suite_regressed(monkeypatch):
    monkeypatch.setattr(monitor, "_SUITES", [("regressed_suite", lambda: {"regressed_metrics": ["recall"]})])
    calls = []
    monkeypatch.setattr(monitor, "send_ops_alert", lambda message: calls.append(message) or True)

    monitor.main()

    assert len(calls) == 1
    assert "regressed_suite" in calls[0]
    assert "recall" in calls[0]


def test_main_does_not_alert_when_nothing_regressed(monkeypatch):
    monkeypatch.setattr(monitor, "_SUITES", [("clean_suite", lambda: {"regressed_metrics": []})])
    calls = []
    monkeypatch.setattr(monitor, "send_ops_alert", lambda message: calls.append(message) or True)

    monitor.main()

    assert calls == []
