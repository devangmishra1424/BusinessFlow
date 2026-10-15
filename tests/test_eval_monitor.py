"""Unit tests for the eval monitor's regression aggregation and one-shot
main(). No real eval suites run here (that needs real Groq/Postgres/RAG,
exercised by eval/*.py's own manual runs) -- _SUITES and send_ops_alert
are monkeypatched so these test only run_all_suites'/main's own logic.

Scheduling ("when does this run") moved entirely to systemd
(businessflow-eval-monitor.timer, OnCalendar=*-*-* 21:00:00 UTC) -- there
is no more in-Python "next run time" logic to test, unlike
run_outbound_scheduler.py's own resident loop.
"""

import pytest

import scripts.run_eval_monitor as monitor
from scripts.run_eval_monitor import run_all_suites


class _FakeConnection:
    def execute(self, query, params=None):
        return None


@pytest.fixture(autouse=True)
def _on_the_eval_copy(monkeypatch):
    """main() refuses to run anywhere but the evals' own copy of the tables; these tests are about what it does once
    it does run, so they stand in for a ready copy (no database needed)."""
    monkeypatch.setattr(monitor, "database_schema", lambda: "eval")
    monkeypatch.setattr(monitor, "get_connection", lambda: _FakeConnection())


def test_main_refuses_to_run_on_the_live_tables_and_says_so(monkeypatch):
    # Found live: run on the live tables at 21:00 UTC, every suite's reseed wiped what people had done on the demo.
    monkeypatch.setattr(monitor, "database_schema", lambda: None)
    ran, alerts = [], []
    monkeypatch.setattr(monitor, "_SUITES", [("suite", lambda: ran.append(1) or {})])
    monkeypatch.setattr(monitor, "send_ops_alert", lambda message: alerts.append(message) or True)

    with pytest.raises(SystemExit):
        monitor.main()

    assert ran == [] and len(alerts) == 1 and "did NOT run" in alerts[0]


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
