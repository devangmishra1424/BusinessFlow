"""Unit tests for the eval monitor's one-shot main() and how it gathers each
suite's outcome. No real eval suites run here (that needs real
Groq/Postgres/RAG, exercised by eval/*.py's own manual runs) -- _SUITES,
send_ops_alert and the results directory are monkeypatched so these test
only run_all_suites'/main's own logic. The report's wording is
tests/test_eval_report.py.

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
def _on_the_eval_copy(monkeypatch, tmp_path):
    """main() refuses to run anywhere but the evals' own copy of the tables; these tests are about what it does once
    it does run, so they stand in for a ready copy (no database needed), and keep the report out of eval/results."""
    monkeypatch.setattr(monitor, "database_schema", lambda: "eval")
    monkeypatch.setattr(monitor, "get_connection", lambda: _FakeConnection())
    monkeypatch.setattr(monitor, "RESULTS_DIR", tmp_path)


def _alerts(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(monitor, "send_ops_alert", lambda message: calls.append(message) or True)
    return calls


def test_main_refuses_to_run_on_the_live_tables_and_says_so(monkeypatch):
    # Found live: run on the live tables at 21:00 UTC, every suite's reseed wiped what people had done on the demo.
    monkeypatch.setattr(monitor, "database_schema", lambda: None)
    ran = []
    monkeypatch.setattr(monitor, "_SUITES", [("suite", lambda: ran.append(1) or {})])
    alerts = _alerts(monkeypatch)

    with pytest.raises(SystemExit):
        monitor.main()

    assert ran == [] and len(alerts) == 1 and "did NOT run" in alerts[0]


def test_run_all_suites_returns_every_suites_outcome_with_its_regressions(monkeypatch):
    monkeypatch.setattr(monitor, "_SUITES", [
        ("clean_suite", lambda: {"regressed_metrics": []}),
        ("regressed_suite", lambda: {"regressed_metrics": ["recall"]}),
        ("no_key_suite", lambda: {}),  # a suite whose main() predates regressed_metrics -- must not KeyError
    ])

    outcomes = run_all_suites()

    assert [(o.name, o.regressed, o.error) for o in outcomes] == [
        ("clean_suite", [], None), ("regressed_suite", ["recall"], None), ("no_key_suite", [], None),
    ]


def test_a_suite_that_raises_is_recorded_with_its_error_without_losing_the_others(monkeypatch):
    def _boom():
        raise RuntimeError("transient API blip")

    monkeypatch.setattr(monitor, "_SUITES", [
        ("broken_suite", _boom),
        ("healthy_suite", lambda: {"regressed_metrics": ["pass_rate"]}),
    ])

    broken, healthy = run_all_suites()

    assert broken.error == "RuntimeError: transient API blip" and broken.summary is None
    assert healthy.regressed == ["pass_rate"]


def test_main_sends_the_report_when_a_suite_got_worse(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "_SUITES", [("regressed_suite", lambda: {"regressed_metrics": ["recall"]})])
    alerts = _alerts(monkeypatch)

    monitor.main()

    assert len(alerts) == 1
    assert "Needs a look" in alerts[0] and "taking the actions it should" in alerts[0]  # recall, in plain words
    assert (tmp_path / "nightly_report.txt").read_text(encoding="utf-8").strip() == alerts[0]


def test_main_sends_the_report_when_a_suite_could_not_run(monkeypatch):
    # Before, a suite that crashed every night was only logged -- nobody heard about it.
    def _boom():
        raise TimeoutError("Groq timed out")

    monkeypatch.setattr(monitor, "_SUITES", [("red_team", _boom)])
    alerts = _alerts(monkeypatch)

    monitor.main()

    assert len(alerts) == 1 and "couldn't run" in alerts[0] and "Groq timed out" in alerts[0]


def test_main_saves_the_report_but_sends_nothing_when_everything_held_steady(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "_SUITES", [("clean_suite", lambda: {"regressed_metrics": []})])
    alerts = _alerts(monkeypatch)

    monitor.main()

    assert alerts == []
    assert "held steady" in (tmp_path / "nightly_report.txt").read_text(encoding="utf-8")
