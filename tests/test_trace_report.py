"""The quality report (observability/trace_report.py): pure building from traces, then reading a real window with its
later-truth outcomes and safeguard gates.
"""

from businessflow.observability.trace_report import build_report, collect, summary_lines


def _trace(kind="web_text", spans=(), turn_failed=(), seconds=2.0, **attrs):
    spans = [dict(s) for s in spans]
    return {"kind": kind, "spans": spans, "turn_failed": list(turn_failed), "seconds": seconds, "attrs": attrs,
            "failed": sorted({f for s in spans for f in s.get("failed", [])} | set(turn_failed)), "id": 1}


def _s(name, seconds=0.5, parent=None, failed=(), at=0.0, **attrs):
    return {"name": name, "seconds": seconds, "parent": parent, "at": at, "attrs": attrs, "failed": list(failed)}


def _turn(blocked=False, tool_outcome="ok"):
    return _trace(spans=[
        _s("agent", 1.5),
        _s("llm", 1.0, parent="agent", at=0.1, offered_tools=True, tokens_in=1000),
        _s("tool", 0.2, parent="agent", at=1.1, tool="get_payment_status", outcome=tool_outcome,
           failed=["tool.error"] if tool_outcome == "error" else []),
        _s("guardrail", 0.01, parent="agent", at=1.4, blocked=blocked, failed=["reply.blocked"] if blocked else []),
    ])


def test_too_few_steps_is_said_not_shown_as_healthy():
    report = build_report([_turn()])
    assert {v["status"] for v in report["verdict"]} == {"not_enough_data"}


def test_the_weakest_component_comes_first_with_its_reason_and_action():
    traces = [_turn() for _ in range(18)] + [_turn(blocked=True), _turn(blocked=True)]
    report = build_report(traces)
    first = report["verdict"][0]
    assert first["component"] == "Reply truthfulness" and first["status"] == "red"  # 10% blocked, a critical check
    truth = next(c for c in report["components"] if c["component"] == "Reply truthfulness")
    assert truth["failed_rate"] == 0.1 and truth["checks"][0]["name"] == "reply.blocked" and truth["checks"][0]["action"]
    tools = next(c for c in report["components"] if c["component"] == "Tools")
    assert tools["status"] == "green" and tools["actions"][0] == {"name": "get_payment_status", "count": 20, "failed_rate": 0.0, "p50_seconds": 0.2}


def test_a_component_that_got_worse_shows_where_it_was():
    before = [_turn() for _ in range(10)]
    now = [_turn(tool_outcome="error")] + [_turn() for _ in range(9)]
    tools = next(c for c in build_report(now, before)["components"] if c["component"] == "Tools")
    assert (tools["failed_rate"], tools["previous_failed_rate"]) == (0.1, 0.0)


def test_a_chain_shows_where_its_time_goes_and_where_failures_start():
    traces = [_trace(kind="telegram_voice", spans=[
        _s("stt", 3.0, at=0.0, failed=["stt.low_confidence"] if i == 0 else []),
        _s("agent", 2.0, at=3.0), _s("llm", 1.5, parent="agent", at=3.1), _s("guardrail", 0.0, parent="agent", at=4.9),
        _s("tts", 1.0, at=5.0),
    ]) for i in range(4)]
    [chain] = build_report(traces)["chains"]
    assert chain["kind"] == "telegram_voice" and chain["success_rate"] == 0.75
    assert chain["failures_start_at"] == {"stt": 1}
    assert max(chain["time_share"], key=chain["time_share"].get) == "stt"


def test_the_nightly_lines_put_a_broken_safeguard_first():
    report = build_report([_turn() for _ in range(6)], gates=[{"name": "Every model reply went through the guardrail", "passed": False, "detail": "1 turn"}])
    lines = summary_lines(report)
    assert lines[0].startswith("⛔") and any("Tools" in line for line in lines)


def test_a_real_window_reads_traces_outcomes_and_gates(reseed_accounts):
    from eval.conversation_lab import CASES, run_case

    for case in CASES:
        if case.case_id in ("grounded_amount_goes_out", "promise_read_back_then_recorded"):
            run_case(case)
    report = collect(days=1)

    assert report["window"]["traces"] >= 1  # each lab case reseeds the demo accounts, which clears their earlier traces
    names = {c["component"] for c in report["components"]}
    assert {"Model calls", "Tools", "Reply truthfulness"} <= names
    gates = {g["name"]: g["passed"] for g in report["gates"]}
    assert gates["Every model reply went through the guardrail"] is True
    assert gates["No promise recorded from chat without reading the terms back"] is True
    assert {"repeats", "bank_matcher", "promises", "reminders"} <= set(report["outcomes"])


def test_old_traces_are_purged(reseed_accounts):
    from businessflow.accounts import store

    conn = store.get_connection()
    conn.execute("insert into events (event_type, details, created_at) values ('trace', '{}', now() - interval '40 days')")
    assert store.purge_old_traces() >= 1
    assert conn.execute("select count(*) as n from events where event_type = 'trace' and created_at < now() - interval '30 days'").fetchone()["n"] == 0


def test_waiting_on_a_rate_limit_is_slow_not_wrong():
    slow = [_trace(spans=[_s("llm", 25, offered_tools=True, failed=["llm.rate_limited", "llm.slow"])]) for _ in range(6)]
    report = build_report(slow)
    model = next(c for c in report["components"] if c["component"] == "Model calls")
    assert (model["status"], model["failed_rate"], model["slow_rate"]) == ("slow", 0.0, 1.0)
    assert report["verdict"][0]["why"].startswith("answers right, but slow")
    [chain] = report["chains"]
    assert (chain["success_rate"], chain["slowed_rate"]) == (1.0, 1.0)


def test_a_blocked_call_is_a_protection_that_worked_not_a_failure():
    traces = [_turn(tool_outcome="blocked") for _ in range(6)]
    for t in traces:
        for s in t["spans"]:
            if s["name"] == "tool":
                s["failed"] = ["tool.blocked"]
    tools = next(c for c in build_report(traces)["components"] if c["component"] == "Tools")
    assert (tools["status"], tools["failed_rate"], tools["protections"]) == ("green", 0.0, 6)
    assert next(k for k in tools["checks"] if k["name"] == "tool.blocked")["kind"] == "protection"
