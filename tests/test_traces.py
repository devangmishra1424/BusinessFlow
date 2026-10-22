"""Turn traces (observability/trace.py) and the rules that grade them (observability/checks.py).

Pure first: spans, joining, threads, and every check both firing and not firing. Then against real Postgres: a trace
is saved as one events row with no borrower text in it, and a conversation-lab replay traces every step of a turn.
"""

import asyncio
import json

import pytest

from businessflow.observability import checks, trace


def _saved(monkeypatch) -> list[dict]:
    rows = []
    monkeypatch.setattr("businessflow.accounts.store.log_event", lambda account_id, kind, details: rows.append(
        {"account_id": account_id, "kind": kind, "details": details}))
    return rows


# --- the trace itself --------------------------------------------------------------------------------------------

def test_spans_nest_time_and_save_as_one_row(monkeypatch):
    rows = _saved(monkeypatch)
    with trace.turn("chat_turn", account_id="BF-1001"):
        with trace.span("agent"):
            with trace.span("llm", model="m") as rec:
                rec["tokens_in"] = 10
                trace.note(rate_limit_waits=1)
                trace.note(rate_limit_waits=1)

    [row] = rows
    assert (row["kind"], row["account_id"]) == ("trace", "BF-1001")
    llm = next(s for s in row["details"]["spans"] if s["name"] == "llm")
    assert llm["parent"] == "agent" and llm["attrs"] == {"model": "m", "tokens_in": 10, "rate_limit_waits": 2}
    assert llm["failed"] == ["llm.rate_limited"] and row["details"]["failed"] == ["llm.rate_limited"]


def test_with_no_trace_open_nothing_is_recorded(monkeypatch):
    rows = _saved(monkeypatch)
    with trace.span("llm") as rec:
        rec["tokens_in"] = 5
    trace.note(x=1)
    trace.keep("reply", "hi")
    assert rows == []


def test_an_inner_turn_joins_the_outer_one_and_only_the_outer_saves(monkeypatch):
    rows = _saved(monkeypatch)
    with trace.turn("telegram_voice"):
        with trace.span("stt"):
            pass
        with trace.turn("chat_turn", account_id="BF-1002"):  # run_turn inside a channel's turn
            with trace.span("agent"):
                pass
    [row] = rows
    assert row["details"]["kind"] == "telegram_voice" and row["account_id"] == "BF-1002"
    assert [s["name"] for s in row["details"]["spans"]] == ["stt", "agent"]


def test_a_step_run_in_a_worker_thread_lands_in_its_turn(monkeypatch):
    rows = _saved(monkeypatch)

    def work():
        with trace.span("tool", tool="get_payment_status") as rec:
            rec["outcome"] = "ok"

    async def turn():
        with trace.turn("telegram_text"):
            await asyncio.to_thread(work)

    asyncio.run(turn())
    assert [s["name"] for s in rows[0]["details"]["spans"]] == ["tool"]


def test_text_kept_for_checks_is_never_saved(monkeypatch):
    rows = _saved(monkeypatch)
    with trace.turn("chat_turn"):
        trace.keep("reply", "Your EMI is ₹12,500, Priya.")
        trace.keep("user_text", "my secret question")
        with trace.span("guardrail", blocked=False):
            pass
    saved = json.dumps(rows[0]["details"], ensure_ascii=False)
    assert "Priya" not in saved and "secret" not in saved


def test_an_error_inside_a_turn_is_recorded_and_still_raised(monkeypatch):
    rows = _saved(monkeypatch)
    with pytest.raises(ValueError):
        with trace.turn("chat_turn"):
            with trace.span("tool"):
                raise ValueError("boom")
    assert rows[0]["details"]["attrs"]["error"] == "ValueError"
    assert rows[0]["details"]["spans"][0]["attrs"]["error"] == "ValueError"


# --- every check, both ways --------------------------------------------------------------------------------------

def _span(name, seconds=0.1, **attrs):
    return {"name": name, "seconds": seconds, "attrs": attrs}


@pytest.mark.parametrize("check, failing, passing", [
    ("stt.empty", _span("stt", chars=0), _span("stt", chars=12)),
    ("stt.low_confidence", _span("stt", avg_logprob=-1.4), _span("stt", avg_logprob=-0.3)),
    ("stt.mostly_silence", _span("stt", no_speech_prob=0.9), _span("stt", no_speech_prob=0.1)),
    ("stt.wrong_script", _span("stt", chars=20, language="hi", script="latin"), _span("stt", chars=20, language="hi", script="devanagari")),
    ("stt.too_sparse", _span("stt", audio_seconds=10, chars=5), _span("stt", audio_seconds=10, chars=80)),
    ("llm.malformed_tool_call", _span("llm", malformed_retries=1), _span("llm")),
    ("llm.rate_limited", _span("llm", rate_limit_waits=2), _span("llm")),
    ("llm.slow", _span("llm", seconds=31), _span("llm", seconds=2)),
    ("tool.error", _span("tool", outcome="error"), _span("tool", outcome="ok")),
    ("tool.blocked", _span("tool", outcome="blocked"), _span("tool", outcome="ok")),
    ("tool.slow", _span("tool", seconds=15), _span("tool", seconds=1)),
    ("retrieval.empty", _span("retrieval", results=0), _span("retrieval", results=2)),
    ("retrieval.unsure", _span("retrieval", kind="rerank", margin=0.2), _span("retrieval", kind="rerank", margin=3.1)),
    ("reply.blocked", _span("guardrail", blocked=True), _span("guardrail", blocked=False)),
    ("tts.symbols_left", _span("tts", unverbalized=True), _span("tts", unverbalized=False)),
    ("tts.implausible_length", _span("tts", chars=200, audio_seconds=0.1), _span("tts", chars=200, audio_seconds=12)),
    ("tts.slow", _span("tts", seconds=12), _span("tts", seconds=2)),
    ("compose.redrafted", _span("compose", attempts=2), _span("compose", attempts=1)),
    ("compose.fallback", _span("compose", fallback=True), _span("compose")),
    ("agent.round_cap", _span("agent", hit_round_cap=1), _span("agent")),
])
def test_each_span_check_fires_when_it_should_and_only_then(check, failing, passing):
    rule = checks.BY_NAME[check].rule
    assert rule(failing) and not rule(passing)


def _turn(**scratch):
    t = trace.Trace(kind="chat_turn", account_id=None)
    t.scratch.update(scratch)
    return t


def test_retrieved_but_unused_is_judged_on_english_replies_only():
    rule = checks.BY_NAME["retrieval.unused"].rule
    policy = ["Restructuring options include a tenure extension of up to six months."]
    assert rule(_turn(retrieved_texts=policy, reply="Sure, I can help you with that today, please wait."))
    assert not rule(_turn(retrieved_texts=policy, reply="A tenure extension of up to six months is possible."))
    assert not rule(_turn(retrieved_texts=policy, reply="आपकी अवधि छह महीने तक बढ़ाई जा सकती है।"))


def test_a_reply_in_the_other_script_is_flagged():
    rule = checks.BY_NAME["reply.wrong_language"].rule
    assert rule(_turn(user_text="मेरी EMI कितनी है?", reply="Your EMI is 12,500 rupees, due on the 25th."))
    assert not rule(_turn(user_text="meri emi kitni hai?", reply="Aapki EMI 12,500 rupees hai, 25 tareekh ko."))


def test_every_check_says_what_it_means_and_what_to_do():
    for c in checks.CHECKS:
        assert c.means and c.action and c.component, c.name
    assert len(checks.BY_NAME) == len(checks.CHECKS)


# --- against real Postgres ---------------------------------------------------------------------------------------

def _latest_trace(account_id: str) -> dict:
    from businessflow.accounts import store

    row = store.get_connection().execute(
        "select details from events where event_type = 'trace' and account_id = %s order by id desc limit 1", (account_id,)
    ).fetchone()
    return row["details"]


def test_a_lab_turn_is_traced_step_by_step(reseed_accounts):
    from eval.conversation_lab import CASES, run_case

    case = next(c for c in CASES if c.case_id == "grounded_amount_goes_out")
    assert run_case(case)["passed"]

    t = _latest_trace("BF-1001")
    names = [s["name"] for s in t["spans"]]
    assert t["kind"] == "chat_turn" and names.count("llm") == 2 and "tool" in names and "guardrail" in names and "agent" in names
    tool = next(s for s in t["spans"] if s["name"] == "tool")
    assert tool["attrs"] == {"tool": "get_payment_status", "outcome": "ok"} and tool["parent"] == "agent"
    assert t["failed"] == []


def test_a_blocked_reply_is_recorded_as_a_failed_check(reseed_accounts):
    from eval.conversation_lab import CASES, run_case

    case = next(c for c in CASES if c.case_id == "invented_amount_is_blocked")
    run_case(case)
    t = _latest_trace("BF-1001")
    guard = next(s for s in t["spans"] if s["name"] == "guardrail")
    assert guard["attrs"]["blocked"] is True and guard["attrs"]["check"] == "GroundingFailure"
    assert "reply.blocked" in t["failed"]
