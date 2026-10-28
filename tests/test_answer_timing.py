"""The small line under a reply -- how long the server took and which tools it used (observability/trace.py's
answer_timing, the `timing` field of browser_api's message responses).

No database and no model: the agent's turn is replaced by a function that opens the same spans the real one does
(llm, tool, stt), so what is checked is what the line is built from -- the numbers, the order, a tool that failed, a
reply written without the model -- and that nothing but names and numbers ever leaves in it. How the line looks on the
page is tests/js/answermeta.test.cjs; a real browser is not involved here.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from businessflow.channels import browser_api
from businessflow.observability import trace

client = TestClient(browser_api.app)


def _span(name, seconds, **attrs):
    return {"name": name, "at": 0.0, "seconds": seconds, "parent": None, "attrs": attrs}


def _trace_with(*spans) -> trace.Trace:
    t = trace.Trace(kind="web_text", account_id=None)
    t.spans.extend(spans)
    return t


# --- answer_timing itself ----------------------------------------------------------------------------------------

def test_the_timing_adds_up_the_model_calls_and_lists_each_tool_in_order():
    timing = trace.answer_timing(_trace_with(
        _span("llm", 0.8, model="m"), _span("tool", 0.04, tool="get_payment_status", outcome="ok"),
        _span("tool", 0.41, tool="check_policy", outcome="ok", passages=3), _span("llm", 0.9, model="m"),
        _span("guardrail", 0.002),
    ))

    assert timing["model_calls"] == 2 and timing["model_seconds"] == pytest.approx(1.7)
    assert [(t["tool"], t["seconds"], t["outcome"]) for t in timing["tools"]] == [
        ("get_payment_status", 0.04, "ok"), ("check_policy", 0.41, "ok"),
    ]
    assert timing["listen_seconds"] is None  # a text turn has nothing to listen to
    assert timing["seconds"] >= 0


def test_a_voice_turn_reports_the_time_spent_listening():
    timing = trace.answer_timing(_trace_with(_span("stt", 1.2, language="hi"), _span("llm", 0.7)))
    assert timing["listen_seconds"] == pytest.approx(1.2)


def test_a_tool_that_failed_says_so_and_one_that_raised_counts_as_an_error():
    timing = trace.answer_timing(_trace_with(
        _span("tool", 0.02, tool="get_payment_status", outcome="blocked"),
        _span("tool", 0.01, tool="check_policy", error="TimeoutError"),  # raised: the span never got an outcome
    ))
    assert [t["outcome"] for t in timing["tools"]] == ["blocked", "error"]


def test_with_no_trace_open_there_is_no_timing():
    assert trace.answer_timing() is None


def test_a_turn_with_no_model_calls_reports_zero_not_nothing():
    timing = trace.answer_timing(_trace_with())
    assert timing["model_calls"] == 0 and timing["model_seconds"] == 0 and timing["tools"] == []


# --- the message endpoints ---------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    # trace.turn saves each trace as an events row when it closes; these tests have no database and are not about that.
    monkeypatch.setattr("businessflow.accounts.store.log_event", lambda *a, **kw: None)


def _session(account_id=None) -> str:
    conversation_id = f"timing-test-{uuid.uuid4()}"
    browser_api._conversations[conversation_id] = {"account_id": account_id, "language": "en", "messages": []}
    return conversation_id


def _fake_agent_turn(monkeypatch, *, tool="check_policy", outcome="ok"):
    """run_turn_with_memory replaced by something that records the spans a real turn would, and returns the same shape."""
    def fake(messages, account_id):
        with trace.span("llm", model="m"):
            pass
        with trace.span("tool", tool=tool) as rec:
            rec["outcome"] = outcome
        with trace.span("llm", model="m"):
            pass
        call = {"id": "1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
        return messages + [{"role": "assistant", "tool_calls": [call]}, {"role": "assistant", "content": "The late fee is 500."}], "The late fee is 500."

    monkeypatch.setattr(browser_api, "run_turn_with_memory", fake)


def test_a_typed_question_comes_back_with_its_timing_and_the_tool_it_used(monkeypatch):
    _fake_agent_turn(monkeypatch)
    response = client.post(f"/conversations/{_session()}/messages", json={"message": "what is the late fee?"})

    assert response.status_code == 200
    body = response.json()
    assert [c["tool"] for c in body["tool_calls"]] == ["check_policy"]
    timing = body["timing"]
    assert timing["model_calls"] == 2 and timing["listen_seconds"] is None
    assert [(t["tool"], t["outcome"]) for t in timing["tools"]] == [("check_policy", "ok")]
    assert timing["seconds"] >= timing["model_seconds"] + timing["tools"][0]["seconds"]


def test_a_tool_that_did_not_go_through_is_reported_as_such(monkeypatch):
    _fake_agent_turn(monkeypatch, tool="get_payment_status", outcome="blocked")
    body = client.post(f"/conversations/{_session()}/messages", json={"message": "my balance?"}).json()
    assert [(t["tool"], t["outcome"]) for t in body["timing"]["tools"]] == [("get_payment_status", "blocked")]


def test_the_timing_holds_names_and_numbers_only_never_what_was_said_or_asked(monkeypatch):
    _fake_agent_turn(monkeypatch)
    body = client.post(f"/conversations/{_session()}/messages", json={"message": "my PAN is ABCDE1234F"}).json()
    flat = str(body["timing"])
    assert "ABCDE1234F" not in flat and "late fee" not in flat
    assert set(body["timing"]) == {"seconds", "listen_seconds", "model_seconds", "model_calls", "tools"}
    assert set(body["timing"]["tools"][0]) == {"tool", "seconds", "outcome"}


def test_a_reply_written_without_the_model_has_no_timing():
    # An already-verified chat that sends its credentials again is answered by a fixed sentence, never the model.
    body = client.post(f"/conversations/{_session('BF-1001')}/messages", json={"message": "BF-1001 482913"}).json()
    assert body["timing"] is None and body["tool_calls"] == []


def test_a_voice_turn_includes_the_time_spent_listening(monkeypatch):
    _fake_agent_turn(monkeypatch)

    def fake_listen(raw, language):
        with trace.span("stt", language=language):
            return "what is the late fee"

    monkeypatch.setattr(browser_api, "_decode_and_transcribe_voice", fake_listen)
    body = client.post(
        f"/conversations/{_session()}/messages/voice", files={"audio": ("recording.wav", b"unused", "audio/wav")},
    ).json()

    assert body["transcript"] == "what is the late fee"
    assert body["timing"]["listen_seconds"] is not None and body["timing"]["listen_seconds"] >= 0


def test_a_recording_nobody_could_make_out_has_no_timing(monkeypatch):
    monkeypatch.setattr(browser_api, "_decode_and_transcribe_voice", lambda raw, language: None)
    body = client.post(
        f"/conversations/{_session()}/messages/voice", files={"audio": ("recording.wav", b"unused", "audio/wav")},
    ).json()
    assert "couldn't make out" in body["reply"] and body["timing"] is None
