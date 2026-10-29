"""Turn traces: what every AI step in one borrower turn did, how long it took, and which checks it failed.

One trace per turn (a chat message, a voice note, a reminder drafted, a bank statement matched). Each step inside it --
speech-to-text, every model call, every tool, retrieval, the guardrail, text-to-speech -- adds a span with its timing
and a few numbers (tokens, scores, counts). When the turn ends, observability/checks.py grades every span with plain
rules (no model), and the whole trace is saved as ONE row in the events table (event_type 'trace'): one write per
turn, a few KB, purged after TRACE_RETENTION_DAYS. observability/trace_report.py turns them into the quality report.

What is saved is numbers, names and outcomes, never what the borrower said or what the reply said: that text is
already in the conversation's own events, and a trace is read far more widely (reports, alerts). Text a check needs
(the reply, what retrieval returned) is kept in `scratch` for the length of the turn only.

Usage:
    with trace.turn("telegram_voice", account_id=...):     # opens a trace, or joins the one already open
        with trace.span("stt", language="hi") as s:
            ...
            s["chars"] = len(text)

Concurrency: the open trace and span live in context variables. asyncio tasks and asyncio.to_thread copy the context,
so a step run in a worker thread still lands in its turn's trace (spans go into one shared list; list.append is
atomic). Steps that run with no trace open record nothing -- a script, a test of one function -- so instrumented code
never depends on a trace being there.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TRACE_RETENTION_DAYS = 30
SCHEMA_VERSION = 1


@dataclass
class Trace:
    kind: str
    account_id: str | None
    started: float = field(default_factory=time.perf_counter)
    spans: list[dict] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    # Text the checks need while the turn runs (the reply, retrieved passages, the borrower's words). Never saved.
    scratch: dict = field(default_factory=dict)


_current_trace: ContextVar[Trace | None] = ContextVar("businessflow_trace", default=None)
_current_span: ContextVar[dict | None] = ContextVar("businessflow_span", default=None)


def current() -> Trace | None:
    return _current_trace.get()


@contextmanager
def turn(kind: str, account_id: str | None = None, **attrs):
    """Opens a trace for one turn, or joins the trace already open (a channel opens it, run_turn joins it, so a voice
    turn's speech-to-text, agent and text-to-speech are one chain). Only the opener saves it. Yields the Trace."""
    existing = _current_trace.get()
    if existing is not None:
        if account_id and not existing.account_id:
            existing.account_id = account_id
        existing.attrs.update({k: v for k, v in attrs.items() if k not in existing.attrs})
        yield existing
        return
    t = Trace(kind=kind, account_id=account_id, attrs=dict(attrs))
    token = _current_trace.set(t)
    try:
        yield t
    except BaseException as e:
        t.attrs["error"] = type(e).__name__
        raise
    finally:
        _current_trace.reset(token)
        _save(t)


@contextmanager
def span(name: str, **attrs):
    """Times one step and records it on the open trace. Yields a dict: put the step's numbers in it. With no trace
    open, yields a throwaway dict and records nothing."""
    t = _current_trace.get()
    record = {"name": name, **attrs}
    if t is None:
        yield record
        return
    parent = _current_span.get()
    start = time.perf_counter()
    token = _current_span.set(record)
    try:
        yield record
    except BaseException as e:
        record["error"] = type(e).__name__
        raise
    finally:
        _current_span.reset(token)
        t.spans.append({
            "name": name,
            "at": round(start - t.started, 3),
            "seconds": round(time.perf_counter() - start, 3),
            "parent": parent.get("name") if parent else None,
            "attrs": {k: v for k, v in record.items() if k != "name"},
        })


def note(**counts) -> None:
    """Adds to counters on the innermost open span (e.g. note(rate_limit_waits=1) from inside the model call)."""
    s = _current_span.get()
    if s is not None:
        for k, v in counts.items():
            s[k] = s.get(k, 0) + v


def keep(key: str, value, *, append: bool = False) -> None:
    """Keeps text a check needs for the rest of this turn (never saved); append=True collects several under one key.
    No-op with no trace open."""
    t = _current_trace.get()
    if t is None:
        return
    if append:
        t.scratch.setdefault(key, []).append(value)
    else:
        t.scratch[key] = value


def set_attr(**attrs) -> None:
    t = _current_trace.get()
    if t is not None:
        t.attrs.update(attrs)


def answer_timing(t: Trace | None = None) -> dict | None:
    """What the chat shows small under a reply: how long the turn has taken so far, and where the time went -- listening
    (speech-to-text), the model's calls together, and each tool in the order it ran. Read from the same spans the
    quality report is built from, so the line under a reply and the report can never disagree. Names, outcomes and
    seconds only, never any text. None when no trace is open (a script, a test of one function)."""
    t = t or _current_trace.get()
    if t is None:
        return None
    spans = list(t.spans)  # a copy: a worker thread may still be appending
    listen = [s["seconds"] for s in spans if s["name"] == "stt"]
    model = [s["seconds"] for s in spans if s["name"] == "llm"]
    # Time a model call spent asleep waiting for the provider's per-minute allowance to refill (agent/loop.py notes it on
    # the call). It is inside the call's seconds, so it is taken out of the model's time and shown as its own step:
    # a 2-minute reply that spent 5 s thinking and 115 s waiting must not read as "the model took 2 minutes".
    waited = sum(s["attrs"].get("rate_limit_wait_seconds", 0) for s in spans if s["name"] == "llm")
    return {
        "seconds": round(time.perf_counter() - t.started, 3),
        "listen_seconds": round(sum(listen), 3) if listen else None,
        "model_seconds": round(max(sum(model) - waited, 0.0), 3),
        "wait_seconds": round(waited, 3),
        "model_calls": len(model),
        "tools": [
            # A tool span that raised has no outcome of its own, only the error the span recorded.
            {"tool": s["attrs"].get("tool"), "seconds": s["seconds"], "outcome": s["attrs"].get("outcome", "error")}
            for s in spans if s["name"] == "tool"
        ],
    }


def to_record(t: Trace) -> dict:
    """The saved form: numbers, names, outcomes and failed checks -- no scratch text."""
    from businessflow.observability import checks

    failed_by_span = checks.grade(t)
    spans = [{**s, "failed": failed_by_span.get(i, [])} for i, s in enumerate(t.spans)]
    turn_failed = failed_by_span.get("turn", [])
    return {
        "v": SCHEMA_VERSION,
        "kind": t.kind,
        "seconds": round(time.perf_counter() - t.started, 3),
        "attrs": t.attrs,
        "spans": spans,
        "turn_failed": turn_failed,
        "failed": sorted({f for s in spans for f in s["failed"]} | set(turn_failed)),
    }


def _save(t: Trace) -> None:
    """One events row per turn. A trace that cannot be saved must not take the borrower's reply down with it (the reply
    has already been produced), so the failure is logged with what to check, not raised."""
    if not t.spans:
        return
    import psycopg

    from businessflow.accounts import store

    try:
        record = to_record(t)
        store.log_event(t.account_id, "trace", record)
    except (psycopg.Error, RuntimeError) as e:  # RuntimeError: no DATABASE_URL (a script or test without a database)
        logger.warning("trace %s not saved (%s: %s) -- the quality report will miss this turn; check the database "
                       "connection", t.kind, type(e).__name__, e)
