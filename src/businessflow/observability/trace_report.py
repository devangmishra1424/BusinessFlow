"""The quality report: how every AI component and every chain of them did, from the turn traces (observability/trace.py)
and the system's own later events.

Four levels, most important first:

1. Verdict -- each component green / amber / red, the weakest first. A component with too few steps to judge says so
   ("not enough data") instead of looking green. Gates are invariants that must hold in the data whatever the
   averages say (every reply was checked; no promise from chat without a read-back).
2. Components and their actions (each tool, each speech engine) -- volume, the share of steps failing a check and
   which checks, p50/p90 time, tokens, and the change from the previous period of the same length.
3. Chains (a web voice turn, a Telegram text turn, a reminder, a bank import) -- success rate, where the time goes
   step by step, and the step where failures start.
4. Evidence -- the worst traces, by id, so every number leads to a case.

Plus outcomes judged later by what actually happened, at no model cost: borrowers repeating themselves right after a
voice note (a mis-hearing), people accepting or changing the bank matcher's suggestions, promises kept by how they were
made, reminders delivered and paid within three days by tone.

build_report is pure (rows in, report out); collect() does the reading -- bounded queries over one window.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from businessflow.observability.checks import BY_NAME, CHECKS

MAX_TRACES = 5000  # one window's traces read at most; the report says when it was capped
MIN_STEPS_FOR_VERDICT = 5
AMBER_RATE, RED_RATE = 0.05, 0.15

COMPONENT_OF_SPAN = {
    "stt": "Speech-to-text", "llm": "Model calls", "agent": "Model calls", "tool": "Tools", "retrieval": "Retrieval",
    "guardrail": "Reply truthfulness", "tts": "Text-to-speech", "compose": "Reminder writer", "bank_match": "Bank matching",
}
COMPONENT_ORDER = ["Speech-to-text", "Model calls", "Tools", "Retrieval", "Reply truthfulness", "Text-to-speech",
                   "Reminder writer", "Bank matching"]
# What names an "action" inside a component.
ACTION_ATTR = {"tool": "tool", "tts": "engine", "llm": "offered_tools", "compose": "reminder_kind", "stt": "language"}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 3)


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _status(steps: int, failed_rate: float | None, critical_failures: int, critical_rate: float | None) -> str:
    if steps < MIN_STEPS_FOR_VERDICT:
        return "not_enough_data"
    if (critical_rate or 0) >= AMBER_RATE or (failed_rate or 0) >= RED_RATE:
        return "red"
    if critical_failures or (failed_rate or 0) >= AMBER_RATE:
        return "amber"
    return "green"


def _components(traces: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in traces:
        for s in t["spans"]:
            comp = COMPONENT_OF_SPAN.get(s["name"])
            if comp is None:
                continue
            c = out.setdefault(comp, {"steps": 0, "failed_steps": 0, "critical_steps": 0, "seconds": [],
                                      "tokens_in": 0, "tokens_out": 0, "checks": Counter(), "actions": defaultdict(list)})
            if s["name"] == "agent":  # the loop as a whole: counted for its checks, not as a model call
                for f in s.get("failed", []):
                    c["checks"][f] += 1
                continue
            c["steps"] += 1
            c["seconds"].append(s.get("seconds", 0))
            a = s.get("attrs") or {}
            c["tokens_in"] += a.get("tokens_in") or 0
            c["tokens_out"] += a.get("tokens_out") or 0
            failed = s.get("failed", [])
            if failed:
                c["failed_steps"] += 1
            if any(BY_NAME[f].critical for f in failed if f in BY_NAME):
                c["critical_steps"] += 1
            for f in failed:
                c["checks"][f] += 1
            key = ACTION_ATTR.get(s["name"])
            action = a.get(key) if key else s["name"]
            if s["name"] == "llm":
                action = "with tools" if action else "final answer"
            c["actions"][str(action if action is not None else "-")].append((s.get("seconds", 0), bool(failed)))
        for f in t.get("turn_failed", []):
            comp = BY_NAME[f].component if f in BY_NAME else None
            if comp:
                c = out.setdefault(comp, {"steps": 0, "failed_steps": 0, "critical_steps": 0, "seconds": [],
                                          "tokens_in": 0, "tokens_out": 0, "checks": Counter(), "actions": defaultdict(list)})
                c["checks"][f] += 1
    return out


def _chains(traces: list[dict]) -> list[dict]:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        by_kind[t["kind"]].append(t)
    chains = []
    for kind, ts in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        time_by_step: Counter = Counter()
        first_fail: Counter = Counter()
        ok = 0
        for t in ts:
            top = [s for s in t["spans"] if s.get("parent") in (None, "agent")]
            for s in top:
                if s["name"] != "agent":
                    time_by_step[s["name"]] += s.get("seconds", 0)
            failing = [s for s in t["spans"] if s.get("failed")]
            if failing or t.get("turn_failed") or (t.get("attrs") or {}).get("error"):
                first = min(failing, key=lambda s: s.get("at", 0))["name"] if failing else "turn"
                first_fail[first] += 1
            else:
                ok += 1
        total_time = sum(time_by_step.values()) or 1
        secs = [t.get("seconds", 0) for t in ts]
        chains.append({
            "kind": kind, "turns": len(ts), "success_rate": _rate(ok, len(ts)),
            "p50_seconds": _pct(secs, 0.5), "p90_seconds": _pct(secs, 0.9),
            "time_share": {k: round(v / total_time, 3) for k, v in time_by_step.most_common()},
            "failures_start_at": dict(first_fail.most_common()),
        })
    return chains


def build_report(traces: list[dict], previous: list[dict] | None = None, outcomes: dict | None = None,
                 gates: list[dict] | None = None, window: dict | None = None) -> dict:
    """traces: saved trace records (each with "id" and "created_at" added). previous: the period before, for trends."""
    comps = _components(traces)
    prev = _components(previous or [])
    components = []
    for name in COMPONENT_ORDER:
        c = comps.get(name)
        if c is None:
            continue
        failed_rate = _rate(c["failed_steps"], c["steps"])
        critical_rate = _rate(c["critical_steps"], c["steps"])
        p = prev.get(name)
        checks = [{
            "name": n, "count": k, "rate": _rate(k, c["steps"]), "means": BY_NAME[n].means, "action": BY_NAME[n].action,
            "critical": BY_NAME[n].critical,
        } for n, k in c["checks"].most_common() if n in BY_NAME]
        actions = [{
            "name": a, "count": len(v), "failed_rate": _rate(sum(f for _, f in v), len(v)),
            "p50_seconds": _pct([s for s, _ in v], 0.5),
        } for a, v in sorted(c["actions"].items(), key=lambda kv: -len(kv[1]))]
        components.append({
            "component": name, "status": _status(c["steps"], failed_rate, c["critical_steps"], critical_rate),
            "steps": c["steps"], "failed_rate": failed_rate, "previous_failed_rate": _rate(p["failed_steps"], p["steps"]) if p else None,
            "p50_seconds": _pct(c["seconds"], 0.5), "p90_seconds": _pct(c["seconds"], 0.9),
            "tokens_in": c["tokens_in"] or None, "tokens_out": c["tokens_out"] or None,
            "checks": checks, "actions": actions,
        })
    severity = {"red": 0, "amber": 1, "green": 2, "not_enough_data": 3}
    verdict = sorted(
        ({"component": c["component"], "status": c["status"],
          "why": (f"{c['checks'][0]['count']} × {c['checks'][0]['means']}" if c["checks"] else
                  ("too few steps to judge" if c["status"] == "not_enough_data" else "every step passed its checks"))}
         for c in components),
        key=lambda v: (severity[v["status"]], COMPONENT_ORDER.index(v["component"])),
    )
    evidence = sorted(
        (t for t in traces if t.get("failed") or (t.get("attrs") or {}).get("error")),
        key=lambda t: (-sum(BY_NAME[f].critical for f in t.get("failed", []) if f in BY_NAME), -len(t.get("failed", [])), -t.get("seconds", 0)),
    )[:15]
    return {
        "window": {**(window or {}), "traces": len(traces)},
        "verdict": verdict,
        "gates": gates or [],
        "components": components,
        "chains": _chains(traces),
        "outcomes": outcomes or {},
        "evidence": [{"id": t.get("id"), "at": t.get("created_at"), "kind": t["kind"], "account_id": t.get("account_id"),
                      "seconds": t.get("seconds"), "failed": t.get("failed", []), "error": (t.get("attrs") or {}).get("error")}
                     for t in evidence],
        "checks_defined": [{"name": c.name, "component": c.component, "means": c.means, "critical": c.critical} for c in CHECKS],
    }


# --- reading -----------------------------------------------------------------------------------------------------

_WORDS = re.compile(r"\w{3,}")


def _repeat_rate(conn, since: datetime) -> dict:
    """Borrowers repeating themselves: two of their messages within two minutes sharing most of their words. After a
    voice note that is most often a mis-hearing; after text, a reply that missed the point."""
    rows = conn.execute(
        "select account_id, created_at, details->>'content' as text from events where event_type = 'user_message' "
        "and account_id is not null and created_at > %s order by account_id, created_at limit 20000",
        (since,),
    ).fetchall()
    repeats = 0
    for a, b in zip(rows, rows[1:]):
        if a["account_id"] != b["account_id"] or (b["created_at"] - a["created_at"]).total_seconds() > 120:
            continue
        wa, wb = set(_WORDS.findall((a["text"] or "").lower())), set(_WORDS.findall((b["text"] or "").lower()))
        if wa and wb and len(wa & wb) / len(wa | wb) >= 0.5:
            repeats += 1
    return {"messages": len(rows), "repeated": repeats, "rate": _rate(repeats, len(rows)),
            "how": "Two messages from one borrower within 2 minutes sharing at least half their words."}


def _bank_matcher(conn, since: datetime) -> dict:
    """The matcher judged by what people did with its output: suggestions kept as suggested or changed."""
    lines = conn.execute(
        "select status, decided_by from bank_lines where created_at > %s", (since,),
    ).fetchall()
    decided = conn.execute(
        "select details->>'suggested_account_id' as suggested, account_id, details->>'was_status' as was from events "
        "where event_type = 'bank_credit_applied' and details ? 'was_status' and created_at > %s", (since,),
    ).fetchall()
    suggestions = [d for d in decided if d["was"] == "suggested" and d["suggested"]]
    kept = sum(1 for d in suggestions if d["suggested"] == d["account_id"])
    return {
        "lines": len(lines), "auto_posted": sum(1 for r in lines if r["decided_by"] == "bank statement (automatic)"),
        "waiting": sum(1 for r in lines if r["status"] in ("suggested", "unmatched")),
        "suggestions_decided": len(suggestions), "suggestions_kept": kept, "suggestion_precision": _rate(kept, len(suggestions)),
        "how": "Of the matcher's suggestions a person posted, how many went to the account it suggested.",
    }


def _promises_by_channel(conn, since: datetime) -> dict:
    rows = conn.execute(
        "select coalesce(made_via, 'unknown') as via, plan_id, bool_or(resolution = 'broken') as broken, "
        "bool_and(resolution = 'kept') as kept from promises where resolved_on > %s group by 1, 2", (since.date(),),
    ).fetchall()
    out: dict = {}
    for r in rows:
        o = out.setdefault(r["via"], {"kept": 0, "broken": 0})
        if r["broken"]:
            o["broken"] += 1
        elif r["kept"]:
            o["kept"] += 1
    for o in out.values():
        o["kept_rate"] = _rate(o["kept"], o["kept"] + o["broken"])
    return {"by_channel": out, "how": "Promises settled in the window, kept out of kept plus broken, by how they were made."}


def _reminders(conn, since: datetime) -> dict:
    rows = conn.execute(
        """select e.details->>'tone' as tone, (e.details->>'delivered_via_telegram')::boolean as delivered,
                  exists (select 1 from payment_history p where p.account_id = e.account_id
                          and p.payment_date between e.created_at::date and e.created_at::date + 3) as paid
           from events e where e.event_type = 'reminder_sent' and e.created_at > %s and e.created_at < now() - interval '3 days'
           limit 20000""",
        (since,),
    ).fetchall()
    by_tone: dict = {}
    for r in rows:
        t = by_tone.setdefault(r["tone"] or "unrecorded", {"sent": 0, "paid_within_3_days": 0})
        t["sent"] += 1
        t["paid_within_3_days"] += bool(r["paid"])
    for t in by_tone.values():
        t["rate"] = _rate(t["paid_within_3_days"], t["sent"])
    all_rows = conn.execute(
        "select (details->>'delivered_via_telegram')::boolean as delivered from events where event_type = 'reminder_sent' "
        "and created_at > %s limit 20000", (since,),
    ).fetchall()
    return {
        "sent": len(all_rows), "delivered": sum(1 for r in all_rows if r["delivered"]),
        "delivered_rate": _rate(sum(1 for r in all_rows if r["delivered"]), len(all_rows)),
        "paid_within_3_days_by_tone": by_tone,
        "how": "Delivered: reached the borrower on Telegram (an unlinked account's reminders are only logged). Paid: any "
               "payment within 3 days of a reminder sent at least 3 days ago.",
    }


def _gates(conn, since: datetime, traces: list[dict]) -> list[dict]:
    """Invariants the data must show whatever the rates: if one fails, something bypassed a safeguard."""
    unchecked = [t for t in traces if any(s["name"] == "llm" for s in t["spans"])
                 and not any(s["name"] == "guardrail" for s in t["spans"]) and not (t.get("attrs") or {}).get("error")]
    no_read_back = conn.execute(
        """select count(*) as n from promises p where p.made_via = 'chat' and p.part_no = 1 and p.created_at > %s
           and not exists (select 1 from events e where e.account_id = p.account_id and e.event_type = 'promise_read_back'
                           and e.created_at between p.created_at - interval '30 minutes' and p.created_at)""",
        (since,),
    ).fetchone()["n"]
    return [
        {"name": "Every model reply went through the guardrail", "passed": not unchecked,
         "detail": f"{len(unchecked)} turn(s) with a model reply and no guardrail step" if unchecked else "all checked"},
        {"name": "No promise recorded from chat without reading the terms back", "passed": no_read_back == 0,
         "detail": f"{no_read_back} chat promise(s) with no read-back in the 30 minutes before" if no_read_back else "none"},
    ]


def collect(days: int = 7, conn=None, since: datetime | None = None) -> dict:
    """Reads one window (the last `days`, or from `since`), the period of the same length before it for trends, and the
    later-truth outcomes, and builds the report. Bounded: at most MAX_TRACES traces per window."""
    from businessflow.accounts.db import get_connection

    conn = conn or get_connection()
    # The database's clock, not this machine's: rows are stamped by the database, and the two clocks differ (found in a
    # test, where a trace saved a moment ago fell after a window ending at this machine's "now").
    now = conn.execute("select now() as now").fetchone()["now"]
    start = since or now - timedelta(days=days)
    length = now - start

    def read(a: datetime, b: datetime) -> list[dict]:
        rows = conn.execute(
            "select id, account_id, created_at, details from events where event_type = 'trace' and created_at > %s "
            "and created_at <= %s order by id desc limit %s",
            (a, b, MAX_TRACES),
        ).fetchall()
        return [{**r["details"], "id": r["id"], "account_id": r["account_id"], "created_at": r["created_at"].isoformat()}
                for r in rows]

    traces = read(start, now)
    previous = read(start - length, start)
    outcomes = {
        "repeats": _repeat_rate(conn, start), "bank_matcher": _bank_matcher(conn, start),
        "promises": _promises_by_channel(conn, start), "reminders": _reminders(conn, start - timedelta(days=3)),
    }
    return build_report(traces, previous, outcomes, _gates(conn, start, traces),
                        {"from": start.isoformat(), "to": now.isoformat(), "capped": len(traces) >= MAX_TRACES})


def summary_lines(report: dict) -> list[str]:
    """The verdict as short lines for the nightly Telegram message, weakest first."""
    icon = {"red": "🔴", "amber": "🟠", "green": "🟢", "not_enough_data": "⚪"}
    lines = [f"{icon[v['status']]} {v['component']}: {v['why']}" for v in report["verdict"]]
    for g in report["gates"]:
        if not g["passed"]:
            lines.insert(0, f"⛔ {g['name']}: {g['detail']}")
    return lines
