"""The nightly eval report, in plain words.

The nightly check (scripts/run_eval_monitor.py) used to send only "eval regression detected: red_team: pass_rate" --
no numbers, no idea what the check is for, which scenario failed, or whether it was one unlucky run. This builds the
report from what each suite already saves: its summary, its per-scenario results (eval/results/<suite>.json) and its
run history (eval/results/<suite>_history.jsonl). For every suite it says:

  - the result in words ("7 of 8 attacks handled"), against last night and the usual (median of recent runs);
  - what the check is for, in one line;
  - what failed, scenario by scenario, with the reason the suite recorded;
  - if the suite could not run at all, the error.

Pure functions plus file reads -- no model calls, so it is cheap to rebuild and easy to test.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IST = timezone(timedelta(hours=5, minutes=30), "IST")
USUAL_WINDOW = 7  # "usual" = the median of up to this many runs before this one
TELEGRAM_LIMIT = 3900  # Telegram's hard limit is 4096 characters; the full report is also saved to a file
_MAX_FAILURES_LISTED = 4
_MAX_REASON_CHARS = 160

# What each metric a suite can flag means, in plain words.
METRIC_WORDS = {
    "precision": "avoiding actions it shouldn't take",
    "recall": "taking the actions it should",
    "scenario_success_rate": "scenarios fully right",
    "reasoning_accuracy_rate": "replies stating the facts correctly",
    "pass_rate": "attacks handled safely",
    "p50_seconds": "typical reply time",
    "p90_seconds": "slow-end reply time",
    "mean_seconds": "average reply time",
    "verbalizer_failures": "replies that would read symbols aloud",
    "duration_implausible_count": "speech of an implausible length",
}


@dataclass
class SuiteOutcome:
    """One suite's run: its summary when it ran, or the error when it couldn't."""

    name: str
    summary: dict | None = None
    error: str | None = None

    @property
    def regressed(self) -> list[str]:
        return list((self.summary or {}).get("regressed_metrics") or [])


@dataclass
class _Section:
    status: str  # "worse" | "ok" | "failed"
    text: str
    details: list[str] = field(default_factory=list)


def _clip(text: str, limit: int = _MAX_REASON_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def read_results(name: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    path = results_dir / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("results") or []


def read_history(name: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Every recorded run's summary, oldest first (the run just finished is the last one)."""
    path = results_dir / f"{name}_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["summary"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _earlier(history: list[dict]) -> tuple[dict | None, list[dict]]:
    """(last night's summary, up to USUAL_WINDOW runs before this one) -- this run is history's last line."""
    before = history[:-1]
    return (before[-1] if before else None), before[-USUAL_WINDOW:]


def _usual(runs: list[dict], value: Callable[[dict], float | None]) -> float | None:
    values = [v for v in (value(r) for r in runs) if v is not None]
    return statistics.median(values) if values else None


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{round(x * 100)}%"


def _secs(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f} s"


def _versus(last: str | None, usual: str | None) -> str:
    parts = [f"last night {last}" if last else None, f"usual {usual}" if usual else None]
    parts = [p for p in parts if p]
    return f" ({' · '.join(parts)})" if parts else " (first run, nothing to compare with yet)"


def _rate(summary: dict, numerator: str, denominator: str, rate: str) -> tuple[int, int, float | None]:
    return summary.get(numerator, 0), summary.get(denominator, 0), summary.get(rate)


# ------------------------------------------------------------------------------------------------- per suite


def _tool_calling(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    last, runs = _earlier(history)
    passed, total, rate = _rate(s, "scenarios_passed", "scenario_count", "scenario_success_rate")
    headline = f"{passed} of {total} scenarios fully right ({_pct(rate)})" + _versus(
        _pct(last.get("scenario_success_rate")) if last else None,
        _pct(_usual(runs, lambda r: r.get("scenario_success_rate"))) if runs else None,
    )
    details = []
    for r in [r for r in results if not r.get("passed")][:_MAX_FAILURES_LISTED]:
        why = []
        if r.get("missed_required"):
            why.append(f"didn't call {', '.join(map(str, r['missed_required']))}")
        if r.get("forbidden_violations"):
            why.append(f"called {', '.join(map(str, r['forbidden_violations']))} when it shouldn't")
        details.append(f"{r.get('scenario_id')}: {'; '.join(why) or 'failed'}")
    return headline, details


def _reasoning(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    last, runs = _earlier(history)
    accurate, judged, rate = _rate(s, "accurate_count", "judged_count", "reasoning_accuracy_rate")
    headline = f"{accurate} of {judged} replies stated the facts correctly ({_pct(rate)})" + _versus(
        _pct(last.get("reasoning_accuracy_rate")) if last else None,
        _pct(_usual(runs, lambda r: r.get("reasoning_accuracy_rate"))) if runs else None,
    )
    details = [
        f"{r.get('scenario_id')}: {_clip(r.get('reason'))}"
        for r in [r for r in results if r.get("accurate") is False][:_MAX_FAILURES_LISTED]
    ]
    if s.get("judge_call_failures"):
        details.append(f"{s['judge_call_failures']} reply(s) not judged: the judging call itself failed (left out of the rate)")
    return headline, details


def _red_team(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    last, runs = _earlier(history)
    passed, total, rate = _rate(s, "passed_count", "scenario_count", "pass_rate")
    headline = f"{passed} of {total} attacks handled safely ({_pct(rate)})" + _versus(
        _pct(last.get("pass_rate")) if last else None,
        _pct(_usual(runs, lambda r: r.get("pass_rate"))) if runs else None,
    )
    details = [
        f"{r.get('scenario_id')} ({r.get('category')}): {_clip(r.get('reason'))}"
        for r in [r for r in results if not r.get("passed")][:_MAX_FAILURES_LISTED]
    ]
    return headline, details


def _latency(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    last, runs = _earlier(history)
    clean = s.get("clean") or {}

    def _p(r: dict | None, key: str) -> float | None:
        return ((r or {}).get("clean") or {}).get(key)

    headline = f"typical reply {_secs(clean.get('p50_seconds'))}, slow end {_secs(clean.get('p90_seconds'))}" + _versus(
        f"{_secs(_p(last, 'p50_seconds'))} / {_secs(_p(last, 'p90_seconds'))}" if last else None,
        f"{_secs(_usual(runs, lambda r: _p(r, 'p50_seconds')))} / {_secs(_usual(runs, lambda r: _p(r, 'p90_seconds')))}" if runs else None,
    )
    details = []
    # Each scenario runs more than once; name each one once, by its slowest run.
    worst: dict[str, float] = {}
    for r in results:
        worst[r.get("scenario_id")] = max(worst.get(r.get("scenario_id"), 0.0), r.get("elapsed_seconds") or 0.0)
    slowest = [{"scenario_id": k, "elapsed_seconds": v} for k, v in sorted(worst.items(), key=lambda kv: kv[1], reverse=True)[:2]]
    if slowest:
        details.append("slowest: " + ", ".join(f"{r.get('scenario_id')} {_secs(r.get('elapsed_seconds'))}" for r in slowest))
    if s.get("degraded_run_count"):
        details.append(
            f"{s['degraded_run_count']} of {s.get('run_count')} replies hit a Groq rate limit mid-reply "
            "(left out of these numbers; a sign the shared daily quota is running low)"
        )
    return headline, details


def _voice(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    total = s.get("scenario_count", 0)
    symbols, lengths = s.get("verbalizer_failures", 0), s.get("duration_implausible_count", 0)
    if not symbols and not lengths:
        return f"all {total} replies spoken cleanly, at a plausible length", []
    headline = f"{symbols} of {total} would read symbols aloud, {lengths} of {total} had an implausible length"
    details = [
        f"{r.get('scenario_id')} ({r.get('language')}): "
        + ", ".join(x for x in (
            "symbols left in" if r.get("verbalizer_left_a_stray_pattern") else None,
            "implausible length" if r.get("duration_plausible") is False else None,
        ) if x)
        for r in results
        if r.get("verbalizer_left_a_stray_pattern") or r.get("duration_plausible") is False
    ][:_MAX_FAILURES_LISTED]
    return headline, details


def _lab(s: dict, results: list[dict], history: list[dict]) -> tuple[str, list[str]]:
    headline = f"{s.get('passed', 0)} of {s.get('cases', 0)} labelled conversations right"
    details = [
        f"{r.get('case_id')}: " + next((p for t in r.get("turns", []) for p in t.get("problems", [])), "failed")
        for r in results if not r.get("passed")
    ][:_MAX_FAILURES_LISTED]
    return headline, details


@dataclass(frozen=True)
class _SuiteInfo:
    title: str
    purpose: str
    describe: Callable[[dict, list[dict], list[dict]], tuple[str, list[str]]]


SUITES = {
    "tool_calling_benchmark": _SuiteInfo(
        "Actions", "does the agent take the right action (payment link, promise, dispute) and avoid wrong ones?", _tool_calling,
    ),
    "reasoning_accuracy": _SuiteInfo(
        "Facts", "a second AI checks each reply's figures and claims against the real account data", _reasoning,
    ),
    "red_team": _SuiteInfo(
        "Safety", "deliberate attacks: prompt injection, other customers' data, pressure to waive fees", _red_team,
    ),
    "latency_benchmark": _SuiteInfo(
        "Speed", "time from a borrower's message to the full reply, tool calls included", _latency,
    ),
    "conversation_lab": _SuiteInfo(
        "Lab", "labelled conversations replayed through the real tools and safety checks (eval/conversation_lab.py)", _lab,
    ),
    "voice_naturalness_benchmark": _SuiteInfo(
        "Voice", "spoken replies never read symbols or links aloud, and come out at a plausible length", _voice,
    ),
}


def _section(outcome: SuiteOutcome, results_dir: Path) -> _Section:
    info = SUITES.get(outcome.name)
    title = info.title if info else outcome.name
    if outcome.error is not None:
        return _Section("failed", f"⛔ {title}: couldn't run", [_clip(outcome.error, 220), f"re-run: python -m eval.{outcome.name}"])
    summary = outcome.summary or {}
    if info is None:
        headline, details = "ran", []
    else:
        headline, details = info.describe(summary, read_results(outcome.name, results_dir), read_history(outcome.name, results_dir))
    if outcome.regressed:
        worse = ", ".join(METRIC_WORDS.get(m, m) for m in outcome.regressed)
        lines = [f"Worse at: {worse}."]
        if info:
            lines.insert(0, f"Checks: {info.purpose}")
        return _Section("worse", f"⚠️ {title}: {headline}", lines + details)
    return _Section("ok", f"✅ {title}: {headline}")


def build_report(outcomes: list[SuiteOutcome], when: datetime | None = None, results_dir: Path = RESULTS_DIR) -> str:
    """The whole report: what needs a look first (worse, or couldn't run), then everything that held steady."""
    when = (when or datetime.now(timezone.utc)).astimezone(IST)
    sections = [_section(o, results_dir) for o in outcomes]
    worse = [s for s in sections if s.status == "worse"]
    failed = [s for s in sections if s.status == "failed"]
    ok = [s for s in sections if s.status == "ok"]

    if worse or failed:
        parts = []
        if worse:
            parts.append(f"{len(worse)} of {len(sections)} checks got worse than last night")
        if failed:
            parts.append(f"{len(failed)} couldn't run")
        verdict = "Needs a look: " + ", ".join(parts) + "."
    else:
        verdict = f"All {len(sections)} checks held steady."

    lines = [f"BusinessFlow nightly check · {when.strftime('%d %b %Y, %H:%M')} IST", verdict, ""]
    for s in failed + worse:
        lines.append(s.text)
        lines.extend(f"   • {d}" for d in s.details)
        lines.append("")
    lines.extend(s.text for s in ok)
    lines += [
        "",
        "One night's dip can be noise (the AI's answers vary run to run); the same dip two nights running is worth a look.",
        "Full details on the server: eval/results/<suite>.json, or re-run python -m eval.<suite>.",
    ]
    return "\n".join(lines).strip()


def for_telegram(report: str) -> str:
    """The report, cut to fit one Telegram message; the full one is in eval/results/nightly_report.txt."""
    if len(report) <= TELEGRAM_LIMIT:
        return report
    note = "\n… (cut to fit Telegram; the full report is in eval/results/nightly_report.txt on the server)"
    return report[: TELEGRAM_LIMIT - len(note)].rstrip() + note
