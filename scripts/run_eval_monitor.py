"""One-shot nightly eval regression check -- triggered by
businessflow-eval-monitor.timer (systemd, OnCalendar=*-*-* 21:00:00 UTC),
NOT an always-on resident process. Runs a fixed, fast subset of eval/*.py's
own real benchmarks against real Groq/Postgres/RAG -- each one already
knows how to compare itself against its own run history and flag a
regression (eval/tool_scoring.py's record_run_history/
print_regression_delta). The only thing genuinely missing before this was
someone -- or something -- actually running them daily and looking at the
answer, instead of only on a human's own manual `python -m eval.foo`.

Deliberately NOT a resident sleep-loop like run_outbound_scheduler.py:
this needs the full torch/transformers/TTS/ASR/RAG stack loaded (hundreds
of MB of import weight before any real inference even starts) to do
anything at all, and it only actually needs to DO anything once a night.
Found live: keeping that loaded 24/7 as a 5th always-on service, on a VM
with only 3.8GB total RAM, was a real, direct contributor to the kernel
OOM-killing a DIFFERENT, borrower-facing service (businessflow-bot) mid-
conversation -- see the timer/service unit files this module's own
deploy notes reference. A one-shot process that starts, runs, and exits
holds that memory for a few minutes a night instead of all day.

wer_benchmark.py and retrieval_benchmark.py are deliberately excluded:
wer_benchmark needs the MUCS dataset checked out locally (not present on
the VM) and takes hours even sampled; retrieval_benchmark doesn't use the
record_run_history/regression convention the other four already do. Both
stay manual, run-when-you-mean-to checks.

voice_naturalness_benchmark.py is included, but only its deterministic
correctness checks (verbalizer coverage, duration sanity) can actually
trigger an alert here -- its SQUIM naturalness SCORES are real but
genuinely noisy run to run (Piper/MMS-TTS are both stochastic), so that
module deliberately keeps those out of its own regressed_metrics to avoid
crying wolf nightly; see that module's own docstring. This suite needs no
Groq/Postgres at all (TTS + a local SQUIM model only), so it still runs
even on a night the other four can't reach Groq.

Deploy (VM, one-time setup):
  businessflow-eval-monitor.service -- Type=oneshot, no [Install]/Restart=
    (a oneshot unit that "fails" every night would be noise; a genuine
    failure inside is already caught per-suite by run_all_suites and
    logged/alerted, not left to systemd's own failure handling).
  businessflow-eval-monitor.timer -- OnCalendar=*-*-* 21:00:00 UTC,
    Persistent=true (catches up on the next boot if the VM was down
    exactly at 21:00), WantedBy=timers.target.
  `sudo systemctl daemon-reload && sudo systemctl enable --now
  businessflow-eval-monitor.timer` -- do NOT enable the .service itself
  (the timer activates it; enabling both double-registers the same unit
  with multi-user.target for no reason).

Its own copy of the data: every suite reseeds the demo accounts, so this
runs on the "eval" schema (eval/__init__.py sets DATABASE_SCHEMA; build or
refresh the copy with python -m scripts.setup_eval_schema) and refuses to
start on the live tables. Found live: run against them at 21:00 UTC, it
wiped a payment a person had made on the demo that day, along with every
chat and request on all ten demo accounts.

Run manually: python -m scripts.run_eval_monitor
"""

import logging

from businessflow.accounts.db import database_schema, get_connection
from businessflow.outbound.send import send_ops_alert
from eval import latency_benchmark, reasoning_accuracy, red_team, tool_calling_benchmark, voice_naturalness_benchmark
from eval.report import RESULTS_DIR, SuiteOutcome, build_report, for_telegram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SUITES = [
    ("tool_calling_benchmark", tool_calling_benchmark.main),
    ("reasoning_accuracy", reasoning_accuracy.main),
    ("red_team", red_team.main),
    ("latency_benchmark", latency_benchmark.main),
    ("voice_naturalness_benchmark", voice_naturalness_benchmark.main),
]


def run_all_suites() -> list[SuiteOutcome]:
    """Runs every suite in _SUITES against real Groq/Postgres/RAG and returns
    one outcome per suite: its summary (with regressed_metrics, the metrics
    worse than the previous run), or the error if it couldn't run.

    A suite that raises outright is logged loudly and recorded, not
    allowed to crash the rest -- one suite's transient failure (a bad API
    response, a momentary DB blip) must not silence every other suite's
    own real signal for the night. Broad on purpose: any failure in a suite
    is that suite's result for the night, and it goes into the report
    (it used to be logged only, so a suite that never ran told no one).
    Investigate with `python -m eval.<name>`."""
    outcomes = []
    for name, run in _SUITES:
        try:
            outcomes.append(SuiteOutcome(name, summary=run()))
        except Exception as e:
            logger.exception("eval suite %r failed to run -- reported as 'couldn't run'; investigate with `python -m eval.%s`", name, name)
            outcomes.append(SuiteOutcome(name, error=f"{type(e).__name__}: {e}"))
    return outcomes


def main() -> None:
    """Runs every suite exactly once, alerts if anything regressed, then
    returns -- systemd's timer owns WHEN this runs, not this function.
    A genuine, unexpected failure here (not an individual suite's own
    failure, already caught inside run_all_suites) propagates and this
    process exits non-zero -- visible in `systemctl status`/journalctl
    for that run, not silently swallowed into "will retry tomorrow" the
    way the old resident loop's own outer try/except used to."""
    schema = database_schema()
    if schema is None:
        message = (
            "BusinessFlow nightly evals did NOT run: DATABASE_SCHEMA isn't set to their own copy of the tables, and "
            "every suite reseeds the demo accounts -- on the live tables that wipes what people did on the demo today."
        )
        logger.error(message)
        send_ops_alert(message)
        raise SystemExit(1)
    # Opening the pool checks the copy is complete (accounts/db.py's use_schema) and raises if it isn't, before any
    # suite runs -- a suite's own failure would only be logged and skipped.
    get_connection().execute("select 1")
    logger.info("running nightly eval regression check on schema %r", schema)
    outcomes = run_all_suites()

    # The whole report every night, on the server and in the log; Telegram only when something needs a look.
    report = build_report(outcomes, results_dir=RESULTS_DIR)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "nightly_report.txt").write_text(report + "\n", encoding="utf-8")
    needs_a_look = any(o.regressed or o.error for o in outcomes)
    if not needs_a_look:
        logger.info("nightly eval check: all suites held steady\n%s", report)
        return

    logger.error("nightly eval check needs a look:\n%s", report)
    if not send_ops_alert(for_telegram(report)):
        logger.warning("nightly eval report could not be delivered via Telegram (OPS_ALERT_TELEGRAM_CHAT_ID unset or delivery failed) -- see the ERROR line above instead")


if __name__ == "__main__":
    main()
