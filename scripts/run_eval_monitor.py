"""Real, standalone scheduler for nightly eval regression monitoring --
same "long-running local process, wakes at a fixed daily hour" pattern as
run_outbound_scheduler.py (see that module's docstring for why a cloud
cron isn't an option for this project yet). Runs a fixed, fast subset of
eval/*.py's own real benchmarks against real Groq/Postgres/RAG -- each one
already knows how to compare itself against its own run history and flag
a regression (eval/tool_scoring.py's record_run_history/
print_regression_delta). The only thing genuinely missing before this was
someone -- or something -- actually running them daily and looking at the
answer, instead of only on a human's own manual `python -m eval.foo`.

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

Runs at 21:00 UTC (~2:30am IST) -- deliberately off-peak, so this suite's
own real Groq calls a night don't compete with real borrower conversations
for the same shared per-minute/per-day quota that has genuinely run dry
before (see outbound/run.py's own history with it).

Run: python -m scripts.run_eval_monitor
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from businessflow.outbound.send import send_ops_alert
from eval import latency_benchmark, reasoning_accuracy, red_team, tool_calling_benchmark, voice_naturalness_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUN_HOUR_UTC = 21
_POLL_INTERVAL_SECONDS = 30 * 60  # wake up at least this often to check

_SUITES = [
    ("tool_calling_benchmark", tool_calling_benchmark.main),
    ("reasoning_accuracy", reasoning_accuracy.main),
    ("red_team", red_team.main),
    ("latency_benchmark", latency_benchmark.main),
    ("voice_naturalness_benchmark", voice_naturalness_benchmark.main),
]


def _next_run_time(now: datetime) -> datetime:
    candidate = now.replace(hour=_RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def run_all_suites() -> dict[str, list[str]]:
    """Runs every suite in _SUITES against real Groq/Postgres/RAG, returns
    {suite_name: [regressed metric names]} for suites that actually
    regressed -- a suite with nothing to report isn't included at all, so
    an empty dict back means a clean night across the board.

    A suite that raises outright is logged loudly and skipped, not
    allowed to crash the rest -- one suite's transient failure (a bad API
    response, a momentary DB blip) must not silence every other suite's
    own real signal for the night. Investigate a skipped suite by running
    it directly: `python -m eval.<name>`."""
    regressions_by_suite = {}
    for name, run in _SUITES:
        try:
            summary = run()
        except Exception:
            logger.exception("eval suite %r failed to run -- skipping it for tonight; investigate with `python -m eval.%s`", name, name)
            continue
        regressed = summary.get("regressed_metrics") or []
        if regressed:
            regressions_by_suite[name] = regressed
    return regressions_by_suite


def main() -> None:
    logger.info("eval monitor started -- nightly regression check fires at %02d:00 UTC", _RUN_HOUR_UTC)
    while True:
        now = datetime.now(timezone.utc)
        next_run = _next_run_time(now)
        time.sleep(max(min((next_run - now).total_seconds(), _POLL_INTERVAL_SECONDS), 1))

        if datetime.now(timezone.utc) < next_run:
            continue  # woke up early for a routine poll, not the actual run time yet

        logger.info("running nightly eval regression check")
        try:
            regressions = run_all_suites()
        except Exception:
            # Shouldn't happen -- run_all_suites already catches per-suite --
            # but a long-running process still must not die outright over
            # something unexpected in the loop itself.
            logger.exception("nightly eval regression check failed outright -- will retry at the next scheduled run")
            continue

        if not regressions:
            logger.info("nightly eval check: no regressions")
            continue

        message = "BusinessFlow eval regression detected:\n" + "\n".join(
            f"- {suite}: {', '.join(metrics)}" for suite, metrics in regressions.items()
        ) + "\n\nCheck eval/results/<suite>_history.jsonl or re-run `python -m eval.<suite>` to investigate."
        logger.error(message)
        if not send_ops_alert(message):
            logger.warning("regression alert could not be delivered via Telegram (OPS_ALERT_TELEGRAM_CHAT_ID unset or delivery failed) -- see the ERROR line above instead")


if __name__ == "__main__":
    main()
