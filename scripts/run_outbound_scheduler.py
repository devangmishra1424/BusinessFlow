"""Real, standalone scheduler for the proactive outbound pass -- runs
continuously, firing run_daily_outbound_pass() once per day at a fixed
UTC hour. Same "long-running local process" pattern
channels/telegram_bot.py already uses (this project has no real hosting
yet, so a cloud cron isn't an option -- see outbound/run.py's own
comment on that history). Point a real OS/cloud cron at
scripts/run_outbound_pass.py's main() instead once real hosting exists;
run_daily_outbound_pass() itself is unchanged either way, and is already
idempotent against firing more than once for the same account+kind in a
day, so an extra wakeup near the boundary is harmless.

Run: python -m scripts.run_outbound_scheduler
"""

import logging
import time
from datetime import datetime, timedelta, timezone

from businessflow.outbound.run import run_daily_outbound_pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RUN_HOUR_UTC = 9  # late morning in India (IST = UTC+5:30)
_POLL_INTERVAL_SECONDS = 30 * 60  # wake up at least this often to check


def _next_run_time(now: datetime) -> datetime:
    candidate = now.replace(hour=_RUN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def main() -> None:
    logger.info("outbound scheduler started -- daily pass fires at %02d:00 UTC", _RUN_HOUR_UTC)
    while True:
        now = datetime.now(timezone.utc)
        next_run = _next_run_time(now)
        time.sleep(max(min((next_run - now).total_seconds(), _POLL_INTERVAL_SECONDS), 1))

        if datetime.now(timezone.utc) < next_run:
            continue  # woke up early for a routine poll, not the actual run time yet

        logger.info("running daily outbound pass")
        try:
            sent = run_daily_outbound_pass()
            logger.info("daily outbound pass sent %d reminder(s)", len(sent))
        except Exception:
            # A long-running process: one bad day (a transient DB blip,
            # say) must not silence every future day's reminders. Logged
            # loudly (exc_info via .exception) rather than swallowed --
            # the loop just retries at the next scheduled run.
            logger.exception("daily outbound pass failed -- will retry at the next scheduled run")


if __name__ == "__main__":
    main()
