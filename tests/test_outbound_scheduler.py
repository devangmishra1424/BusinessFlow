"""Unit test for the scheduler's pure timing logic -- no real sleeping,
no real outbound pass. The loop itself (scripts/run_outbound_scheduler.py's
main()) is a real, always-running process by design and isn't exercised
here, the same reason channels/telegram_bot.py's run_polling() isn't
either.
"""

from datetime import datetime, timezone

from scripts.run_outbound_scheduler import _RUN_HOUR_UTC, _next_run_time


def test_next_run_time_before_the_run_hour_is_later_today():
    now = datetime(2026, 8, 24, 3, 0, tzinfo=timezone.utc)  # before 09:00 UTC

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 24, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)


def test_next_run_time_after_the_run_hour_is_tomorrow():
    now = datetime(2026, 8, 24, 15, 0, tzinfo=timezone.utc)  # after 09:00 UTC

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 25, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)


def test_next_run_time_exactly_at_the_run_hour_is_tomorrow():
    # <=, not < -- once the run hour has actually arrived, the next
    # scheduled run is tomorrow's, not "right now" again.
    now = datetime(2026, 8, 24, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)

    next_run = _next_run_time(now)

    assert next_run == datetime(2026, 8, 25, _RUN_HOUR_UTC, 0, tzinfo=timezone.utc)
