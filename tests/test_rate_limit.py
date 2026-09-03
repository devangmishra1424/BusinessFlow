"""Tests for RateLimiter -- pure logic, no DB/network needed. Uses a real
clock (time.monotonic) rather than mocking it, matching this project's
"real over mocked" convention -- the sleeps here are short and few."""

import time

from businessflow.rate_limit import RateLimiter


def test_allows_up_to_max_requests_within_the_window():
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is True


def test_rejects_the_request_that_exceeds_max_within_the_window():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is False


def test_different_keys_are_tracked_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.check("1.2.3.4") is True
    assert limiter.check("5.6.7.8") is True  # a different key, its own fresh budget
    assert limiter.check("1.2.3.4") is False
    assert limiter.check("5.6.7.8") is False


def test_a_rejected_check_does_not_itself_count_as_a_hit():
    # Otherwise a caller hammering a limited endpoint would never recover
    # even after the window passes, since every rejected attempt would
    # keep re-extending its own window.
    limiter = RateLimiter(max_requests=1, window_seconds=0.2)

    assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is False
    assert limiter.check("1.2.3.4") is False  # still within window -- still rejected

    time.sleep(0.25)
    assert limiter.check("1.2.3.4") is True  # window passed -- allowed again


def test_old_hits_outside_the_window_are_forgotten():
    limiter = RateLimiter(max_requests=1, window_seconds=0.2)

    assert limiter.check("1.2.3.4") is True
    time.sleep(0.25)
    assert limiter.check("1.2.3.4") is True  # the first hit has aged out


def test_cleanup_removes_stale_keys_without_affecting_active_ones():
    limiter = RateLimiter(max_requests=1, window_seconds=0.1)
    limiter.check("stale-key")
    time.sleep(0.15)  # stale-key's one hit ages out before the sweep below

    limiter._calls_since_cleanup = 999  # force the *next* check() to trigger a sweep
    allowed = limiter.check("active-key")  # this call is what triggers _cleanup()

    assert allowed is True
    assert "stale-key" not in limiter._hits
    assert "active-key" in limiter._hits
