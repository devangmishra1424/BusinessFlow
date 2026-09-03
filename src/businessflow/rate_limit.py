"""In-memory, per-process rate limiting: a fixed-window counter keyed by
whatever the caller identifies a client by (an IP address, typically). No
Redis, no new external service -- this project runs one process per
channel (see README's systemd services), so a shared, cross-process store
would be solving a problem this deployment doesn't have. Would need one
(Redis with TTL keys, most likely) if this ever ran as multiple replicas
behind a load balancer.

Deliberately per-endpoint, not global middleware: rate-limiting every
request the same way would throttle a legitimate operator's dashboard
auto-refresh as readily as an attacker's brute force. Each call site
below decides what actually counts as a "hit" -- see ops/api.py's and
browser_api.py's own usage for why only FAILED auth attempts count there,
not every request.
"""

import time
from collections import defaultdict

_CLEANUP_EVERY_N_CALLS = 1000


class RateLimiter:
    """check(key) -> True if this call is allowed (and records it as a
    hit), False if key has already hit max_requests within the trailing
    window_seconds. Never raises itself -- callers decide what "not
    allowed" means (429, log-and-continue, whatever fits).

    Memory is bounded for any single active key (old hits are trimmed off
    the front of its list every call), and a periodic full sweep drops
    keys with no hits left in the window at all, so a long-running
    process doesn't accumulate one empty list per distinct IP it has ever
    seen. This is still an in-memory, best-effort bound appropriate for
    this project's real scale -- a sustained attack from a very large
    number of distinct IPs would still grow memory between sweeps; a real
    high-traffic deployment would want Redis with TTL keys instead."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._calls_since_cleanup = 0

    def check(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds

        hits = self._hits[key]
        while hits and hits[0] < cutoff:
            hits.pop(0)

        allowed = len(hits) < self.max_requests
        if allowed:
            hits.append(now)

        self._calls_since_cleanup += 1
        if self._calls_since_cleanup >= _CLEANUP_EVERY_N_CALLS:
            self._cleanup(cutoff)

        return allowed

    def _cleanup(self, cutoff: float) -> None:
        self._calls_since_cleanup = 0
        stale_keys = [k for k, hits in self._hits.items() if not hits or hits[-1] < cutoff]
        for k in stale_keys:
            del self._hits[k]
