"""Database access for every store function, backed by a real connection
pool (not one single cached connection) -- a single connection can't
safely serve two overlapping requests at once (e.g. two simultaneous
conversations through the FastAPI channel), which is exactly the
production-grade gap this replaces.

get_connection() still returns one object, cached for the process's
lifetime, and every call site still does get_connection().execute(...)
exactly as before -- no call site elsewhere in the codebase needed to
change. What changed is what that object does internally: each
.execute() call now checks a real connection out of the pool, runs the
query, buffers the results, and returns the connection to the pool
before returning -- rather than holding one connection open forever.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

_MIN_POOL_SIZE = 2
# Capped well under Supabase's own Session Pooler limit for this project
# (a hard 15 concurrent connections total, shared across everything that
# connects -- discovered by actually running a concurrency test at
# max_size=10 and hitting "EMAXCONNSESSION" from Supabase's own pooler,
# not a limit on our side). Callers beyond this queue briefly for a free
# connection rather than erroring -- that's the pool doing its job, not
# a problem.
_MAX_POOL_SIZE = 5


class _BufferedCursor:
    """A read-only stand-in for a psycopg Cursor, holding results that
    were already fetched before the real connection went back to the
    pool -- callers still call .fetchone()/.fetchall() on it exactly like
    a live cursor, they just can't run a second query through it."""

    def __init__(self, rows: list[dict] | None):
        self._rows = rows or []
        self._index = 0

    def fetchone(self) -> dict | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> list[dict]:
        rows = self._rows[self._index:]
        self._index = len(self._rows)
        return rows


class _PooledExecutor:
    """Drop-in replacement for the single cached psycopg.Connection this
    project used to hand out directly -- same .execute(...) call
    convention, but backed by a real pool underneath."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def execute(self, query: str, params: tuple | None = None) -> _BufferedCursor:
        with self._pool.connection() as conn:
            cursor = conn.execute(query, params)
            # cursor.description is None for statements with no result
            # set (a plain INSERT/UPDATE with no RETURNING) -- nothing to
            # buffer, and calling fetchall() on those raises.
            rows = cursor.fetchall() if cursor.description is not None else None
            return _BufferedCursor(rows)


@lru_cache(maxsize=1)
def get_connection() -> _PooledExecutor:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")
    pool = ConnectionPool(
        database_url,
        min_size=_MIN_POOL_SIZE,
        max_size=_MAX_POOL_SIZE,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=True,
    )
    # open=True returns before min_size connections are necessarily ready
    # (they open in background worker threads) -- wait() blocks until the
    # pool is actually usable, so the very first query doesn't race a
    # still-initializing connection.
    pool.wait()
    return _PooledExecutor(pool)
