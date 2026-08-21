"""The one Postgres connection every store function goes through. Cached
for the process's lifetime -- opening a fresh connection per tool call
would add a real, unnecessary round-trip on every single lookup.
"""

import os
from functools import lru_cache

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()


@lru_cache(maxsize=1)
def get_connection() -> psycopg.Connection:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in")
    return psycopg.connect(database_url, row_factory=dict_row, autocommit=True)
