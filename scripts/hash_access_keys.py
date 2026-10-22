"""Hashes every borrower access key still stored as plain text (accounts/pins.py), and drops the plain text.

Run once after migrations/2026-10-25_hashed_pins.sql and the code that reads access_key_hash are live:

    python -m scripts.hash_access_keys

Re-runnable: an account already hashed is left alone. Prints how many were hashed, never a key.
"""

import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from businessflow.accounts.pins import hash_pin


def main() -> None:
    load_dotenv()
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row, connect_timeout=15) as conn:
        rows = conn.execute("select account_id, access_key from accounts where access_key is not null for update").fetchall()
        for r in rows:
            conn.execute(
                "update accounts set access_key_hash = %s, access_key = null where account_id = %s and access_key = %s",
                (hash_pin(r["access_key"]), r["account_id"], r["access_key"]),
            )
        conn.commit()
        left = conn.execute("select count(*) as n from accounts where access_key is not null").fetchone()["n"]
        missing = conn.execute("select count(*) as n from accounts where access_key_hash is null").fetchone()["n"]
    print(f"hashed {len(rows)} access key(s); {left} still plain text; {missing} account(s) with no key at all")


if __name__ == "__main__":
    main()
