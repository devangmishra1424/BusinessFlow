"""Benchmarks and evals.

Every suite reseeds the demo accounts (scripts/seed_accounts.py) and logs tool calls as it runs. Run against the live
database -- as the nightly check did, at 21:00 UTC -- that wiped whatever people had done on the demo that day (found
live: a borrower's payment gone the next morning). So against the production database the evals always use their own
copy of the tables, schema "eval" (accounts/db.py's DATABASE_SCHEMA; build it with python -m scripts.setup_eval_schema).

Imported before any suite module, so this runs before anything opens a database connection. The live app never imports
this package. Tests are unaffected: tests/conftest.py blanks a production DATABASE_URL before this is imported, and CI's
database is its own disposable one.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # DATABASE_URL must be loaded before it is checked -- the same reason as tests/conftest.py
if "supabase" in os.environ.get("DATABASE_URL", "").lower() and not os.environ.get("DATABASE_SCHEMA"):
    os.environ["DATABASE_SCHEMA"] = "eval"
