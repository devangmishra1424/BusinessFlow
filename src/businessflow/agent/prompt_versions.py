"""A/B testing for the system prompt: lets a second prompt VARIANT run
against a real percentage of live conversations, side by side with the
existing, hand-tuned baseline (agent/client.py's own
_SYSTEM_PROMPT_TEMPLATE) -- so a wording change can be measured against
real conversation outcomes before fully replacing the baseline, instead
of only being eyeballed in a manual test conversation.

Built on one new table, not a new service -- same "a table, not a new
system" approach as everything else in this project's real state:

    create table prompt_versions (
        version_id text primary key,
        description text not null,
        system_prompt_template text not null,
        rollout_percent integer not null default 0 check (rollout_percent between 0 and 100),
        active boolean not null default true,
        created_at timestamptz not null default now()
    );

system_prompt_template must use the exact same {placeholder} names as
agent/client.py's _SYSTEM_PROMPT_TEMPLATE (build_system_prompt.format()s
whichever template it's given) -- a variant is a rewording, not a
different set of inputs.

Routing is deterministic, not a coin-flip per turn: the SAME bucket_key
(an account_id, since that borrower's identity is stable across separate
conversations) always lands in the same arm for as long as the active
variant's rollout_percent doesn't change -- otherwise a single borrower
could be bounced between prompt variants across calls, which would be
actively confusing to debug and isn't how an A/B test is supposed to
behave (a subject stays in one arm). An anonymous, no-account
conversation has no stable identity to bucket by at all, so it's given a
fresh random key instead -- effectively an independent per-conversation
coin flip, and (matching memory/conversation_memory.py's own convention
for anonymous sessions) never logged for comparison, since there's no
account to attribute an outcome to anyway.

Safety default: with no row in prompt_versions (this project's normal,
unmigrated state) or every row inactive/at 0% rollout, EVERY conversation
gets the untouched baseline -- this ships without changing production
behavior for a single real borrower until an operator deliberately
writes a variant row with rollout_percent > 0.

Honest scope note: with a handful of real seeded demo accounts, a
percentage rollout bucketed by account_id just deterministically splits
those few accounts, not a real statistical sample -- meaningful at real
traffic volume, illustrative at this project's actual size.
"""

import hashlib
import logging

import psycopg

from businessflow.accounts.db import get_connection

logger = logging.getLogger(__name__)

BASELINE_VERSION_ID = "baseline"


def get_active_variant() -> dict | None:
    """The single active, non-baseline variant currently being tested, or
    None if there isn't one. Deliberately at most one at a time -- that
    keeps "why did this conversation get this prompt" a one-step
    question, not an N-way branch to reason about.

    This is called from start_conversation -- the hot path every single
    conversation goes through, including every eval/test/local-dev call
    that runs with no DATABASE_URL at all (this project's own local-test
    convention, see tests/conftest.py) and every real deployment BEFORE
    an operator has actually run the prompt_versions migration. Neither
    is a bug -- both just mean "no experiment configured yet" -- so both
    fall back to the baseline (return None) instead of taking the whole
    conversation down over an optional feature. Caught specifically, not
    a bare Exception: RuntimeError is get_connection()'s own explicit
    "DATABASE_URL is not set" signal, and UndefinedTable is Postgres's
    real error for a migration that hasn't been run yet -- anything else
    (a genuine connection failure, a permissions error) is a real bug and
    still propagates."""
    try:
        row = get_connection().execute(
            "select version_id, system_prompt_template, rollout_percent "
            "from prompt_versions where active = true and version_id != %s "
            "order by created_at desc limit 1",
            (BASELINE_VERSION_ID,),
        ).fetchone()
    except RuntimeError:
        logger.debug("prompt_versions: no DATABASE_URL configured -- using the baseline prompt")
        return None
    except psycopg.errors.UndefinedTable:
        logger.debug("prompt_versions: table not migrated yet -- using the baseline prompt")
        return None
    return dict(row) if row else None


def choose_prompt_version(bucket_key: str) -> tuple[str, str | None]:
    """Returns (version_id, system_prompt_template) -- template is None
    when the baseline applies, telling the caller to fall back to its
    own default template rather than a stored one. Uses hashlib rather
    than Python's built-in hash() because the latter is salted per
    process (PYTHONHASHSEED) -- the same bucket_key must route the same
    way across restarts, not just within one running process."""
    variant = get_active_variant()
    if variant is None:
        return BASELINE_VERSION_ID, None

    bucket = int(hashlib.sha256(bucket_key.encode()).hexdigest(), 16) % 100
    if bucket < variant["rollout_percent"]:
        return variant["version_id"], variant["system_prompt_template"]
    return BASELINE_VERSION_ID, None
