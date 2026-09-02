"""Detects an unauthenticated message shaped like "<account_id> <6-digit
access key>" (e.g. "BF-1001 482913") -- shared by every borrower-facing
channel (telegram_bot.py, browser_api.py) that lets someone verify by
typing their credentials directly into the chat rather than through a
separate login form. Kept in its own tiny module rather than defined in
telegram_bot.py (where this started) so browser_api.py can reuse it
without pulling in python-telegram-bot/soundfile/torch -- none of which
a plain-text HTTP chat channel needs.

Also holds the Telegram deep-link payload encoding (build/parse_telegram_
start_payload) -- the same "no heavy channel deps" reasoning applies:
ops/api.py needs to BUILD a payload to hand back on account creation, but
must not import telegram_bot.py itself just for that (which would pull in
python-telegram-bot's whole Application machinery for a plain string
operation).
"""

import re

CREDENTIALS_PATTERN = re.compile(r"^(\S+)\s+(\d{6})$")


def looks_like_credentials(text: str) -> bool:
    return bool(CREDENTIALS_PATTERN.match(text.strip()))


def parse_credentials(text: str) -> tuple[str, str]:
    """Only valid to call after looks_like_credentials(text) is True."""
    match = CREDENTIALS_PATTERN.match(text.strip())
    return match.group(1), match.group(2)


# Telegram's own /start deep-link payload charset is limited to
# [A-Za-z0-9_-], max 64 chars -- account_id already contains a hyphen
# ("BF-1001"), so an underscore is used as the account_id/access_key
# separator to keep splitting unambiguous rather than colliding with it.
_TELEGRAM_START_PAYLOAD_PATTERN = re.compile(r"^([A-Za-z0-9-]+)_(\d{6})$")


def build_telegram_start_payload(account_id: str, access_key: str) -> str:
    return f"{account_id}_{access_key}"


def parse_telegram_start_payload(payload: str) -> tuple[str, str] | None:
    """None for anything not shaped like a real payload this module itself
    produced -- callers must not guess/proceed on a malformed or foreign
    /start argument, just fall back to the normal "type your credentials"
    flow."""
    match = _TELEGRAM_START_PAYLOAD_PATTERN.match(payload.strip())
    if match is None:
        return None
    return match.group(1), match.group(2)
