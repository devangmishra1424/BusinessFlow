"""Detects an unauthenticated message shaped like "<account_id> <6-digit
access key>" (e.g. "BF-1001 482913") -- shared by every borrower-facing
channel (telegram_bot.py, browser_api.py) that lets someone verify by
typing their credentials directly into the chat rather than through a
separate login form. Kept in its own tiny module rather than defined in
telegram_bot.py (where this started) so browser_api.py can reuse it
without pulling in python-telegram-bot/soundfile/torch -- none of which
a plain-text HTTP chat channel needs.
"""

import re

CREDENTIALS_PATTERN = re.compile(r"^(\S+)\s+(\d{6})$")


def looks_like_credentials(text: str) -> bool:
    return bool(CREDENTIALS_PATTERN.match(text.strip()))


def parse_credentials(text: str) -> tuple[str, str]:
    """Only valid to call after looks_like_credentials(text) is True."""
    match = CREDENTIALS_PATTERN.match(text.strip())
    return match.group(1), match.group(2)
