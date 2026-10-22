"""Borrower access keys (6-digit PINs), stored hashed.

A PIN used to sit in accounts.access_key as plain text, so anyone who could read the table could sign in as any
borrower. Now only a salted scrypt hash is stored (accounts.access_key_hash, "scrypt$<salt hex>$<hash hex>"). A
6-digit PIN has only a million values, so a hash does not make a stolen table safe forever -- scrypt makes trying
them all expensive, and the per-account limit on failed sign-ins (agent/loop.py) is what stops guessing online.

Pure: no I/O.
"""

import hashlib
import hmac
import secrets

_N, _R, _P = 2**14, 8, 1  # ~16 MB and tens of milliseconds per check: cheap for one sign-in, slow for a million guesses
_PREFIX = "scrypt"


def new_pin() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_pin(pin: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"{_PREFIX}${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, stored: str | None) -> bool:
    """True if `pin` is the one `stored` was made from. A missing or malformed hash never verifies."""
    if not stored or not pin:
        return False
    try:
        prefix, salt_hex, digest_hex = stored.split("$")
        if prefix != _PREFIX:
            return False
        digest = hashlib.scrypt(pin.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=_N, r=_R, p=_P, dklen=32)
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)
