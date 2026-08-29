"""Shared path-safety logic for per-account uploaded documents (signed loan
agreements, KYC, etc.), stored on disk under data/documents/{account_id}/ --
the same directory ops/api.py's upload_account_document() writes into.

Factored out here, rather than duplicated, because TWO different HTTP
surfaces read these same files with the same safety requirement (never
resolve outside the intended account's own subdirectory): the ops-only
staff endpoints (ops/api.py, gated by X-API-Key, any account) and the
borrower-facing, conversation-scoped endpoints (channels/browser_api.py,
gated by a verified session, that borrower's own account only). Living in
accounts/ (not ops/ or channels/) avoids a circular import either way
could otherwise create by reaching into the other's module.
"""

from datetime import datetime, timezone
from pathlib import Path

# Mirrors ops/api.py's _DOCUMENTS_DIR exactly (same four parents-up from
# this file to the repo root: accounts -> businessflow -> src -> repo root).
_DOCUMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "documents"


def list_documents_for_account(account_id: str) -> list[dict]:
    """filename, size_bytes, uploaded_at (the file's real mtime, UTC) for
    every document on disk for this account, oldest upload first. Returns
    an empty list -- not an error -- for an account with no documents dir
    yet at all (the common case: most demo accounts have never had a file
    uploaded), same as an account whose dir exists but is empty."""
    account_dir = _DOCUMENTS_DIR / account_id
    if not account_dir.is_dir():
        return []
    documents = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "uploaded_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        }
        for path in account_dir.iterdir()
        if path.is_file()
    ]
    documents.sort(key=lambda d: d["uploaded_at"])
    return documents


def resolve_document_path(account_id: str, filename: str) -> Path | None:
    """Safely resolves `filename` to a real file under this account's own
    documents directory, or None if it isn't one -- covers a directory-
    traversal attempt (`../BF-9999/agreement.pdf`), an absolute path, a
    filename that simply doesn't exist for this account, and a filename
    that exists only for a DIFFERENT account (all indistinguishable to the
    caller: every rejection looks identical, so a caller can never probe
    whether a file exists under some other account_id).

    Mirrors the basename-only handling ops/api.py's upload_account_document
    already applies on the way IN (Path(...).name strips any directory
    components from client-supplied input) -- applied here again on the way
    OUT, since a filename arriving via a URL path segment is exactly as
    untrusted as one arriving in a multipart upload.
    """
    account_dir = (_DOCUMENTS_DIR / account_id).resolve()
    candidate = (account_dir / Path(filename).name).resolve()
    if account_dir not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate
