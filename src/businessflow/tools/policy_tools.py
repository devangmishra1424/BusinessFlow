"""check_policy -- backed by real hybrid retrieval (BM25 + e5 embeddings +
cross-encoder rerank) over whatever's been ingested into the persistent
document store: general policy docs, plus a specific borrower's own
uploaded documents when account_id is given.
"""

import time

from businessflow.rag.retriever import DocumentRetriever
from businessflow.tools.server import mcp

# How long a cached DocumentRetriever is trusted before this process
# rebuilds it from the persistent store. A document ops/api.py ingests
# (upload_account_document) is in the persistent store the instant that
# call returns, but a long-running borrower-facing process here
# (channels/browser_api.py, channels/telegram_bot.py) only snapshots the
# corpus when DocumentRetriever is constructed -- so without *some* refresh
# it would never see that document until this process happened to restart.
# Real cross-process invalidation (this process being told, via pub/sub or
# similar, the instant a new document lands) would actually eliminate the
# staleness window, but that's real new infra with no existing message-bus
# to hook into here, and isn't justified yet for how rarely documents
# actually get uploaded. Polling wall-clock time on every call is a much
# cheaper, real improvement over the previous behavior (@lru_cache(maxsize=1)
# forever, i.e. staleness bounded only by "until someone restarts the
# process"): it bounds staleness to at most this many seconds, at the cost
# of an occasional rebuild (a few seconds, reloading the embedding/rerank
# models) -- acceptable given how infrequently documents actually get
# uploaded relative to how often check_policy gets called. Tune by changing
# this constant.
_REFRESH_INTERVAL_SECONDS = 10 * 60

_cached_retriever: DocumentRetriever | None = None
_last_built_at: float = 0.0


def _retriever() -> DocumentRetriever:
    """Returns the cached DocumentRetriever, rebuilding it (real work --
    reloading the embedding/rerank models) only if more than
    _REFRESH_INTERVAL_SECONDS have passed since it was last built;
    otherwise reuses the cached instance."""
    global _cached_retriever, _last_built_at
    now = time.time()
    if _cached_retriever is None or (now - _last_built_at) > _REFRESH_INTERVAL_SECONDS:
        _cached_retriever = DocumentRetriever()
        _last_built_at = now
    return _cached_retriever


@mcp.tool(
    description=(
        "Look up the written policy relevant to a free-text question, e.g. 'can I get "
        "more time to pay' or 'what does my loan agreement say about prepayment'. Pass "
        "account_id to also search that borrower's own uploaded documents (their loan "
        "agreement), not just general policy. Ground any statement made to a borrower in "
        "the returned text, rather than stating a policy number or contract term from "
        "memory."
    )
)
def check_policy(query: str, account_id: str | None = None) -> dict:
    """Look up the written policy relevant to a free-text question, e.g.
    'can I get more time to pay' or 'what does my loan agreement say about
    prepayment'. Pass account_id to also search that borrower's own
    uploaded documents (their loan agreement), not just general policy.
    Ground any statement made to a borrower in the returned text, rather
    than stating a policy number or contract term from memory."""
    results = _retriever().retrieve(query, top_k=2, account_id=account_id)
    return {"query": query, "results": results}
