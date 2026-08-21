"""check_policy -- backed by real hybrid retrieval (BM25 + e5 embeddings +
cross-encoder rerank) over whatever's been ingested into the persistent
document store: general policy docs, plus a specific borrower's own
uploaded documents when account_id is given.
"""

from functools import lru_cache

from businessflow.rag.retriever import DocumentRetriever
from businessflow.tools.server import mcp


@lru_cache(maxsize=1)
def _retriever() -> DocumentRetriever:
    """Built once per process -- loading the embedding/rerank models is
    real work. Rebuild the process (or clear this cache) after ingesting
    new documents, since this snapshots the corpus at construction time."""
    return DocumentRetriever()


@mcp.tool
def check_policy(query: str, account_id: str | None = None) -> dict:
    """Look up the written policy relevant to a free-text question, e.g.
    'can I get more time to pay' or 'what does my loan agreement say about
    prepayment'. Pass account_id to also search that borrower's own
    uploaded documents (their loan agreement), not just general policy.
    Ground any statement made to a borrower in the returned text, rather
    than stating a policy number or contract term from memory."""
    results = _retriever().retrieve(query, top_k=2, account_id=account_id)
    return {"query": query, "results": results}
