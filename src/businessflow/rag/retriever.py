"""Hybrid retrieval over whatever's actually been ingested into the
persistent store: BM25 keyword search + multilingual-e5 embedding search,
combined and reranked by a cross-encoder. Renamed from PolicyRetriever --
this now serves any ingested document (policy, loan agreements,
regulatory circulars), not just the hand-written policy KB it started as.
"""

import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from businessflow.rag.embeddings import embed_query
from businessflow.rag.store import get_collection

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DocumentRetriever:
    """Builds its BM25 index fresh from whatever's currently in the
    persistent store at construction time -- so it picks up anything
    ingested since the last time this was built. Rebuild (create a new
    instance) after ingesting new documents rather than reusing a stale
    one for the lifetime of a long-running process."""

    def __init__(self):
        self._collection = get_collection()
        corpus = self._collection.get(include=["documents", "metadatas"])
        self._ids = corpus["ids"]
        self._texts = corpus["documents"]
        self._metadatas = corpus["metadatas"]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts]) if self._texts else None

        self._reranker = CrossEncoder(
            _RERANK_MODEL, backend="onnx", model_kwargs={"file_name": "onnx/model_quint8_avx2.onnx"}
        )

    def retrieve(self, query: str, top_k: int = 2, candidate_pool: int = 4, account_id: str | None = None) -> list[dict]:
        """Up to top_k chunks most relevant to a free-text query, after
        reranking. account_id, if given, restricts results to general
        documents plus that specific borrower's own documents -- never a
        different borrower's. candidate_pool is how many candidates each
        of BM25 and embedding search contribute before reranking."""
        if not self._texts:
            return []

        allowed_scopes = {"general", account_id} if account_id else {"general"}
        eligible = [i for i, m in enumerate(self._metadatas) if m.get("account_id") in allowed_scopes]
        if not eligible:
            return []

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(eligible, key=lambda i: bm25_scores[i], reverse=True)
        bm25_candidates = set(bm25_ranked[:candidate_pool])

        where = {"account_id": {"$in": list(allowed_scopes)}}
        embed_result = self._collection.query(
            query_embeddings=[embed_query(query)], n_results=candidate_pool, where=where
        )
        embed_candidate_ids = set(embed_result["ids"][0])
        embed_candidates = {self._ids.index(doc_id) for doc_id in embed_candidate_ids}

        candidate_indices = bm25_candidates | embed_candidates
        pairs = [(query, self._texts[i]) for i in candidate_indices]
        rerank_scores = self._reranker.predict(pairs)

        ranked = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        return [
            {
                "text": self._texts[i],
                "relevance_score": float(score),
                **{k: v for k, v in self._metadatas[i].items()},
            }
            for i, score in ranked[:top_k]
        ]
