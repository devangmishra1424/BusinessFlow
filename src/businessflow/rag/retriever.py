"""Hybrid retrieval over whatever's actually been ingested into the
persistent store: BM25 keyword search + multilingual-e5 embedding search,
combined and reranked by a cross-encoder. Renamed from PolicyRetriever --
this now serves any ingested document (policy, loan agreements,
regulatory circulars), not just the hand-written policy KB it started as.
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from businessflow.rag import query_llm
from businessflow.rag.embeddings import embed_query
from businessflow.rag.store import get_collection
from businessflow.rag.tokenize import tokenize as _tokenize

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # English-only -- see _embedding_only_results below


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

    def retrieve(
        self, query: str, top_k: int = 2, candidate_pool: int = 4,
        account_id: str | None = None, expand: bool = False,
    ) -> list[dict]:
        """Up to top_k chunks most relevant to a free-text query, after
        reranking. account_id, if given, restricts results to general
        documents plus that specific borrower's own documents -- never a
        different borrower's. candidate_pool is how many candidates each
        of BM25 and embedding search contribute before reranking, per
        query variant. expand=True additionally generates a couple of
        alternate phrasings of the query (query_llm.expand_query) and
        pools candidates across all of them -- a real extra Groq call, so
        it's opt-in rather than always-on; see eval/retrieval_benchmark.py
        for the measured effect before deciding whether to default it on
        for a given caller."""
        if not self._texts:
            return []

        allowed_scopes = {"general", account_id} if account_id else {"general"}
        eligible = [i for i, m in enumerate(self._metadatas) if m.get("account_id") in allowed_scopes]
        if not eligible:
            return []

        if query_llm.needs_translation(query):
            query = query_llm.translate_to_english(query)

        if not _tokenize(query):
            # Translation didn't run (query_llm.needs_translation said no)
            # or it ran and failed (Groq down, falls back to the original
            # query unchanged) -- either way, still no Latin-alphanumeric
            # content. Two real, separately-verified problems rule out the
            # normal hybrid+rerank path here:
            #  1. BM25Okapi.get_scores([]) ties every chunk at 0.0, and
            #     sorted() on an all-tie array is stable -- so BM25
            #     candidates would silently become "whichever chunks were
            #     ingested first," not a real signal.
            #  2. _RERANK_MODEL is English-only (MS MARCO); confirmed live
            #     against real Devanagari eval queries that it can actively
            #     DEMOTE the correct chunk below where the multilingual
            #     embedding search alone had already ranked it correctly.
            # So for this case, skip both BM25 and the reranker, and trust
            # the multilingual embedding ranking as-is.
            return self._embedding_only_results(query, top_k, candidate_pool, allowed_scopes)

        queries = query_llm.expand_query(query) if expand else [query]
        return self._hybrid_rerank_results(queries, eligible, top_k, candidate_pool, allowed_scopes)

    def _embed_candidates(self, query: str, candidate_pool: int, allowed_scopes: set) -> dict:
        """Runs one embedding query; returns {doc_id: distance} in the
        best-first order Chroma already returns them in."""
        where = {"account_id": {"$in": list(allowed_scopes)}}
        result = self._collection.query(query_embeddings=[embed_query(query)], n_results=candidate_pool, where=where)
        return dict(zip(result["ids"][0], result["distances"][0]))

    def _embedding_only_results(self, query: str, top_k: int, candidate_pool: int, allowed_scopes: set) -> list[dict]:
        distance_by_id = self._embed_candidates(query, candidate_pool, allowed_scopes)
        ranked_ids = list(distance_by_id)  # already best-first from Chroma
        return [
            {
                "text": self._texts[i],
                "relevance_score": -distance_by_id[self._ids[i]],  # higher = more relevant, matching rerank_score's sense
                **{k: v for k, v in self._metadatas[i].items()},
            }
            for i in [self._ids.index(doc_id) for doc_id in ranked_ids][:top_k]
        ]

    def _hybrid_rerank_results(
        self, queries: list[str], eligible: list[int], top_k: int, candidate_pool: int, allowed_scopes: set,
    ) -> list[dict]:
        """Pools BM25 + embedding candidates across every query variant,
        then reranks the union against EVERY variant and keeps each
        candidate's best score -- a candidate only needs to answer one
        phrasing of the underlying question to rank well, which is the
        whole point of expansion (queries beyond the first are only ever
        real when expand=True passed more than one variant in)."""
        candidate_indices = set()
        for q in queries:
            q_tokens = _tokenize(q)
            if q_tokens:
                bm25_scores = self._bm25.get_scores(q_tokens)
                bm25_ranked = sorted(eligible, key=lambda i: bm25_scores[i], reverse=True)
                candidate_indices |= set(bm25_ranked[:candidate_pool])
            embed_ids = self._embed_candidates(q, candidate_pool, allowed_scopes)
            candidate_indices |= {self._ids.index(doc_id) for doc_id in embed_ids}

        if not candidate_indices:
            return []

        best_score: dict[int, float] = {}
        for q in queries:
            pairs = [(q, self._texts[i]) for i in candidate_indices]
            for i, score in zip(candidate_indices, self._reranker.predict(pairs)):
                score = float(score)
                if score > best_score.get(i, float("-inf")):
                    best_score[i] = score

        ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "text": self._texts[i],
                "relevance_score": score,
                **{k: v for k, v in self._metadatas[i].items()},
            }
            for i, score in ranked[:top_k]
        ]
