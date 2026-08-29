"""Hybrid retrieval over whatever's actually been ingested into the
persistent store: BM25 keyword search + multilingual-e5 embedding search,
combined and reranked by a cross-encoder. Renamed from PolicyRetriever --
this now serves any ingested document (policy, loan agreements,
regulatory circulars), not just the hand-written policy KB it started as.
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from businessflow.accounts.db import get_connection
from businessflow.rag import query_llm
from businessflow.rag.embeddings import embed_query
from businessflow.rag.store import embedding_literal
from businessflow.rag.tokenize import dominant_script as _dominant_script
from businessflow.rag.tokenize import tokenize as _tokenize

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # English-only -- see _embedding_only_results below


def _min_max_normalize(raw_scores: dict[int, float]) -> dict[int, float]:
    """Independently rescales one group's scores to [0, 1], higher = more
    relevant. Used by _hybrid_rerank_results to merge the cross-encoder
    group's scores with the Devanagari-candidate group's embedding-distance
    scores onto a comparable scale before sorting them together -- see that
    method's docstring for why min-max was chosen over a rank-position
    merge, and why a flat group (one candidate, or a genuine tie: min ==
    max, so the real formula would divide by zero) maps every member to
    0.5 (neutral) instead of an arbitrary 1.0."""
    if not raw_scores:
        return {}
    lo, hi = min(raw_scores.values()), max(raw_scores.values())
    if hi == lo:
        return {i: 0.5 for i in raw_scores}
    return {i: (score - lo) / (hi - lo) for i, score in raw_scores.items()}


class DocumentRetriever:
    """Builds its BM25 index fresh from whatever's currently in the
    persistent store at construction time -- so it picks up anything
    ingested since the last time this was built. Rebuild (create a new
    instance) after ingesting new documents rather than reusing a stale
    one for the lifetime of a long-running process."""

    def __init__(self):
        self._conn = get_connection()

        # Active means superseded_at is null (see ingest.py's
        # _supersede_existing_chunks) -- a corrected document's old chunks
        # stay in the store for the compliance trail, but must never be
        # retrievable. A plain WHERE clause now; the live ANN query in
        # _embed_candidates uses the same condition directly rather than
        # needing a separately-tracked exclusion list the way Chroma's
        # filter grammar (no "is null" operator) once required.
        rows = self._conn.execute(
            "select id, document_text, source_document, document_type, account_id, chunk_index, headings "
            "from document_chunks where superseded_at is null"
        ).fetchall()
        self._ids = [r["id"] for r in rows]
        self._texts = [r["document_text"] for r in rows]
        self._metadatas = [
            {
                "source_document": r["source_document"], "document_type": r["document_type"],
                "account_id": r["account_id"], "chunk_index": r["chunk_index"], "headings": r["headings"],
            }
            for r in rows
        ]

        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts]) if self._texts else None

        self._reranker = CrossEncoder(
            _RERANK_MODEL, backend="onnx", model_kwargs={"file_name": "onnx/model_quint8_avx2.onnx"}
        )

    def retrieve(
        self, query: str, top_k: int = 2, candidate_pool: int = 4,
        account_id: str | None = None, expand: bool = False, document_type: str | None = None,
    ) -> list[dict]:
        """Up to top_k chunks most relevant to a free-text query, after
        reranking. account_id, if given, restricts results to general
        documents plus that specific borrower's own documents -- never a
        different borrower's. document_type, if given, additionally
        restricts to chunks ingested with that exact document_type (e.g.
        "loan_agreement") -- ingest.py has always stamped this on every
        chunk, but nothing read it back until now; every existing caller
        that omits it gets the previous, unfiltered behavior unchanged.
        candidate_pool is how many candidates each of BM25 and embedding
        search contribute before reranking, per query variant. expand=True
        additionally generates a couple of alternate phrasings of the
        query (query_llm.expand_query) and pools candidates across all of
        them -- a real extra Groq call, so it's opt-in rather than
        always-on; see eval/retrieval_benchmark.py for the measured effect
        before deciding whether to default it on for a given caller."""
        if not self._texts:
            return []

        allowed_scopes = {"general", account_id} if account_id else {"general"}
        eligible = [
            i for i, m in enumerate(self._metadatas)
            if m.get("account_id") in allowed_scopes and (document_type is None or m.get("document_type") == document_type)
        ]
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
            return self._embedding_only_results(query, top_k, candidate_pool, allowed_scopes, document_type)

        queries = query_llm.expand_query(query) if expand else [query]
        return self._hybrid_rerank_results(queries, eligible, top_k, candidate_pool, allowed_scopes, document_type)

    def _embed_candidates(self, query: str, candidate_pool: int, allowed_scopes: set, document_type: str | None = None) -> dict:
        """Runs one embedding query; returns {doc_id: distance} in the
        best-first order the ORDER BY already returns them in.

        A live ANN query straight against document_chunks, capped at
        candidate_pool results -- excludes superseded chunks in the same
        WHERE clause (not left to Python-side filtering after the fact,
        the way __init__'s own corpus load can afford to be): a stale
        chunk with a near-identical embedding to its own corrected
        successor could otherwise occupy one of a small pool's slots ahead
        of a genuinely active candidate, or get returned as a doc_id
        self._ids (active-only) can't look up at all.
        """
        params: list = [embedding_literal(embed_query(query))]
        where = "account_id = any(%s) and superseded_at is null"
        params.append(list(allowed_scopes))
        if document_type is not None:
            where += " and document_type = %s"
            params.append(document_type)
        params.append(candidate_pool)
        rows = self._conn.execute(
            f"select id, embedding <=> %s::vector as distance from document_chunks "
            f"where {where} order by distance limit %s",
            tuple(params),
        ).fetchall()
        return {r["id"]: r["distance"] for r in rows}

    def _embedding_only_results(
        self, query: str, top_k: int, candidate_pool: int, allowed_scopes: set, document_type: str | None = None,
    ) -> list[dict]:
        distance_by_id = self._embed_candidates(query, candidate_pool, allowed_scopes, document_type)
        ranked_ids = list(distance_by_id)  # already best-first from the ORDER BY
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
        document_type: str | None = None,
    ) -> list[dict]:
        """Pools BM25 + embedding candidates across every query variant,
        then reranks the union against EVERY variant and keeps each
        candidate's best score -- a candidate only needs to answer one
        phrasing of the underlying question to rank well, which is the
        whole point of expansion (queries beyond the first are only ever
        real when expand=True passed more than one variant in).

        retrieve()'s own comment above explains a real, already-found bug:
        _RERANK_MODEL (English-only MS MARCO) can DEMOTE a correct
        Devanagari result when the QUERY has no Latin content -- handled
        there by skipping straight to _embedding_only_results. That guard
        only looks at the query. The same unreliable English-only-reranker-
        scoring-Devanagari pairing happens here even when the query IS
        English, if a CANDIDATE CHUNK is itself Devanagari-dominant (e.g. a
        real Hindi-language ingested document, an RBI circular, a Hindi
        loan clause) -- the reranker still doesn't understand that chunk's
        language regardless of which side of the pair is Devanagari. So
        before calling the cross-encoder, candidates are split by their OWN
        text's dominant script: Devanagari-dominant candidates skip the
        cross-encoder entirely and are scored by embedding distance instead
        (the same multilingual signal _embedding_only_results already
        trusts for this exact reranker limitation) -- reusing the distances
        _embed_candidates already returned in the pooling loop below rather
        than issuing another embedding query. latin/mixed/none candidates
        are reranked exactly as before.

        The two groups' scores are not on a comparable scale (cross-encoder
        logits vs. embedding distance), so each group is independently
        min-max normalized to [0, 1] (see _min_max_normalize) before the
        merged sort. Chose min-max over a plain rank-position merge because
        it preserves within-group separation in the final score -- a
        Devanagari candidate that's a landslide best match by embedding
        distance should land near 1.0, not merely "ahead of its
        groupmates," which is all rank position would encode; with only
        rank position, a decisive win and a coin-toss margin would look
        identical in the merged score. When there are no Devanagari
        candidates at all (the common case), this whole split is skipped
        and scoring is byte-for-byte the original single-group behavior --
        no scale change for the path this fix isn't targeting.
        """
        candidate_indices = set()
        best_distance: dict[int, float] = {}
        for q in queries:
            q_tokens = _tokenize(q)
            if q_tokens:
                bm25_scores = self._bm25.get_scores(q_tokens)
                bm25_ranked = sorted(eligible, key=lambda i: bm25_scores[i], reverse=True)
                candidate_indices |= set(bm25_ranked[:candidate_pool])
            embed_ids = self._embed_candidates(q, candidate_pool, allowed_scopes, document_type)
            for doc_id, distance in embed_ids.items():
                i = self._ids.index(doc_id)
                candidate_indices.add(i)
                if distance < best_distance.get(i, float("inf")):
                    best_distance[i] = distance

        if not candidate_indices:
            return []

        devanagari_indices = {i for i in candidate_indices if _dominant_script(self._texts[i]) == "devanagari"}
        reranked_indices = candidate_indices - devanagari_indices

        best_score: dict[int, float] = {}
        if reranked_indices:
            for q in queries:
                pairs = [(q, self._texts[i]) for i in reranked_indices]
                for i, score in zip(reranked_indices, self._reranker.predict(pairs)):
                    score = float(score)
                    if score > best_score.get(i, float("-inf")):
                        best_score[i] = score

        if not devanagari_indices:
            # Common case, unchanged: no scale-merging needed or done.
            ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
        else:
            # A Devanagari candidate that never appeared in ANY query
            # variant's top candidate_pool embedding results (only entered
            # via a BM25 tie at score 0 -- the same degenerate "whichever
            # chunk was ingested first" tie-break retrieve()'s own comment
            # documents for the whole-query case) has no embedding distance
            # to reuse. Getting one would mean a fresh embedding call this
            # fix is deliberately avoiding, so instead it's treated as the
            # least relevant member of its own group -- conservative, not a
            # crash-avoidance hack: the embedding stage itself never judged
            # it a close match for any phrasing of this query.
            fallback_distance = max(best_distance.values(), default=0.0) + 1.0
            devanagari_raw = {i: -best_distance.get(i, fallback_distance) for i in devanagari_indices}

            normalized = _min_max_normalize(best_score)
            normalized.update(_min_max_normalize(devanagari_raw))
            ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "text": self._texts[i],
                "relevance_score": score,
                **{k: v for k, v in self._metadatas[i].items()},
            }
            for i, score in ranked[:top_k]
        ]
