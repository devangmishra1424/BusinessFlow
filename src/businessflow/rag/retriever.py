"""Hybrid retrieval over whatever's actually been ingested into the
persistent store: BM25 keyword search + multilingual-e5 embedding search,
combined and reranked by a cross-encoder. Renamed from PolicyRetriever --
this now serves any ingested document (policy, loan agreements,
regulatory circulars), not just the hand-written policy KB it started as.

One consistent snapshot. A DocumentRetriever loads every active chunk WITH
its embedding when it is built and answers every query from that memory --
BM25, embedding search and reranking all see exactly the same chunks. It
used to read the text from a snapshot but run the embedding search live
against the table, so a chunk ingested after the snapshot was found by the
embedding search and then missing from the snapshot (a ValueError for the
caller), while a re-ingested document's old text stayed searchable by BM25.
Exact in-memory search is the right tool at this size (hundreds to a few
thousand chunks: well under a megabyte of vectors); the pgvector HNSW index
stays in the schema as the scale-out path, which would need
hnsw.iterative_scan for the account-filtered queries. Staying current is the
cache's job (tools/policy_tools.py): it compares corpus_fingerprint() and
rebuilds only when the corpus actually changed.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from businessflow.accounts.db import get_connection
from businessflow.rag import query_llm
from businessflow.rag.embeddings import embed_query
from businessflow.rag.tokenize import dominant_script as _dominant_script
from businessflow.rag.tokenize import tokenize as _tokenize
from businessflow.rag.tokenize import with_headings

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # English-only -- see _embedding_only_results below


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """One reranker per process, shared by every DocumentRetriever. The old code loaded a fresh one in every
    constructor, so each 10-minute rebuild cost ~13 s (measured) on whichever request triggered it."""
    return CrossEncoder(_RERANK_MODEL, backend="onnx", model_kwargs={"file_name": "onnx/model_quint8_avx2.onnx"})


_ACTIVE_CHUNKS_SQL = (
    "select id, document_text, source_document, document_type, account_id, chunk_index, headings, "
    "embedding::real[] as embedding from document_chunks where superseded_at is null "
    "order by source_document, chunk_index"
)
_FINGERPRINT_SQL = (
    "select count(*) as total, count(*) filter (where superseded_at is null) as active, "
    "max(created_at) filter (where superseded_at is null) as newest, "
    "max(superseded_at) as last_superseded from document_chunks"
)


def corpus_fingerprint() -> tuple:
    """A cheap summary of which chunks are active right now: it changes whenever a document is ingested,
    superseded or purged. One small query, so a long-running process can check it every few seconds and
    rebuild its DocumentRetriever only when it differs from the one the snapshot was built under."""
    row = get_connection().execute(_FINGERPRINT_SQL).fetchone()
    return (row["total"], row["active"], row["newest"], row["last_superseded"])


# late_fee_policy.md's own real text: "...grace period (see grace_period.md)
# is charged..." -- the exact, hand-written convention this project's
# policy docs already use to cross-reference each other. Extracts the
# referenced filename fragment, matched case-insensitively against known
# source_document basenames below.
_EXPLICIT_REFERENCE_RE = re.compile(r"\(see ([\w./-]+\.\w+)\)", re.IGNORECASE)

_RRF_K = 60  # the conventional reciprocal-rank-fusion constant

_MIN_TITLE_WORDS = 2  # excludes a short, generic single-word title (e.g. "faq") from ever substring-matching


def _derive_title(source_document: str) -> str | None:
    """A human-readable name derived from the file itself, not from its
    own internal headings -- confirmed live against this project's real
    uploaded loan agreements that a real per-account document's own
    chunk_index=0 heading is often just letterhead (a lender's company
    name), not anything another document would ever reference by name.
    Filenames are always well-formed and controllable, so they're the
    reliable source here instead: "data/kb/grace_period.md" -> "grace
    period"; "sunita_patil_loan_agreement.pdf" -> "sunita patil loan
    agreement". None for a title too short/generic to safely
    substring-match against (see _MIN_TITLE_WORDS)."""
    title = Path(source_document).stem.replace("_", " ").replace("-", " ").strip().lower()
    return title if len(title.split()) >= _MIN_TITLE_WORDS else None


def _find_referenced_documents(text: str, known_titles: dict[str, str], exclude: str) -> set[str]:
    """source_document values (from known_titles) that text appears to
    reference -- exact "(see X.md)" citations plus a broader substring
    check against every OTHER known document's derived title. exclude is
    the chunk's OWN source_document, so a document never "references"
    itself just because its own title appears in its own text.

    The substring check is a real, explainable heuristic, not NLP -- it
    catches "as per the late fee policy" but could miss a paraphrase, or
    (bounded by _MIN_TITLE_WORDS) very rarely hit a coincidental match on
    a short title. That's a known, accepted limitation of staying
    light/explainable rather than reaching for real entity extraction."""
    lower_text = text.lower()
    found = set()

    for match in _EXPLICIT_REFERENCE_RE.finditer(text):
        cited = match.group(1).lower()
        for source_document in known_titles:
            if source_document == exclude:
                continue
            if Path(source_document).name.lower() == cited or Path(source_document).stem.lower() == Path(cited).stem.lower():
                found.add(source_document)

    for source_document, title in known_titles.items():
        if source_document == exclude or source_document in found:
            continue
        if title in lower_text:
            found.add(source_document)

    return found


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
    """An immutable snapshot of every active chunk (text, metadata and embedding), built once at construction.
    Rebuild -- or better, let tools/policy_tools.py's cache do it when corpus_fingerprint() changes -- rather than
    reusing an instance for the lifetime of a long-running process.

    `rows` is for callers that already hold the chunks (the retrieval benchmark builds its corpus from files, with no
    database); each row needs id, document_text, source_document, document_type, account_id, chunk_index, headings and
    embedding. Left out, the rows are read from document_chunks.

    Two options exist so the benchmark (eval/retrieval_benchmark.py) can test them; both default to what it found:
    `use_headings` puts each chunk's heading path in front of its text for BM25 and the reranker (measured: no
    difference on held-out questions, so off), and `fusion` is how the final order is made -- "rerank" (the
    cross-encoder's order) or "rrf" (reciprocal-rank fusion of the cross-encoder's and the embedding's orders).
    `keyword_overlap_required` keeps a chunk that shares no word with the query out of the keyword candidates (when
    the corpus holds fewer chunks with the query's words than the pool has slots, the old code padded the pool with
    zero-score chunks in ingestion order); False restores that.
    """

    def __init__(self, rows: list[dict] | None = None, *, use_headings: bool = False, fusion: str = "rerank",
                 keyword_overlap_required: bool = True):
        if fusion not in ("rerank", "rrf"):
            raise ValueError(f"fusion must be 'rerank' or 'rrf', got {fusion!r}")
        self.fusion = fusion
        self.keyword_overlap_required = keyword_overlap_required
        if rows is None:
            # The fingerprint is read BEFORE the rows: if a document lands in between, the next fingerprint check
            # sees a difference and rebuilds, whereas the other order could record "up to date" for stale rows.
            self.fingerprint = corpus_fingerprint()
            rows = get_connection().execute(_ACTIVE_CHUNKS_SQL).fetchall()
        else:
            self.fingerprint = None

        self._ids = [r["id"] for r in rows]
        self._texts = [r["document_text"] for r in rows]
        self._metadatas = [
            {
                "source_document": r["source_document"], "document_type": r["document_type"],
                "account_id": r["account_id"], "chunk_index": r["chunk_index"], "headings": r["headings"],
            }
            for r in rows
        ]
        self._embeddings = np.asarray([r["embedding"] for r in rows], dtype=np.float32)

        # What BM25 and the reranker read. Not what is returned.
        self._search_texts = [with_headings(m["headings"], t) if use_headings else t for m, t in zip(self._metadatas, self._texts)]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._search_texts]) if self._texts else None

        self._by_scope: dict[str, list[int]] = {}
        for i, m in enumerate(self._metadatas):
            self._by_scope.setdefault(m["account_id"], []).append(i)

        self._reranker = _load_reranker()

        self._document_titles = {
            m["source_document"]: title
            for m in self._metadatas
            if (title := _derive_title(m["source_document"])) is not None
        }

    def retrieve(
        self, query: str, top_k: int = 2, candidate_pool: int = 4,
        account_id: str | None = None, expand: bool = False, document_type: str | None = None,
        follow_references: bool = False, max_referenced_chunks: int = 2,
    ) -> list[dict]:
        """Up to top_k chunks most relevant to a free-text query, after
        reranking. account_id, if given, restricts results to general
        documents plus that specific borrower's own documents -- never a
        different borrower's. document_type, if given, additionally
        restricts to chunks ingested with that exact document_type (e.g.
        "loan_agreement"). candidate_pool is how many candidates each of
        BM25 and embedding search contribute before reranking, per query
        variant. expand=True additionally
        generates a couple of alternate phrasings of the query
        (query_llm.expand_query) and pools candidates across all of them --
        a real extra Groq call, so it's opt-in; see eval/retrieval_benchmark.py
        for the measured effect.

        follow_references=True additionally checks each result for a
        mention of another known document (either the hand-written
        "(see X.md)" convention, or a broader substring match against
        every OTHER document's own filename-derived title -- see
        _find_referenced_documents) and, for up to max_referenced_chunks
        of them, pulls in that document's own best-matching chunk too --
        even if it wouldn't have ranked in the original top_k on its own.

        Each result carries score_kind: "rerank" (cross-encoder logit),
        "embedding" (negative cosine distance) or "mixed" (min-max
        normalised across both). Scores are only comparable within a kind.
        """
        if not self._texts:
            return []

        allowed_scopes = {"general", account_id} if account_id else {"general"}
        eligible = [
            i for scope in allowed_scopes for i in self._by_scope.get(scope, ())
            if document_type is None or self._metadatas[i]["document_type"] == document_type
        ]
        eligible.sort()
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
            results = self._embedding_only_results(query, top_k, candidate_pool, eligible)
        else:
            queries = query_llm.expand_query(query) if expand else [query]
            results = self._hybrid_rerank_results(queries, eligible, top_k, candidate_pool)

        if follow_references and results:
            results = self._expand_with_references(results, query, eligible, max_referenced_chunks)
        return results

    def _result(self, i: int, score: float, score_kind: str, **extra) -> dict:
        return {"text": self._texts[i], "relevance_score": score, "score_kind": score_kind, **self._metadatas[i], **extra}

    def _expand_with_references(self, results: list[dict], query: str, eligible: list[int], max_referenced_chunks: int) -> list[dict]:
        already_included = {r["source_document"] for r in results}
        referenced: list[str] = []
        for r in results:
            for source_document in _find_referenced_documents(r["text"], self._document_titles, exclude=r["source_document"]):
                if source_document not in already_included and source_document not in referenced:
                    referenced.append(source_document)

        query_vector = np.asarray(embed_query(query), dtype=np.float32)
        for source_document in referenced[:max_referenced_chunks]:
            in_that_document = [i for i in eligible if self._metadatas[i]["source_document"] == source_document]
            if not in_that_document:
                continue  # the reference is to a document outside this caller's account scope / document_type
            distances = self._distances(query_vector, in_that_document)
            best = min(distances, key=distances.get)
            results.append(self._result(best, -distances[best], "embedding", referenced_from=True))
        return results

    def _distances(self, query_vector: np.ndarray, indices: list[int]) -> dict[int, float]:
        """Cosine distance (1 - similarity; both sides are L2-normalised) from the query to each indexed chunk."""
        sims = self._embeddings[indices] @ query_vector
        return {i: float(1.0 - s) for i, s in zip(indices, sims)}

    def _nearest(self, query_vector: np.ndarray, candidate_pool: int, eligible: list[int]) -> dict[int, float]:
        """{chunk index: distance} for the candidate_pool nearest eligible chunks, best first."""
        distances = self._distances(query_vector, eligible)
        return dict(sorted(distances.items(), key=lambda kv: kv[1])[:candidate_pool])

    def _embed_candidates(self, query: str, candidate_pool: int, eligible: list[int]) -> dict[int, float]:
        return self._nearest(np.asarray(embed_query(query), dtype=np.float32), candidate_pool, eligible)

    def _embedding_only_results(self, query: str, top_k: int, candidate_pool: int, eligible: list[int]) -> list[dict]:
        distance_by_index = self._embed_candidates(query, candidate_pool, eligible)
        return [self._result(i, -d, "embedding") for i, d in list(distance_by_index.items())[:top_k]]

    def _fuse(self, by_rerank: list[tuple[int, float]], query_vectors: list[np.ndarray], best_distance: dict[int, float]) -> list[tuple[int, float]]:
        """Reciprocal-rank fusion of the cross-encoder's order and the embedding's order (each candidate's nearest
        distance over the query variants). Parameter-free apart from the conventional k=60. The score each result
        keeps is still its cross-encoder score, the number the 'is this answerable' check reads."""
        candidates = [i for i, _ in by_rerank]
        for query_vector in query_vectors:
            for i, distance in self._distances(query_vector, candidates).items():
                if distance < best_distance.get(i, float("inf")):
                    best_distance[i] = distance
        rerank_rank = {i: r for r, i in enumerate(candidates, start=1)}
        embedding_rank = {i: r for r, i in enumerate(sorted(candidates, key=lambda i: best_distance[i]), start=1)}
        fused = {i: 1 / (_RRF_K + rerank_rank[i]) + 1 / (_RRF_K + embedding_rank[i]) for i in candidates}
        score = dict(by_rerank)
        return [(i, score[i]) for i in sorted(candidates, key=lambda i: fused[i], reverse=True)]

    def _hybrid_rerank_results(self, queries: list[str], eligible: list[int], top_k: int, candidate_pool: int) -> list[dict]:
        """Pools BM25 + embedding candidates across every query variant,
        then reranks the union against EVERY variant and keeps each
        candidate's best score -- a candidate only needs to answer one
        phrasing of the underlying question to rank well, which is the
        whole point of expansion (queries beyond the first are only ever
        real when expand=True passed more than one variant in).

        _RERANK_MODEL (English-only MS MARCO) can DEMOTE a correct
        Devanagari result when the QUERY has no Latin content -- handled
        in retrieve() by skipping straight to _embedding_only_results. The
        same unreliable pairing happens here even when the query IS
        English, if a CANDIDATE CHUNK is itself Devanagari-dominant (a real
        Hindi-language ingested document). So before calling the
        cross-encoder, candidates are split by their OWN text's dominant
        script: Devanagari-dominant candidates skip the cross-encoder
        entirely and are scored by embedding distance instead.
        latin/mixed/none candidates are reranked exactly as before.

        The two groups' scores are not on a comparable scale (cross-encoder
        logits vs. embedding distance), so each group is independently
        min-max normalized to [0, 1] (see _min_max_normalize) before the
        merged sort. Chose min-max over a plain rank-position merge because
        it preserves within-group separation in the final score. When
        there are no Devanagari candidates at all (the common case), this
        whole split is skipped and the scores are the raw cross-encoder ones.
        """
        candidate_indices = set()
        best_distance: dict[int, float] = {}
        query_vectors: list[np.ndarray] = []
        for q in queries:
            q_tokens = _tokenize(q)
            if q_tokens:
                bm25_scores = self._bm25.get_scores(q_tokens)
                # A chunk sharing no word with the query is not a keyword candidate, however empty the pool is.
                bm25_ranked = [i for i in sorted(eligible, key=lambda i: bm25_scores[i], reverse=True)
                               if bm25_scores[i] > 0 or not self.keyword_overlap_required]
                candidate_indices |= set(bm25_ranked[:candidate_pool])
            query_vector = np.asarray(embed_query(q), dtype=np.float32)
            query_vectors.append(query_vector)
            for i, distance in self._nearest(query_vector, candidate_pool, eligible).items():
                candidate_indices.add(i)
                if distance < best_distance.get(i, float("inf")):
                    best_distance[i] = distance

        if not candidate_indices:
            return []

        devanagari_indices = {i for i in candidate_indices if _dominant_script(self._texts[i]) == "devanagari"}
        reranked_indices = sorted(candidate_indices - devanagari_indices)

        best_score: dict[int, float] = {}
        if reranked_indices:
            for q in queries:
                pairs = [(q, self._search_texts[i]) for i in reranked_indices]
                for i, score in zip(reranked_indices, self._reranker.predict(pairs)):
                    score = float(score)
                    if score > best_score.get(i, float("-inf")):
                        best_score[i] = score

        if not devanagari_indices:
            ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
            if self.fusion == "rrf":
                ranked = self._fuse(ranked, query_vectors, best_distance)
            kind = "rerank"
        else:
            # A Devanagari candidate that never appeared in ANY query
            # variant's top candidate_pool embedding results (only entered
            # via a BM25 tie) has no embedding distance to reuse, so it is
            # treated as the least relevant member of its own group.
            fallback_distance = max(best_distance.values(), default=0.0) + 1.0
            devanagari_raw = {i: -best_distance.get(i, fallback_distance) for i in devanagari_indices}

            normalized = _min_max_normalize(best_score)
            normalized.update(_min_max_normalize(devanagari_raw))
            ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
            kind = "mixed"

        return [self._result(i, score, kind) for i, score in ranked[:top_k]]
