"""Retrieval benchmark for the RAG pipeline -- built to be defended, not just to produce a number.

What it measures, on what, and what it cannot tell you
------------------------------------------------------
Corpus: the eight policy documents in data/kb/*.md plus nine loan agreements rendered through
scripts/seed_loan_agreements.py's own template from eval/fixtures/rag_benchmark/loan_accounts.json. Chunking and
embedding go through businessflow.rag.ingest (the code that ships), the corpus is searched by DocumentRetriever (the code
that ships), and nothing touches a database, so anyone can re-run it from a clean checkout and get the same numbers
(the models are deterministic; Hindi/Hinglish questions are translated once and the English is frozen in
translations.json).

Four question sets, none of them used to judge another:
  dev       38 policy questions the retriever was developed against. Settings are chosen here.
  heldout   47 policy questions written from a read of the documents before any variant was run on them, frozen by
            sha256. A setting is only believed if it also holds here.
  loan      10 facts x 3 phrasings x 9 borrowers = 270 questions about a borrower's OWN agreement (amount, tenure, EMI,
            rate, late fee, settlement discount, first EMI date, security, court, extension), asked with that
            borrower's account_id. A hit is a chunk of that borrower's own agreement containing the answer string -- or,
            for the three facts the general policy also states, the right policy chunk. These are 30 distinct questions
            asked nine times, so intervals resample the 30, not the 270. Doubles as the isolation test: no result may
            ever belong to another borrower.
  negative  30 questions the documents cannot answer (15 off-topic, 15 close to lending but absent).

Metrics: Recall@k (is a correct chunk among the first k results) and MRR, each with a 95% percentile bootstrap interval.
Each setting is compared with its PARENT (the setting it changes one thing from) on the same questions, paired.

The decision rule, fixed before the numbers were looked at: a change ships only if, on Recall@2 and MRR over the three
question sets, at least one paired interval is entirely above zero ("helps") and none is entirely below ("hurts"). A
change with "no measurable difference" ships only if it costs nothing (none does: each costs latency, a re-embedding of
the corpus, or code to maintain). An interval that includes zero is not a finding. 'Deployed view' is
Recall@2 because check_policy hands the agent two chunks.

"Is there anything relevant?": the top result's reranker score is compared with a threshold below which check_policy
tells the agent it found nothing relevant instead of handing over weak passages. The threshold is fixed by a rule, not by
looking at the unanswerable questions: the highest score that still keeps 95% of the dev questions' top results. It is
then checked on the held-out and loan questions (answerable, should be kept) and the 30 unanswerable ones (should be
dropped). Only results ranked by the cross-encoder (score_kind "rerank") carry a score this threshold applies to.

Limits, stated up front: the questions were written by the same engineer who built the retriever (the held-out set guards
against over-fitting to dev, it is not an independent audit); n is small, so intervals are wide (+-10 points on 40
questions); the policy corpus is eight short documents; the loan agreements all come from one template, so that set
tests finding the right clause in the right borrower's contract, not robustness to varied real contracts; ground truth
is a heading or an answer string, not a human judgement of whether the chunk is a good answer; the dev set is harder than
the held-out set (it was assembled from failures), so absolute numbers differ between them and only the paired
differences are comparable.

Run from the repo root:
    python -m eval.retrieval_benchmark                         # all settings, writes eval/results/retrieval_benchmark.{json,md}
    python -m eval.retrieval_benchmark --refresh-translations  # translate any new Hindi/Hinglish question with Groq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from eval.stats import bootstrap_ci, cluster_bootstrap_ci

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "eval" / "fixtures" / "rag_benchmark"
RESULTS = REPO / "eval" / "results"
KB_DIR = REPO / "data" / "kb"
TOP_K = 5  # ranking depth every metric is read from
SETS = ("dev", "heldout", "loan")
METRICS = ("recall@2", "mrr")


@dataclass(frozen=True)
class Setting:
    """One way of running the retriever. pool is retrieve()'s candidate_pool; headings_search puts the heading path in
    front of the text BM25 and the reranker read; headings_embed does the same for the text that is embedded; fusion is
    how the final order is made. `parent` is the setting this one changes one thing from."""

    name: str
    pool: int
    headings_search: bool
    headings_embed: bool
    parent: str | None = None
    fusion: str = "rerank"
    keyword_overlap_required: bool = True
    translate: bool = True  # False: Hindi/Hinglish questions go to the (English-only) keyword and rerank stages as written


BASELINE = "pool 4 (as deployed)"
# Every other row changes ONE thing from its parent. The first row is what check_policy runs (top_k=2, candidate_pool=4,
# chunk text only, cross-encoder order); tests/test_retrieval_quality.py checks it against the real constants so the two
# cannot drift. Under the decision rule above, none of the changes below was adopted.
SETTINGS = [
    Setting(BASELINE, 4, False, False),
    Setting("pool 4, zero-overlap keyword matches allowed (old behaviour)", 4, False, False, parent=BASELINE, keyword_overlap_required=False),
    Setting("pool 8", 8, False, False, parent=BASELINE),
    Setting("pool 16", 16, False, False, parent="pool 8"),
    Setting("pool 4 + headings in search text", 4, True, False, parent=BASELINE),
    Setting("pool 4 + headings everywhere", 4, True, True, parent=BASELINE),
    Setting("pool 4 + rank fusion", 4, False, False, parent=BASELINE, fusion="rrf"),
    # Not a candidate change: it measures what the LLM translation step is worth, so the README can state it. "Hurts" here
    # means removing the translation hurts, i.e. keep it.
    Setting("pool 4, no query translation", 4, False, False, parent=BASELINE, translate=False),
]
SHIPPED = BASELINE


# --------------------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------------------
def load_fixture(name: str) -> dict:
    path = FIXTURES / name
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return data


class Translator:
    """query_llm.translate_to_english replaced by a lookup into the frozen translations.json, so a run does not depend
    on a live LLM. A question with no frozen translation is an error, not a silent fallback to the original text."""

    def __init__(self, refresh: bool):
        self.path = FIXTURES / "translations.json"
        self.table: dict[str, str] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.refresh = refresh

    def __call__(self, query: str) -> str:
        if query not in self.table:
            if not self.refresh:
                raise KeyError(f"no frozen translation for {query!r}; run with --refresh-translations (needs GROQ_API_KEY)")
            from businessflow.rag import query_llm

            translated = query_llm._real_translate(query)
            if translated == query:
                raise RuntimeError(f"translation of {query!r} failed (returned the input); not freezing it")
            self.table[query] = translated
            self.save()  # as it goes: a killed run must not lose translations already paid for
        return self.table[query]

    def save(self) -> None:
        self.path.write_text(json.dumps(self.table, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def install_translator(translator: Translator) -> None:
    from businessflow.rag import query_llm

    if not hasattr(query_llm, "_real_translate"):
        query_llm._real_translate = query_llm.translate_to_english
    query_llm.translate_to_english = translator


# --------------------------------------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Corpus:
    rows_meta: list[dict]  # id, text, source_document, document_type, account_id, chunk_index, headings
    embeddings_plain: np.ndarray
    embeddings_headings: np.ndarray
    agreements: dict[str, str]  # account_id -> rendered agreement text
    accounts: list[dict]

    def rows(self, headings_embed: bool) -> list[dict]:
        emb = self.embeddings_headings if headings_embed else self.embeddings_plain
        return [
            {
                "id": m["id"], "document_text": m["text"], "source_document": m["source_document"],
                "document_type": m["document_type"], "account_id": m["account_id"], "chunk_index": m["chunk_index"],
                "headings": m["headings"], "embedding": emb[i],
            }
            for i, m in enumerate(self.rows_meta)
        ]


def _agreement_facts(account: dict, template_row: tuple) -> dict[str, str]:
    """The answer string for each fact, exactly as scripts/seed_loan_agreements.py writes it."""
    from businessflow.accounts.policy import LATE_FEE_FLAT_AMOUNT, MAX_RESTRUCTURING_EXTENSION_MONTHS, SETTLEMENT_DISCOUNT_PCT

    emis_elapsed = account["tenure_months"] - account["months_remaining"]
    first_emi = date.fromisoformat(account["emi_due_date"]) - timedelta(days=30 * emis_elapsed)
    secured = template_row[7] != "unsecured"
    return {
        "principal": f"{account['principal']:,}",
        "tenure": f"{account['tenure_months']} months",
        "emi": f"{account['emi']:,.2f}",
        "rate": f"{account['rate']}%",
        "late_fee": f"Rs. {LATE_FEE_FLAT_AMOUNT}",
        "settlement": f"{SETTLEMENT_DISCOUNT_PCT * 100:.0f}% (percent)",
        "first_emi": first_emi.strftime("%d %B %Y"),
        "security": "hypothecates" if secured else "unsecured",
        "court": "courts at Mumbai",
        "extension": f"{MAX_RESTRUCTURING_EXTENSION_MONTHS} (three) months",
    }


def build_corpus() -> Corpus:
    from scripts import seed_loan_agreements as seed

    from businessflow.rag.embeddings import embed_passages
    from businessflow.rag.ingest import parse_and_chunk

    accounts = load_fixture("loan_accounts.json")["accounts"]
    template_by_id = {row[0]: row for row in seed._ACCOUNTS}

    rows_meta: list[dict] = []
    records_all = []
    agreements: dict[str, str] = {}

    def add(path: Path, source_document: str, document_type: str, account_id: str) -> None:
        for i, record in enumerate(parse_and_chunk(str(path))):
            rows_meta.append({
                "id": f"{source_document}::{i}", "text": record.text, "source_document": source_document,
                "document_type": document_type, "account_id": account_id, "chunk_index": i, "headings": record.headings,
            })
            records_all.append(record)

    for path in sorted(KB_DIR.glob("*.md")):
        add(path, f"data/kb/{path.name}", "policy", "general")

    with tempfile.TemporaryDirectory() as tmp:
        for account in accounts:
            t = template_by_id[account["account_id"]]
            assert abs(t[8] - account["rate"]) < 1e-9, f"fixture rate disagrees with the seed template for {account['account_id']}"
            text = seed._build_document(
                t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[10],
                float(account["principal"]), float(account["emi"]), account["tenure_months"], account["months_remaining"],
                account["loan_type"], date.fromisoformat(account["emi_due_date"]),
            )
            agreements[account["account_id"]] = text
            filename = f"{seed._slug(t[1])}_loan_agreement.md"
            path = Path(tmp) / filename
            path.write_text(text, encoding="utf-8")
            add(path, f"data/documents/{account['account_id']}/{filename}", "loan_agreement", account["account_id"])

    plain = np.asarray(embed_passages([r.text for r in records_all]), dtype=np.float32)
    with_h = np.asarray(embed_passages([r.embedding_text for r in records_all]), dtype=np.float32)
    return Corpus(rows_meta, plain, with_h, agreements, accounts)


# --------------------------------------------------------------------------------------------------------------------
# question sets
# --------------------------------------------------------------------------------------------------------------------
# (fact, [clean, colloquial, hinglish] phrasings, headings of general policy chunks that also answer it)
FACTS = [
    ("principal", ["How much was my loan amount?", "what amount did i borrow from you", "mera loan kitne ka hai"], []),
    ("tenure", ["For how many months does my loan run?", "how long do i have to keep paying this loan", "mera loan kitne mahine ka hai"], []),
    ("emi", ["What is my monthly EMI amount?", "how much do i have to pay each month", "meri monthly kist kitni hai"], []),
    ("rate", ["What interest rate am I paying?", "how much interest do you charge me", "mera interest rate kitna hai"], []),
    ("late_fee", ["What is the late payment charge in my agreement?", "what do i pay if i miss the due date by a few days", "late hone par kitna charge lagega mere agreement me"], ["Late fee policy"]),
    ("settlement", ["Is there a discount if I settle my whole loan at once?", "can i close the loan early and pay less", "ek saath poora loan chukane par discount milta hai kya"], ["One-time settlement"]),
    ("first_emi", ["When was my first EMI due?", "from which date did my emi start", "meri pehli kist kab thi"], []),
    ("security", ["Is my loan secured or unsecured?", "did i have to put up any collateral for this loan", "mere loan ke liye koi security di thi kya"], []),
    ("court", ["Which courts have jurisdiction over my agreement?", "where would a legal dispute about my loan be heard", "agar court jana pada to kahan hoga"], []),
    ("extension", ["Can the repayment period of my loan be extended?", "can you give me more months to repay", "kya meri loan ki period badh sakti hai"], ["Extend tenure"]),
]


def loan_questions(corpus: Corpus) -> list[dict]:
    from scripts import seed_loan_agreements as seed

    template_by_id = {row[0]: row for row in seed._ACCOUNTS}
    questions = []
    for account in corpus.accounts:
        facts = _agreement_facts(account, template_by_id[account["account_id"]])
        agreement = corpus.agreements[account["account_id"]]
        for fact, phrasings, general_headings in FACTS:
            answer = facts[fact]
            assert answer.lower() in agreement.lower(), f"ground truth {fact}={answer!r} is not in {account['account_id']}'s agreement"
            for style, text in zip(("clean", "colloquial", "hinglish"), phrasings):
                questions.append({"text": text, "account_id": account["account_id"], "fact": fact, "answer": answer, "style": style,
                                  "general_headings": general_headings, "cluster": f"{fact}/{style}"})
    return questions


# --------------------------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------------------------
class CachedReranker:
    """The reranker with its scores remembered by (query, text). The model is deterministic, so this changes no result;
    it only stops settings that share candidates from scoring the same pair again. Never used for latency."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[tuple[str, str], float] = {}

    def predict(self, pairs):
        missing = [p for p in pairs if tuple(p) not in self.cache]
        if missing:
            for p, score in zip(missing, self.inner.predict(missing)):
                self.cache[tuple(p)] = float(score)
        return [self.cache[tuple(p)] for p in pairs]


_SHARED_RERANKER: CachedReranker | None = None


def build_retriever(corpus: Corpus, setting: Setting, cached: bool = True):
    from businessflow.rag.retriever import DocumentRetriever, _load_reranker

    retriever = DocumentRetriever(rows=corpus.rows(setting.headings_embed), use_headings=setting.headings_search,
                                  fusion=setting.fusion, keyword_overlap_required=setting.keyword_overlap_required)
    if cached:
        global _SHARED_RERANKER
        if _SHARED_RERANKER is None:
            _SHARED_RERANKER = CachedReranker(_load_reranker())
        retriever._reranker = _SHARED_RERANKER
    return retriever


def _first_rank(results: list[dict], is_hit) -> int | None:
    for rank, r in enumerate(results, start=1):
        if is_hit(r):
            return rank
    return None


def policy_ranks(retriever, queries: list[dict], setting: Setting) -> tuple[list[int | None], list[tuple[float, str]]]:
    """Per question: the rank of the first correct result, and (score, score_kind) of the top result."""
    ranks, tops = [], []
    for q in queries:
        results = retriever.retrieve(q["text"], top_k=TOP_K, candidate_pool=setting.pool)
        ranks.append(_first_rank(results, lambda r: any(h in r["headings"] for h in q["acceptable_headings"])))
        tops.append((results[0]["relevance_score"], results[0]["score_kind"]) if results else (float("-inf"), "none"))
    return ranks, tops


def _loan_hit(q: dict):
    return lambda r: (
        (r["account_id"] == q["account_id"] and q["answer"].lower() in r["text"].lower())
        or (r["account_id"] == "general" and any(h in r["headings"] for h in q["general_headings"]))
    )


def loan_ranks(retriever, questions: list[dict], setting: Setting) -> tuple[list[int | None], int, list[tuple[float, str]]]:
    ranks, leaks, tops = [], 0, []
    for q in questions:
        results = retriever.retrieve(q["text"], top_k=TOP_K, candidate_pool=setting.pool, account_id=q["account_id"])
        leaks += sum(1 for r in results if r["account_id"] not in ("general", q["account_id"]))
        ranks.append(_first_rank(results, _loan_hit(q)))
        tops.append((results[0]["relevance_score"], results[0]["score_kind"]) if results else (float("-inf"), "none"))
    return ranks, leaks, tops


def latency_ms(corpus: Corpus, setting: Setting, queries: list[dict], n: int = 40) -> dict:
    """Wall time of retrieve() with models warm and NO score cache, over the first n English questions (a question that
    needs translation would time a lookup in a dict, not the production LLM call, so those are left out)."""
    from businessflow.rag import query_llm

    retriever = build_retriever(corpus, setting, cached=False)
    english = [q for q in queries if not query_llm.needs_translation(q["text"])][:n]
    for q in english[:3]:
        retriever.retrieve(q["text"], top_k=TOP_K, candidate_pool=setting.pool)  # warm-up
    times = []
    for q in english:
        t0 = time.perf_counter()
        retriever.retrieve(q["text"], top_k=TOP_K, candidate_pool=setting.pool)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50": round(statistics.median(times), 1), "p95": round(float(np.percentile(times, 95)), 1), "n": len(times)}


def metric_values(ranks: list[int | None], metric: str) -> list[float]:
    if metric == "mrr":
        return [1.0 / r if r else 0.0 for r in ranks]
    k = int(metric.split("@")[1])
    return [1.0 if r is not None and r <= k else 0.0 for r in ranks]


def _ci(values: list[float], clusters: list[str] | None) -> tuple[float, float, float]:
    return cluster_bootstrap_ci(values, clusters) if clusters else bootstrap_ci(values)


def summarise(ranks: list[int | None], clusters: list[str] | None = None) -> dict:
    out = {"n": len(ranks), "distinct": len(set(clusters)) if clusters else len(ranks)}
    for metric in ("recall@1", "recall@2", "recall@3", "recall@5", "mrr"):
        mean, lo, hi = _ci(metric_values(ranks, metric), clusters)
        out[metric] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
    return out


def paired_difference(a: list[int | None], b: list[int | None], metric: str, clusters: list[str] | None = None) -> dict:
    """`metric` of a minus b over the same questions, with a bootstrap interval; also how many questions a wins/loses."""
    diffs = [x - y for x, y in zip(metric_values(a, metric), metric_values(b, metric))]
    mean, lo, hi = _ci(diffs, clusters)
    return {"diff": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "wins": sum(d > 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs)}


def verdict(paired: dict[str, dict[str, dict]]) -> str:
    """The decision rule from the module docstring, applied to the paired intervals (set -> metric -> interval)."""
    intervals = [paired[s][m] for s in SETS for m in METRICS]
    helps = any(i["lo"] > 0 for i in intervals)
    hurts = any(i["hi"] < 0 for i in intervals)
    if helps and hurts:
        return "mixed: helps on one set, hurts on another -- not shipped"
    if hurts:
        return "hurts -- not shipped"
    if helps:
        return "helps"
    return "no measurable difference -- not shipped"


def auc(positive: list[float], negative: list[float]) -> float:
    """Probability a random answerable question outscores a random unanswerable one (ties count half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


DEV_KEEP_FRACTION = 0.95


def answerability(dev: list[float], heldout: list[float], loan: list[float], negative: list[float], kinds: dict[str, list[str]]) -> dict:
    """Threshold from the dev questions alone (the highest score that keeps DEV_KEEP_FRACTION of their top results),
    then what it does to everything else. Scores are reranker logits; only results ranked by the cross-encoder count."""
    ordered = sorted(dev)
    threshold = ordered[int(len(ordered) * (1 - DEV_KEEP_FRACTION))]
    kept = lambda scores: round(sum(x >= threshold for x in scores) / len(scores), 3) if scores else None
    near = [sc for sc, k in zip(negative, kinds["negative"]) if k == "near_topic"]
    off = [sc for sc, k in zip(negative, kinds["negative"]) if k == "off_topic"]
    return {
        "threshold": round(threshold, 2), "dev_kept": kept(dev), "heldout_kept": kept(heldout), "loan_kept": kept(loan),
        "unanswerable_rejected": round(1 - kept(negative), 3) if negative else None,
        "off_topic_rejected": round(1 - kept(off), 3) if off else None, "near_topic_rejected": round(1 - kept(near), 3) if near else None,
        "n": {"dev": len(dev), "heldout": len(heldout), "loan": len(loan), "unanswerable": len(negative)},
    }


def evaluate(corpus: Corpus, settings: list[Setting], dev, heldout, loans, negatives) -> dict:
    loan_clusters = [q["cluster"] for q in loans]
    results: dict = {}
    from businessflow.rag import query_llm

    real_needs_translation = query_llm.needs_translation
    for s in settings:
        t0 = time.perf_counter()
        query_llm.needs_translation = real_needs_translation if s.translate else (lambda q: False)
        retriever = build_retriever(corpus, s)
        dev_r, dev_tops = policy_ranks(retriever, dev, s)
        held_r, held_tops = policy_ranks(retriever, heldout, s)
        loan_r, leaks, loan_tops = loan_ranks(retriever, loans, s)
        _, neg_tops = policy_ranks(retriever, [dict(q, acceptable_headings=[]) for q in negatives], s)
        pos = [sc for sc, kind in dev_tops + held_tops if kind == "rerank"]
        neg = [sc for sc, kind in neg_tops if kind == "rerank"]
        only_rerank = lambda tops: [sc for sc, kind in tops if kind == "rerank"]
        neg_kinds = [q["kind"] for q, (_, kind) in zip(negatives, neg_tops) if kind == "rerank"]
        by_fact = {f: sum(1 for q, r in zip(loans, loan_r) if q["fact"] == f and r is not None and r <= 2) / sum(1 for q in loans if q["fact"] == f) for f, _, _ in FACTS}
        by_style = {st: sum(1 for q, r in zip(loans, loan_r) if q["style"] == st and r is not None and r <= 2) / sum(1 for q in loans if q["style"] == st) for st in ("clean", "colloquial", "hinglish")}
        print(f"  {s.name}: done in {time.perf_counter() - t0:.0f} s", flush=True)
        results[s.name] = {
            "setting": s.__dict__, "dev": summarise(dev_r), "heldout": summarise(held_r), "loan": summarise(loan_r, loan_clusters),
            "loan_isolation_leaks": leaks, "loan_recall2_by_fact": {k: round(v, 3) for k, v in by_fact.items()},
            "loan_recall2_by_phrasing": {k: round(v, 3) for k, v in by_style.items()},
            "answerable_vs_unanswerable": {"auc": round(auc(pos, neg), 4) if pos and neg else None, "answerable_scores": pos, "unanswerable_scores": neg},
            "relevance_threshold": answerability(only_rerank(dev_tops), only_rerank(held_tops), only_rerank(loan_tops), neg, {"negative": neg_kinds}),
            "latency_ms": latency_ms(corpus, s, dev + heldout),
            "_ranks": {"dev": dev_r, "heldout": held_r, "loan": loan_r},
        }
    query_llm.needs_translation = real_needs_translation
    return results


# --------------------------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------------------------
def _cell(m: dict) -> str:
    return f"{m['mean']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}]"


def _signed(c: dict) -> str:
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] ({c['wins']}-{c['losses']})"


def markdown(summary: dict) -> str:
    first = next(iter(summary["results"].values()))
    lines = ["# RAG retrieval benchmark", "",
             "Read the header of `eval/retrieval_benchmark.py` for what is measured, the decision rule and the limits. Intervals "
             "are 95% percentile bootstrap. 'Deployed view' is Recall@2, because `check_policy` hands the agent two chunks.", ""]
    for key, title in (("dev", "Policy questions: dev set (settings were chosen here)"),
                       ("heldout", "Policy questions: held-out set (confirms the choice)"),
                       ("loan", "Loan-agreement facts, own agreement or the right policy chunk")):
        m0 = first[key]
        note = f", {m0['distinct']} distinct questions asked once per borrower; intervals resample those" if m0["distinct"] != m0["n"] else ""
        lines += [f"## {title} (n={m0['n']}{note})", "", "| setting | Recall@1 | Recall@2 (deployed view) | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|---|---|"]
        for name, r in summary["results"].items():
            m = r[key]
            lines.append(f"| {name} | {_cell(m['recall@1'])} | {_cell(m['recall@2'])} | {_cell(m['recall@3'])} | {_cell(m['recall@5'])} | {_cell(m['mrr'])} |")
        lines.append("")
    lines += ["## Each change against its parent: paired difference [95% interval] (questions won-lost)", "",
              "| setting | parent | Recall@2 dev | Recall@2 held-out | Recall@2 loan | MRR dev | MRR held-out | MRR loan | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for name, c in summary["vs_parent"].items():
        lines.append("| " + " | ".join([name, c["parent"]] + [_signed(c["paired"][s][m]) for m in METRICS for s in SETS] + [c["verdict"]]) + " |")
    lines += ["", "## Isolation and latency", "", "| setting | results from another borrower's documents | retrieve() p50 / p95 ms |", "|---|---|---|"]
    for name, r in summary["results"].items():
        lines.append(f"| {name} | {r['loan_isolation_leaks']} | {r['latency_ms']['p50']} / {r['latency_ms']['p95']} (n={r['latency_ms']['n']}) |")
    shipped = summary["results"][summary["shipped_setting"]]
    lines += ["", f"Isolation: 0 expected; {sum(r['loan_isolation_leaks'] for r in summary['results'].values())} across "
              f"{len(summary['results'])} settings x {first['loan']['n']} questions.", "",
              f"## Where the loan questions still miss ({summary['shipped_setting']}, Recall@2)", "",
              "| fact | " + " | ".join(shipped["loan_recall2_by_fact"]) + " |", "|---|" + "---|" * len(shipped["loan_recall2_by_fact"]),
              "| Recall@2 | " + " | ".join(f"{v:.2f}" for v in shipped["loan_recall2_by_fact"].values()) + " |", "",
              "By phrasing: " + ", ".join(f"{k} {v:.2f}" for k, v in shipped["loan_recall2_by_phrasing"].items()) + ".", "",
              "## Can the scores tell an answerable question from an unanswerable one?", "",
              "| setting | AUC of the top reranker score | answerable n / unanswerable n |", "|---|---|---|"]
    for name, r in summary["results"].items():
        a = r["answerable_vs_unanswerable"]
        lines.append(f"| {name} | {a['auc']} | {len(a['answerable_scores'])} / {len(a['unanswerable_scores'])} |")
    t = shipped["relevance_threshold"]
    lines += ["", f"### The 'nothing relevant' threshold ({summary['shipped_setting']})", "",
              f"Rule: the highest top-result score that keeps {DEV_KEEP_FRACTION:.0%} of the dev questions = **{t['threshold']}** (a cross-encoder logit). "
              "Checked on questions it was not set from:", "",
              "| should be kept | kept | | should be dropped | dropped |", "|---|---|---|---|---|",
              f"| dev (n={t['n']['dev']}) | {t['dev_kept']:.0%} | | all unanswerable (n={t['n']['unanswerable']}) | {t['unanswerable_rejected']:.0%} |",
              f"| held-out (n={t['n']['heldout']}) | {t['heldout_kept']:.0%} | | off-topic | {t['off_topic_rejected']:.0%} |",
              f"| loan facts (n={t['n']['loan']}) | {t['loan_kept']:.0%} | | near-topic (in the lending domain, absent from the documents) | {t['near_topic_rejected']:.0%} |", ""]
    lines += ["", "Fixtures (sha256): " + ", ".join(f"{k} `{v[:12]}`" for k, v in summary["fixtures"].items()), ""]
    return "\n".join(lines)


def run(settings: list[Setting], refresh: bool) -> dict:
    sys.stdout.reconfigure(encoding="utf-8")
    install_translator(Translator(refresh))
    dev_fx, held_fx, neg_fx = load_fixture("dev_queries.json"), load_fixture("heldout_queries.json"), load_fixture("negative_queries.json")
    corpus = build_corpus()
    loans = loan_questions(corpus)
    print(f"corpus: {len(corpus.rows_meta)} chunks; dev {len(dev_fx['queries'])}, held-out {len(held_fx['queries'])}, loan {len(loans)}, negative {len(neg_fx['queries'])}", flush=True)
    results = evaluate(corpus, settings, dev_fx["queries"], held_fx["queries"], loans, neg_fx["queries"])

    clusters = {"dev": None, "heldout": None, "loan": [q["cluster"] for q in loans]}
    vs_parent = {}
    for s in settings:
        if s.parent is None or s.parent not in results:
            continue
        a, b = results[s.name]["_ranks"], results[s.parent]["_ranks"]
        paired = {k: {m: paired_difference(a[k], b[k], m, clusters[k]) for m in METRICS} for k in SETS}
        vs_parent[s.name] = {"parent": s.parent, "paired": paired, "verdict": verdict(paired)}
    for r in results.values():
        r.pop("_ranks")
    return {
        "date": datetime.now().date().isoformat(),
        "fixtures": {"dev": dev_fx["sha256"], "heldout": held_fx["sha256"], "negative": neg_fx["sha256"], "loan_accounts": load_fixture("loan_accounts.json")["sha256"]},
        "shipped_setting": SHIPPED, "results": results, "vs_parent": vs_parent,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-translations", action="store_true")
    parser.add_argument("--only", help="comma-separated setting names to run (default: all); a parent must be included to be compared with")
    args = parser.parse_args()
    settings = SETTINGS if not args.only else [s for s in SETTINGS if s.name in args.only.split(",")]
    summary = run(settings, args.refresh_translations)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(summary)
    (RESULTS / "retrieval_benchmark.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
