"""Measures hybrid retrieval quality directly -- Recall@1/3/5 and MRR --
against a labeled (query -> correct heading) set, instead of the only
existing signal (tests/test_retriever.py's 4 clean, hand-picked
queries, which check correctness but compute no metric).

Every query here is written the way a real borrower actually writes,
not the tidy sentences a benchmark author would default to: typos,
Hindi-English code-switching, vague-but-answerable colloquial phrasing,
run-on venting with a real question buried in it, and pure Devanagari
script. That last style is a genuine stress test, not decoration --
retriever.py's BM25 tokenizer is `[a-z0-9]+` after lowercasing, so a
Devanagari-script query produces ZERO BM25 tokens and falls back
entirely on the multilingual e5 embedding candidates. This benchmark is
what actually shows whether that fallback holds up, rather than
assuming either way.

Ground truth (acceptable_headings) was written by reading every KB doc
in data/kb/*.md directly, not inferred -- and reuses the exact heading
substrings tests/test_retriever.py already asserted true, so those
tests and this benchmark agree by construction.

Calls DocumentRetriever.retrieve() directly (top_k=5, candidate_pool=8),
not the check_policy tool, since the tool's own top_k=2 cap would hide
whether the retriever is actually capable of a good rank-3/5 recall --
top_k=1/3/5 recall is computed from that single top-5 ranked list per
query (a hit in top-1 is necessarily a hit in top-3 and top-5).

No Postgres dependency -- this only reads the persistent Chroma KB
store. Assumes scripts/seed_kb.py has already been run, same
requirement as tests/test_retriever.py.

Run from the project root: python -m eval.retrieval_benchmark
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from businessflow.rag.retriever import DocumentRetriever

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_TOP_K = 5
_CANDIDATE_POOL = 8


@dataclass
class Query:
    text: str
    acceptable_headings: list[str]
    style: str  # "clean" | "typo_slang" | "hinglish" | "vague_colloquial" | "run_on_venting" | "devanagari"
    language: str  # "en" | "hi" | "hinglish"


QUERIES = [
    # --- dispute_handling.md ("Dispute handling policy", no subheadings) ---
    Query("is there someone I can talk to about a payment that's not showing up on my end",
          ["Dispute handling"], "clean", "en"),
    Query("yo i defo already paid tht EMI last wk trust me, sumthing wrong w ur system, pls fix???",
          ["Dispute handling"], "typo_slang", "en"),
    Query("mera jo late fee laga hai na wo galat hai yaar, maine to time pe paisa diya tha",
          ["Dispute handling"], "hinglish", "hinglish"),
    Query("somethings off with my account balance, doesnt look right to me at all",
          ["Dispute handling"], "vague_colloquial", "en"),
    Query("ok so this is driving me crazy the amount you guys are showing is just wrong i paid already "
          "and idk why its still saying pending this cant be right can someone look into this",
          ["Dispute handling"], "run_on_venting", "en"),

    # --- escalation_policy.md ("Escalation policy") ---
    Query("when does this actually get handed off to an actual human being",
          ["Escalation policy"], "clean", "en"),
    Query("agar mai baar baar promise tod du to kya hoga, insaan se baat hogi kya",
          ["Escalation policy", "What blocks an automated offer"], "hinglish", "hinglish"),
    Query("can this get bumped up to a real person or nah",
          ["Escalation policy"], "vague_colloquial", "en"),
    Query("अगर मैं दो बार वादा तोड़ दूं तो क्या होगा",
          ["Escalation policy", "What blocks an automated offer"], "devanagari", "hi"),

    # --- faq_general.md > Promise to pay ---
    Query("if I promise to pay by the 25th and I actually pay on the 27th does that still count",
          ["Promise to pay"], "clean", "en"),
    Query("does it still count if im lyk 2 days off from wat i promised",
          ["Promise to pay"], "typo_slang", "en"),
    Query("maine 20 tak dene ka bola tha, 2 din late ho gaya to chalega kya",
          ["Promise to pay"], "hinglish", "hinglish"),

    # --- faq_general.md > NACH auto-debit mandate ---
    Query("why is an AI calling me, doesn't the bank just auto-deduct this every month anyway",
          ["NACH auto-debit mandate"], "clean", "en"),
    Query("isnt this supposed to come out on its own why am i even hearing from u about it",
          ["NACH auto-debit mandate"], "vague_colloquial", "en"),
    Query("mera to auto debit hai na, phir ye call kyu aa rahi hai mujhe",
          ["NACH auto-debit mandate"], "hinglish", "hinglish"),

    # --- faq_general.md > What the agent cannot do ---
    Query("can you just waive this interest for me, just this once",
          ["What the agent cannot do"], "clean", "en"),
    Query("look i dont have the money can u just like remove the interest or lower the rate or "
          "something i really need a break here",
          ["What the agent cannot do"], "run_on_venting", "en"),
    Query("interest rate thoda kam kar do na please, itna zyada nahi de sakta",
          ["What the agent cannot do"], "hinglish", "hinglish"),

    # --- grace_period.md ("Grace period policy") ---
    Query("is there any buffer time before it's actually counted as late",
          ["Grace period"], "clean", "en"),
    Query("wil i get charged a late fee if im jst 2 days late paying",
          ["Grace period"], "typo_slang", "en"),
    Query("thoda time mil sakta hai kya bina late fee lagaye",
          ["Grace period"], "hinglish", "hinglish"),
    Query("क्या मुझे कुछ और दिन मिल सकते हैं बिना जुर्माने के",
          ["Grace period"], "devanagari", "hi"),

    # --- restructuring_options.md > Extend tenure ---
    Query("can you stretch out my loan so the monthly amount is smaller",
          ["Extend tenure"], "clean", "en"),
    Query("is there a way to make the monthly hit smaller even if it takes longer overall",
          ["Extend tenure"], "vague_colloquial", "en"),
    Query("EMI kam karne ka koi tarika hai kya, chahe time zyada lag jaye",
          ["Extend tenure"], "hinglish", "hinglish"),

    # --- restructuring_options.md > One-time settlement ---
    Query("if I pay it all off right now in one go do I get any kind of discount",
          ["One-time settlement"], "clean", "en"),
    Query("honestly im so done with this loan can i just pay the whole remaining thing today and be "
          "done with it forever",
          ["One-time settlement"], "run_on_venting", "en"),
    Query("poora loan ek baar mein khatam karna hai, discount milega kya usme",
          ["One-time settlement"], "hinglish", "hinglish"),

    # --- restructuring_options.md > What blocks an automated offer ---
    Query("why can't you just restructure my loan right now, whats the holdup",
          ["What blocks an automated offer", "Escalation policy"], "clean", "en"),
    Query("how come the system wont let this go through for me",
          ["What blocks an automated offer"], "vague_colloquial", "en"),

    # --- restructuring_options.md > Partial payment for a single cycle ---
    Query("whats the least amount I could pay this month and still be okay",
          ["Partial payment for a single cycle"], "clean", "en"),
    Query("is mahine thoda kam de doon to chalega kya, poora nahi hai mere paas abhi",
          ["Partial payment for a single cycle"], "hinglish", "hinglish"),
    Query("wats the min i can pay rn instead of the full emi amount",
          ["Partial payment for a single cycle"], "typo_slang", "en"),
]


def _is_hit(result: dict, acceptable_headings: list[str]) -> bool:
    headings = result.get("headings", "")
    return any(h in headings for h in acceptable_headings)


def _first_hit_rank(ranked: list[dict], acceptable_headings: list[str]) -> int | None:
    for rank, result in enumerate(ranked, start=1):
        if _is_hit(result, acceptable_headings):
            return rank
    return None


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "recall@1": None, "recall@3": None, "recall@5": None, "mrr": None}
    return {
        "n": n,
        "recall@1": round(sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 1) / n, 4),
        "recall@3": round(sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 3) / n, 4),
        "recall@5": round(sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5) / n, 4),
        "mrr": round(sum((1 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n, 4),
    }


def run(expand: bool = False) -> dict:
    retriever = DocumentRetriever()
    rows = []

    for q in QUERIES:
        ranked = retriever.retrieve(q.text, top_k=_TOP_K, candidate_pool=_CANDIDATE_POOL, expand=expand)
        rank = _first_hit_rank(ranked, q.acceptable_headings)
        rows.append({
            "query": q.text,
            "style": q.style,
            "language": q.language,
            "acceptable_headings": q.acceptable_headings,
            "rank": rank,
            "top_result_headings": ranked[0]["headings"] if ranked else None,
        })

    overall = _aggregate(rows)
    by_style = {style: _aggregate([r for r in rows if r["style"] == style]) for style in sorted({q.style for q in QUERIES})}
    by_language = {lang: _aggregate([r for r in rows if r["language"] == lang]) for lang in sorted({q.language for q in QUERIES})}

    return {"overall": overall, "by_style": by_style, "by_language": by_language, "rows": rows}


def _print_results(label: str, results: dict):
    print(f"=== {label} ===")
    print(json.dumps({"overall": results["overall"], "by_style": results["by_style"], "by_language": results["by_language"]},
                      indent=2, ensure_ascii=False))
    print()
    misses = [r for r in results["rows"] if r["rank"] is None or r["rank"] > 1]
    if misses:
        print(f"{len(misses)} / {len(results['rows'])} queries did NOT rank the correct chunk #1:")
        for r in misses:
            print(f"  [{r['style']}/{r['language']}] rank={r['rank']} expected one of {r['acceptable_headings']!r} "
                  f"got {r['top_result_headings']!r} -- {r['query']!r}")
    print()


def main():
    # Windows consoles default stdout to the legacy codepage (cp1252),
    # which can't encode the Devanagari-script queries in this eval.
    sys.stdout.reconfigure(encoding="utf-8")

    baseline = run(expand=False)
    _print_results("expand=False (translation only)", baseline)

    expanded = run(expand=True)
    _print_results("expand=True (translation + query expansion)", expanded)

    print("=== delta (expand=True vs expand=False) ===")
    for metric in ("recall@1", "recall@3", "recall@5", "mrr"):
        b, e = baseline["overall"][metric], expanded["overall"][metric]
        print(f"  {metric}: {b} -> {e} ({e - b:+.4f})")

    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = _RESULTS_DIR / "retrieval_benchmark.json"
    out_path.write_text(json.dumps({"expand_false": baseline, "expand_true": expanded}, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\nsaved to {out_path}")

    return baseline["overall"], expanded["overall"]


if __name__ == "__main__":
    main()
