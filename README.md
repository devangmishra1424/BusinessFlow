# BusinessFlow

A Hindi-English AI collections agent for Indian SMB lending -- a borrower can
check their balance, negotiate a restructuring, log a promise to pay, dispute
a charge, or ask a real policy question, in Hindi, English, or Hinglish, and
get an answer grounded in real account data and real policy documents rather
than a model's guess.

Everything here runs against **synthetic demo accounts**. No real payment
ever moves; a generated payment link points at a stub URL, not a live
payment gateway.

## Architecture, in one paragraph

One Groq-hosted LLM (`openai/gpt-oss-20b`) with a belt of 8 tools in a single
bounded reasoning loop -- not a multi-agent swarm. The live conversation has
no latency budget for extra hops between agents, so tool calls (account
lookups, promise logging, dispute flagging, policy retrieval, payment links,
escalation) all live in one loop (`agent/loop.py`), each grounded in a real
Postgres row or a real retrieved document chunk, never invented. A mechanical
**guardrail** (`guardrail/grounding.py`) double-checks every ₹ amount and URL
in the model's final reply against what a real tool result or the borrower's
own message actually said, before the reply ever reaches the borrower --
catching hallucinated numbers a prompt alone can't guarantee against.

A separate, deliberately simpler multi-agent feature exists for the
non-latency-sensitive **report-generation** flow (gather real facts -> write
a plain-language summary -> mechanically check every claim traces back to
those facts) -- see [Status](#status-and-known-gaps) for what's built so far.

## Repository layout

```
src/businessflow/
  accounts/       Account data model + the real Postgres-backed store
                   (accounts, payment history, promises, disputes, escalations)
  agent/          The tool-calling loop and Groq client/system-prompt assembly
  audio/          ASR (fine-tuned Hindi-English Whisper), TTS, VAD
  channels/       HTTP API for the borrower-facing text chat (browser_api.py)
  guardrail/      Post-hoc grounding check on every reply before it ships
  memory/         Cross-session conversation recap for a returning borrower
  observability/  Aggregate operational metrics on top of the events log
  ops/            Account flagging (overdue/disputed/broken-promises) +
                   the internal ops dashboard API (ops/api.py), including a
                   per-account document upload endpoint that feeds the same
                   RAG ingestion pipeline seed_kb.py uses for the policy KB
  outbound/       Proactive reminders: decide who's due/overdue, compose the
                   message, "send" it (a logged stub -- no real SMS/WhatsApp/
                   Telegram channel is wired in yet)
  rag/            Hybrid retrieval (BM25 + multilingual embeddings + a
                   cross-encoder reranker) over the policy knowledge base,
                   plus LLM-based query translation/expansion
  reports/        The on-demand report-generation pipeline (gather -> analyze
                   -> write -> accuracy-check)
  tools/          The 8 MCP tools the agent loop calls

scripts/          seed_accounts.py, seed_kb.py, chat.py (interactive CLI),
                   run_outbound_pass.py (manual trigger -- no scheduler yet)
eval/             Standalone benchmarks (tool-calling, retrieval, reasoning
                   accuracy, a red-team pass, ASR WER) -- not part of the
                   pytest suite; run each with `python -m eval.<name>`
tests/            The pytest suite
data/kb/          The policy knowledge base (markdown)
schema.sql        Reference snapshot of the real Postgres schema (Supabase) --
                   not a migration tool; kept in sync by hand
```

## Setup

Requires Python 3.12+.

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # then fill in the keys below
```

`.env` keys:

| Key | Required for |
|---|---|
| `GROQ_API_KEY` | Anything that talks to the LLM |
| `ALTERNATE_GROQ_KEY` | Automatic fallback when the primary key hits its rate limit (optional) |
| `DATABASE_URL` | Anything that touches account data (a Supabase/Postgres connection string) |
| `OPS_API_KEY` | The ops dashboard API (`ops/api.py`) -- required on every request as an `X-API-Key` header |
| `LANGFUSE_*` | Tracing (optional) |
| `HF_TOKEN`, `KAGGLE_API_TOKEN` | ASR model/dataset access (voice work only) |

Seed the database and the policy knowledge base once:

```bash
python -m scripts.seed_accounts
python -m scripts.seed_kb
```

## Running it

```bash
# Interactive text chat (CLI)
python -m scripts.chat --account BF-1001 --key 482913 --language en

# Borrower-facing chat API
uvicorn businessflow.channels.browser_api:app --reload            # port 8000

# Telegram bot (text + voice notes, long polling -- needs TELEGRAM_BOT_TOKEN)
python -m businessflow.channels.telegram_bot

# Internal ops dashboard API (needs OPS_API_KEY, sent as X-API-Key)
uvicorn businessflow.ops.api:app --reload --port 8001
```

## Testing

```bash
pytest tests/
```

Tests that need a real Groq call, a real Postgres connection, or
`OPS_API_KEY` are individually gated with `pytest.mark.skipif` and skip
cleanly if the relevant `.env` value isn't set -- nothing is mocked in this
project; a test either runs for real or is skipped, never faked. Expect this
to be slow (this session observed ~2 hours end to end) since it includes real
ASR/LLM/DB calls, not unit-test-speed mocks.

Standalone benchmarks under `eval/` (tool-calling accuracy, retrieval
Recall@k/MRR, ASR WER) aren't part of this suite -- run each directly, e.g.
`python -m eval.retrieval_benchmark`.

CI (`.github/workflows/tests.yml`) runs the same suite on every push/PR to
`main`; add the secrets above under the repo's Settings -> Secrets and
variables -> Actions for the gated tests to actually run there instead of
skipping.

## Status and known gaps

Built and verified end-to-end against real Groq/Postgres calls: the
tool-calling agent loop, the guardrail, hybrid retrieval with LLM-based query
translation (a real, measured fix for Hindi/Hinglish queries -- recall@1
0.61 -> 0.79 on the retrieval benchmark), the ops dashboard backend (account
flags, escalation queue, metrics), write-idempotency on the three
state-mutating tools, a brute-force lockout on the per-account access key,
`eval/reasoning_accuracy.py` (an LLM-judge check on whether a reply's stated
reasoning matches the real tool results, not just whether the right tool got
called -- this is what caught a real crash in `log_event` on a hallucinated
account_id, since fixed), `eval/red_team.py` (the blueprint's 8 core
adversarial scenarios -- 8/8 passing, after a prompt fix for a found gap: the
agent wasn't declining out-of-domain legal questions), the report-generation
pipeline (`eval.tool_calling_benchmark.py` and the live `generate_report()`
round-trip both pass), and multi-key Groq fallback (`agent/client.py`
rotates through as many `ALTERNATE_GROQ_KEY`/`ALTERNATE_GROQ_KEY{N}` keys as
are configured, not just one -- built and proven live during this project's
own real daily-quota exhaustion). `eval/tool_scoring.py`'s refactored
per-tool precision/recall aggregator and run-history/regression tracking are
live-verified too -- it correctly flagged a real regression during today's
work before a fix was confirmed.

One known, honestly-unresolved gap found via that benchmark work: on a
long, multi-turn conversation where an account's dispute/broken-promise
block has already come up several times, the agent sometimes states a new
concrete restructuring/partial-payment request is blocked from memory
instead of calling the tool to verify it -- confirmed fixed in the
single-turn case (`tool_calling_benchmark.py`, 11/11), but two separate
prompt refinements did not move this specific multi-turn case
(`realistic_conversation_benchmark.py`'s `many_operations_same_account_en`,
round 4). Documented rather than silently patched over; a mechanical
Guardrail-style check (verify against the real tool result before allowing
the claim through) would likely be the more reliable fix than further
prompt tuning, and is a good next step.

Query expansion (an optional `retrieve(..., expand=True)`) was built and
measured, but left off by default -- A/B testing showed no clear win on this
KB's size, and a real added cost (an extra LLM call per query).

Not yet built: end-to-end latency measurement. Voice, a second channel
(Telegram), a real frontend UI, and hosting are deliberately out of scope
for now.
