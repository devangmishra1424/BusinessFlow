# BusinessFlow

A Hindi-English AI collections agent for Indian SMB lending -- a borrower can
check their balance, negotiate a restructuring, log a promise to pay, dispute
a charge, or ask a real policy question, in Hindi, English, or Hinglish, and
get an answer grounded in real account data and real policy documents rather
than a model's guess. An ops team works the same accounts from an internal
dashboard: escalation queue, per-account document upload with automatic
interest-rate extraction, dispute history, and an EMI calculator for opening
new accounts.

Everything here runs against **synthetic demo accounts** -- no real person's
money or data. Payment *recording* is real (a redeemed payment link genuinely
updates the account's balance and due date in Postgres); the payment link
itself points at a confirmation page inside this app, not a live payment
gateway.

**Live:**
[businessflowai.duckdns.org](https://businessflowai.duckdns.org) (borrower chat + dashboard) ·
[businessflowai-ops.duckdns.org](https://businessflowai-ops.duckdns.org) (ops dashboard, needs an API key)

## Architecture, in one paragraph

One Groq-hosted LLM (`openai/gpt-oss-20b`) with a belt of tools in a single
bounded reasoning loop -- not a multi-agent swarm. The live conversation has
no latency budget for extra hops between agents, so tool calls (account
lookups, promise logging, dispute flagging, policy retrieval, payment links,
escalation) all live in one loop (`agent/loop.py`), each grounded in a real
Postgres row or a real retrieved document chunk, never invented. A mechanical
**guardrail** (`guardrail/grounding.py`) double-checks every ₹ amount and URL
in the model's final reply against what a real tool result or the borrower's
own message actually said, before the reply ever reaches the borrower --
catching hallucinated numbers a prompt alone can't guarantee against.

Three real channels sit on top of that one loop: a borrower-facing browser
chat + account dashboard (`channels/browser_api.py`), a Telegram bot with a
full slash-command menu (`/status`, `/pay`, `/dispute`, `/agent`, `/voice`,
and more -- `channels/telegram_bot.py`), and an internal ops dashboard
(`ops/api.py`) for staff to manage the same accounts. Retrieval (policy KB +
per-account uploaded documents) is hybrid BM25 + multilingual embeddings + a
cross-encoder reranker, backed by pgvector on the same Postgres project as
every other table -- not a separate local vector-store file, specifically so
every environment (a developer's laptop, CI, the VM) reads from one shared,
always-current index instead of N independently-seeded ones.

A separate, deliberately simpler multi-agent feature exists for the
non-latency-sensitive **report-generation** flow (gather real facts -> write
a plain-language summary -> mechanically check every claim traces back to
those facts) -- see [Status](#status-and-known-gaps).

## Repository layout

```
src/businessflow/
  accounts/       Account data model + the real Postgres-backed store
                   (accounts, payment history, promises, disputes, escalations,
                   payment_tokens)
  agent/          The tool-calling loop and Groq client/system-prompt assembly
  audio/          ASR (fine-tuned Hindi-English Whisper), TTS, VAD
  channels/       Borrower-facing HTTP API + dashboard (browser_api.py) and
                   the Telegram bot (telegram_bot.py) -- both drive the same
                   agent loop and the same account data
  guardrail/      Post-hoc grounding check on every reply before it ships
  memory/         Cross-session conversation recap for a returning borrower
  observability/  Aggregate operational metrics on top of the events log
  ops/            Account flagging (overdue/disputed/broken-promises) + the
                   internal ops dashboard API (ops/api.py): account creation,
                   escalation approve/reject, per-account document upload
                   (feeds the same RAG pipeline seed_kb.py uses for the
                   policy KB, plus a best-effort interest-rate extraction
                   pass for a signed loan agreement)
  outbound/       Proactive reminders: decide who's due/overdue, compose the
                   message, send it for real over Telegram (falls back to a
                   logged stub when no Telegram chat is linked yet)
  rag/            Hybrid retrieval (BM25 + multilingual embeddings + a
                   cross-encoder reranker) over the policy knowledge base and
                   per-account uploaded documents, backed by pgvector
  reports/        The on-demand report-generation pipeline (gather -> analyze
                   -> write -> accuracy-check)
  tools/          The MCP tools the agent loop calls

scripts/          seed_accounts.py, seed_kb.py, chat.py (interactive CLI),
                   run_outbound_pass.py (manual trigger -- no scheduler yet)
eval/             Standalone benchmarks (tool-calling, retrieval, reasoning
                   accuracy, a red-team pass, ASR WER) -- not part of the
                   pytest suite; run each with `python -m eval.<name>`
tests/            The pytest suite
data/kb/          The policy knowledge base (markdown)
schema.sql        The real Postgres schema (Supabase in production, a
                   disposable pgvector/pgvector:pg16 container in CI) --
                   not a migration tool; kept in sync by hand, and applied
                   for real at the start of every CI run so it's a tested
                   contract, not just a snapshot
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
| `ALTERNATE_GROQ_KEY` (or `ALTERNATE_GROQ_KEY2`, `3`, ...) | Automatic fallback when a key hits its rate limit (optional) |
| `DATABASE_URL` | Anything that touches account/RAG data (a Postgres connection string -- Supabase in production, any local/CI Postgres with the `vector` extension available otherwise) |
| `OPS_API_KEY` | The ops dashboard API (`ops/api.py`) -- required on every request as an `X-API-Key` header |
| `TELEGRAM_BOT_TOKEN` | The Telegram bot (`channels/telegram_bot.py`) |
| `CHAT_APP_BASE_URL` | Where a generated payment link should point (defaults to `http://localhost:8000`) |
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

# Borrower-facing chat + dashboard
uvicorn businessflow.channels.browser_api:app --reload            # port 8000

# Telegram bot (text + voice notes, long polling)
python -m businessflow.channels.telegram_bot

# Internal ops dashboard API (needs OPS_API_KEY, sent as X-API-Key)
uvicorn businessflow.ops.api:app --reload --port 8001
```

In production (the live links above) these three run as separate systemd
services on a small Azure VM behind Caddy for automatic HTTPS, with GitHub
Actions deploying on every push to `main` that passes CI (a forced-command
SSH key on the VM only ever runs its own fixed pull-and-restart sequence, so
a leaked deploy key can't run arbitrary commands on the box).

## Testing

```bash
pytest tests/
```

Tests that need a real Groq call, a real Postgres connection, or
`OPS_API_KEY` are individually gated with `pytest.mark.skipif` and skip
cleanly if the relevant `.env` value isn't set -- nothing is mocked in this
project; a test either runs for real or is skipped, never faked. Expect this
to be slow (multiple hours end to end) since it includes real ASR/LLM/DB
calls, not unit-test-speed mocks.

Standalone benchmarks under `eval/` (tool-calling accuracy, retrieval
Recall@k/MRR, ASR WER) aren't part of this suite -- run each directly, e.g.
`python -m eval.retrieval_benchmark`.

CI (`.github/workflows/tests.yml`) runs the same suite on every push/PR to
`main`, against its own disposable Postgres+pgvector container -- never the
production database. That separation is deliberate, not incidental: this
repo ran CI against the shared production Supabase database for a while, and
a CI run's own reseeding/test writes collided with real interactive testing
more than once (an escalation vanishing mid-demo, a payment token 404ing, KB
chunks getting purged out from under a developer -- all traced back to
exactly that). Add the Groq secrets under the repo's Settings -> Secrets and
variables -> Actions for the Groq-gated tests to run there too instead of
skipping.

## Status and known gaps

Built and verified end-to-end against real Groq/Postgres calls: the
tool-calling agent loop, the guardrail, hybrid retrieval with LLM-based query
translation (a real, measured fix for Hindi/Hinglish queries -- recall@1
0.61 -> 0.79 on the retrieval benchmark), the ops dashboard, a real payment-
link/redemption flow, the Telegram channel and its slash commands, per-
account document upload with interest-rate extraction, write-idempotency on
the state-mutating tools, a brute-force lockout on the per-account access
key, `eval/reasoning_accuracy.py` (an LLM-judge check on whether a reply's
stated reasoning matches the real tool results, not just whether the right
tool got called), `eval/red_team.py` (8 adversarial scenarios, all passing),
the report-generation pipeline, and multi-key Groq fallback (`agent/client.py`
rotates through as many configured alternate keys as are set, not just one --
built and proven live during this project's own real daily-quota exhaustion).

Known, real, currently-open gaps -- not hidden, because a lender's ops tool
claiming otherwise would be worse than having them:

- **A logged promise-to-pay never resolves.** `log_promise_to_pay` records a
  promise with `kept = NULL`, but nothing in this codebase ever evaluates it
  against what actually got paid and flips it to `true`/`false`. A
  "broken_promises" flag is therefore currently unreachable through any real
  code path -- worth building (compare `promised_date` + `promised_amount`
  against `payment_history` once the date passes), not yet built.
- **Disputes have no resolution path either.** `flag_dispute` opens one;
  nothing closes one back out. Escalations, by contrast, have a real
  approve/reject flow with a resolution reason.
- On a long, multi-turn conversation where an account's dispute/broken-
  promise block has already come up several times, the agent can still state
  a new concrete restructuring/partial-payment request is blocked from
  memory instead of calling the tool to verify it. A mechanical check
  (`guardrail/unverified_restructuring.py`) catches and safely deflects this
  before it reaches the borrower, so the underlying model tendency isn't
  eliminated but the consequence is.
- Query expansion (`retrieve(..., expand=True)`) and a non-default
  `reasoning_effort` were both built and A/B tested; neither showed a clear
  win on this project's scale, so both are left off by default and kept as
  opt-in parameters rather than removed.
- End-to-end latency measurement isn't built. A real frontend design pass
  (beyond the current functional dashboards) and horizontal scaling are out
  of scope for a demo project at this size.
