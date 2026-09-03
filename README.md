# BusinessFlow

**An AI collections agent for Indian SMB lending, live in production, that talks
to borrowers in Hindi, English, or Hinglish -- and never says a number it can't
back up.**

Lending in India runs on relationships and follow-up calls, not dashboards.
BusinessFlow is what that follow-up looks like when an AI agent does it: a
borrower can check their balance, negotiate a restructuring, log a promise to
pay, dispute a charge, or ask a policy question -- by text, by Telegram, or by
voice note -- and get back an answer grounded in a real database row or a
real retrieved document, never a guess. An ops team works the same accounts
from a real internal dashboard: an escalation queue, per-account document
upload with automatic interest-rate extraction, dispute history, and account
creation with a live EMI calculator.

It runs against synthetic demo accounts -- no real person's money moves --
but every system underneath it is real: real Postgres, a real deployed VM,
real payment recording, a real Telegram bot, real evals.

**Live:**
[businessflowai.duckdns.org](https://businessflowai.duckdns.org) (borrower chat + dashboard) ·
[businessflowai-ops.duckdns.org](https://businessflowai-ops.duckdns.org) (ops dashboard, needs an API key)

## What's actually built

**Voice AI, in Indian languages, trained not just prompted.** A Whisper ASR
model fine-tuned specifically for Hindi-English code-switched speech, WER-
benchmarked against the base model (`eval/asr_wer.py`) rather than assumed
better -- plus a VAD stage (Silero) ahead of it and TTS on the way out (Piper
for English, MMS for Hindi). Every model in the pipeline runs int8-quantized
ONNX, chosen deliberately for the RAM/latency envelope a real voice turn
needs, not left at full precision by default.

**A real agentic core, not a prompt wrapped around an API call.** One
tool-calling reasoning loop (`agent/loop.py`) that decides, per turn, which
of its tools to call -- account lookups, promise logging, dispute flagging,
policy retrieval, payment-link generation, escalation -- kept deliberately as
one bounded loop rather than a multi-agent swarm, because a live voice or
chat turn has no latency budget to spend on extra hops between agents. A
second, genuinely multi-stage pipeline exists where that latency budget
*doesn't* apply -- report generation runs gather -> analyze -> write -> a
mechanical accuracy-check pass, each stage's output checked against the last.

**RAG that's tuned, measured, and grounded, not just wired up.** Hybrid
retrieval -- BM25 keyword search plus multilingual embeddings plus a
cross-encoder reranker -- over both the policy knowledge base and
per-account uploaded documents (a signed loan agreement, say), backed by
pgvector on the same Postgres project as every other table rather than a
side vector store nobody keeps in sync. LLM-based query translation was
built and measured against this exact KB: recall@1 went 0.61 -> 0.79 on
Hindi/Hinglish queries once it shipped. Query expansion and a higher
reasoning-effort setting were also built and A/B tested against real traffic
-- both showed no clear win at this scale, so both stayed off by default
rather than shipped on faith.

**A mechanical guardrail, because "the prompt says don't hallucinate" isn't
a real safeguard.** Every reply is checked, after the model writes it and
before the borrower ever sees it, against what a real tool result or the
borrower's own message actually said -- every ₹ amount, every URL. A second,
narrower guardrail catches a specific failure mode that two rounds of prompt
tuning couldn't fully close: the model occasionally restating an old
dispute/restructuring block from memory on a long conversation instead of
re-verifying it. The guardrail intercepts that case and safely deflects
instead of letting an unverified claim reach the borrower.

**Evals that actually gate the work, not a notebook that was run once.** A
red-team suite (8 adversarial scenarios, all passing after a real fix for one
found gap), a reasoning-accuracy LLM-judge that checks a reply's *stated
reasoning* against the real tool results (not just whether the right tool
fired -- this is what caught a real crash on a hallucinated account ID), a
tool-calling benchmark, a retrieval benchmark, and a regression tracker
(`eval/tool_scoring.py`) that has already caught one real regression before a
fix was confirmed. Every one of these runs against real Groq/Postgres calls
-- nothing in this project is mocked; a test either runs for real or is
skipped cleanly when a credential isn't set.

**Shipped, not just running locally.** Three real channels -- browser chat +
dashboard, a Telegram bot with a full slash-command menu, and the ops
dashboard -- deployed as separate services on a real VM behind HTTPS, with
GitHub Actions deploying automatically on every push to `main` that passes
CI. CI runs against its own disposable, schema-verified Postgres+pgvector
container, isolated from the production database on purpose -- a lesson from
a real bug where CI's own test writes collided with live traffic.
Rate-limit resilience is built in too: `agent/client.py` rotates through as
many backup Groq keys as are configured, proven live during this project's
own real quota exhaustion, not a theoretical fallback.

**A real backend and real product surfaces**, not just an API. Postgres for
every piece of account state (payments, promises, disputes, escalations),
input validation at every boundary (E.164 phone format, positive-amount
checks, a brute-force lockout on the borrower access key), and two real
hand-built dashboards -- the ops team's account/escalation console and the
borrower's own account view -- each redesigned this session against real
reference points (not a default AI-generated look) until they read like a
real fintech product, not a demo.

## Architecture, in one paragraph

One Groq-hosted LLM (`openai/gpt-oss-20b`) with a belt of tools in a single
bounded reasoning loop. Tool calls are each grounded in a real Postgres row
or a real retrieved document chunk, never invented, and the grounding
guardrail (`guardrail/grounding.py`) double-checks the model's own final
reply before it ships. Three channels -- browser, Telegram, and the ops
dashboard -- sit on top of that one loop and one shared Postgres+pgvector
backend, so every environment (a developer's laptop, CI, the VM) reads from
one always-current index instead of independently-seeded copies.

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
                   run_outbound_pass.py (manual trigger), run_outbound_scheduler.py
                   (the real always-on scheduler, deployed as a 4th systemd
                   service in production)
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

# Outbound reminder scheduler (real, always-on -- fires the daily pass itself)
python -m scripts.run_outbound_scheduler
```

In production (the live links above) these four run as separate systemd
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
production database.

## Known gaps

Said plainly, because a lending ops tool that hides its own gaps would be
worse than one that lists them:

- **No multi-tenancy.** Everything (schema, ops auth, RAG's `general`
  policy-doc scope) assumes one lending business runs the whole deployment.
  A second real customer would need an `organizations` table threaded
  through every table and query, per-org ops credentials, and per-org RAG
  scoping -- not just a schema column. Deliberately not built: this is a
  single-tenant proof of concept, not a live multi-customer product yet.
- **No bulk/multi-select actions on the account grid.** Staff can trigger
  reminders for the whole filtered view at once (`POST /outbound/run`), but
  there's no way to select several specific accounts and fan out a
  clarification request to just them -- one account at a time for anything
  beyond reminders.
- End-to-end latency measurement isn't built. A production-grade frontend
  design pass beyond the current functional dashboards, and horizontal
  scaling, are out of scope for a project at this size.
