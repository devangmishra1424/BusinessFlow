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
for English, Piper's `hi_IN` voice for Hindi with Meta MMS as the fallback). Every model in the pipeline runs int8-quantized
ONNX, chosen deliberately for the RAM/latency envelope a real voice turn
needs, not left at full precision by default. Multi-sentence replies are
synthesized per sentence and rejoined with a deliberate pause between them
(`audio/tts.py`), not concatenated back-to-back -- and that change is
measured, not just assumed to help: `eval/voice_naturalness_benchmark.py`
scores every synthesized reply with torchaudio's pretrained SQUIM model
(reference-free STOI/PESQ/SI-SDR, already a dependency here -- no new heavy
model), plus a deterministic check that `audio/verbalizer.py` actually
caught every date/amount before it ever reached TTS.

**A real agentic core, not a prompt wrapped around an API call.** One
tool-calling reasoning loop (`agent/loop.py`) that decides, per turn, which
of its tools to call -- account lookups, promise logging, dispute flagging,
policy retrieval, payment-link generation, escalation -- kept deliberately as
one bounded loop rather than a multi-agent swarm, because a live voice or
chat turn has no latency budget to spend on extra hops between agents. A
second, genuinely multi-stage pipeline exists where that latency budget
*doesn't* apply -- report generation runs gather -> analyze -> write -> a
mechanical accuracy-check pass, each stage's output checked against the last.

**RAG that's measured, with the numbers and their limits stated.** Hybrid
retrieval -- BM25 keyword search plus multilingual embeddings plus a
cross-encoder reranker -- over the policy knowledge base and each borrower's
own uploaded documents, stored in pgvector on the same Postgres project as
every other table. Each service answers from one in-memory snapshot of the
chunks and their embeddings and re-checks a one-query fingerprint of the
table every 15 seconds, so an uploaded document is searchable within
seconds with no restart. (An earlier design read the text from a snapshot
but ran the embedding search live against the table, and raised an error
whenever a document arrived between the two.) Ingestion (Docling) is
all-or-nothing: parse, check the text is readable, embed, then replace the
old version in one transaction, so a failed or concurrent upload can never
leave a document retired with no replacement. Uploads take PDF, Word and
Markdown; a scanned PDF goes through Docling's bundled English OCR (rendered
test scans, skewed and noisy ones included, came back with every key
figure), and a scan with no readable text -- blank, too poor, or Hindi, which
that OCR cannot read -- is refused with an explanation instead of being stored
as junk (`tests/test_ingest_readability.py`).

`python -m eval.retrieval_benchmark` rebuilds the corpus from the repo with no
database and scores four question sets: 38 development questions, 47
held-out ones written before any variant was run, 270 loan-agreement fact
questions (30 distinct questions asked once per borrower; the intervals
resample the 30), and 30 the documents cannot answer. Every change is
compared with its parent on the same questions under a decision rule fixed in
advance (a change ships only if a paired 95% interval clears zero), and the
report is `eval/results/retrieval_benchmark.md`. As deployed, a correct chunk
is among the two the agent receives for 71% of the development questions
(55-84%), 94% of the held-out ones (85-100%) and 92% of the loan-agreement
fact questions (82-99%), and none of the 2,160 retrievals the benchmark ran for
borrower-scoped questions returned another borrower's document. The LLM
translation of Hindi and Hinglish questions is worth measuring because the
keyword and rerank stages are English-only: without it Recall@2 falls by 0.09
on the held-out questions and 0.16 on the loan ones, both intervals below
zero (0.08 on the development set, where the interval includes zero). A larger candidate pool, putting heading paths into the text, and
rank fusion were each tried and none cleared the rule (a pool of 8 gained
0.05-0.08 on the development and loan questions but its intervals touch zero,
and it costs roughly 25-90% more latency; headings improved the development
set by 0.06 MRR but the held-out set pointed the other way, -0.02, so the gain
was not confirmed), so none was adopted. A score threshold for "nothing relevant"
keeps 97% of answerable questions but drops only 35% of the in-domain
questions the documents cannot answer, so it is not used as a safety
mechanism. Where it still misses: loosely worded questions (Recall@2 0.76 for
the colloquial phrasing of the loan questions, against 1.00 for the clean and
Hinglish ones). Limits: the questions were written by the engineer who built the
retriever, n is small, the loan agreements all come from one template, and
ground truth is a heading or an answer string, not a human judgement of
the chunk. Query expansion and a higher
reasoning-effort setting were also built and A/B tested against real traffic
-- both showed no clear win at this scale, so both stayed off by default
rather than shipped on faith. That earlier round was a manual, run-then-
compare check; `agent/prompt_versions.py` makes the same idea a standing
capability -- a second system-prompt variant can run against a real,
configurable percentage of live conversations (deterministically bucketed
per borrower, not a coin-flip per turn), with which version each
conversation got logged to `events` for comparison. Off by default
(0% rollout) until an operator deliberately turns an experiment on.

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
skipped cleanly when a credential isn't set. These no longer only run when
a human remembers to: `scripts/run_eval_monitor.py` fires five suites
(tool-calling, reasoning-accuracy, red-team, latency, voice-naturalness)
nightly and alerts (Telegram, with a loud log line either way) the moment
one actually regresses -- voice-naturalness only alerts on its
deterministic correctness checks, not its SQUIM scores, which are real
but too noisy run-to-run (Piper/MMS-TTS are both stochastic) to threshold
without crying wolf; those stay a printed trend line for a human to read.

**Shipped, not just running locally.** Three real channels -- browser chat +
dashboard, a Telegram bot with a full slash-command menu (every proactive reminder also carries one-tap
"I'll pay tomorrow / in 3 days / in a week", "talk to a person" and "dispute" buttons that call the same tools with
no LLM turn, `channels/reminder_actions.py`), and the ops
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

**Money on the books, applied in a stated order.** What a borrower owes is one
definition (`accounts/dues.py`): this month's EMI less any credit, plus every
charge still open. The late fee is a real charge (the `charges` table), assessed
once per EMI still unpaid after the 3-day grace period, never compounded and
never charged interest. A payment is applied by one pure function,
`allocate_payment`: the EMI first, then the fee, then anything extra (which the
borrower chooses how to use) -- so paying the EMI keeps the loan current, and a
fee left unpaid stays visibly owed. The pay page shows that split before the
borrower confirms. A payment dated within the grace period removes a fee
charged before it was recorded; ops can waive a fee only with their name and a
reason, which the borrower is sent.

**A promise is judged by its own words.** A borrower promises in one payment or
in up to 4 parts, each an amount by a date (`accounts/promises.py`). In chat the
assistant must read the exact terms back, and the tool records them only on a
second call after the borrower says yes; this is enforced in code, not only in
the prompt. Ops tick that they read the terms back. A part is kept once the money
paid since the promise reaches it by its date, and broken the day after if it
does not. There is no extra window. One broken part breaks the promise and voids
the rest, and it counts once. A payment judges the promise in the same
transaction, so the borrower sees "Paid" at once. While a promise runs, routine
reminders pause and the borrower is reminded of their own terms the day before
each part and on the day. A new promise replaces an open one, which is not
counted as broken.

**Cash application from the bank statement.** Ops upload the lender's bank
statement as CSV. Each credit is matched to a borrower by what it carries
(`accounts/bank_statement.py`): the account number in the narration, the
reference the borrower gave on an open payment claim, or their phone number.
Exactly one account found this way means the line is recorded at once, dated
the day the bank received it, and it closes that claim. The borrower is told in
the claim's thread. A name alone is only ever a suggestion, waiting with its
reason for one click, as is anything that looks like a payment already on
record. Re-uploading a statement records nothing twice, because each line is
hashed. A line that is not a loan payment is set aside with who and why, never
deleted.

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
| `CHAT_INTERNAL_URL`, `OPS_VOICE_MODEL` | Ops dashboard voice (optional): where the ops service reaches the chat service's speech models (default `http://127.0.0.1:8000`), and the model that reads a request (default `openai/gpt-oss-120b`) |
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

# Eval regression monitor (one-shot -- tool-calling/reasoning/red-team/latency/
# voice-naturalness check against real Groq+Postgres+RAG+TTS, alerts on
# regression; see scripts/run_eval_monitor.py's own docstring for why this is
# NOT a resident process). Every suite reseeds the demo accounts, so against
# the production database the evals use their own copy of the tables, schema
# "eval" -- never the live demo data. Build it once (and re-run after a KB doc
# or loan agreement changes):
python -m scripts.setup_eval_schema
python -m scripts.run_eval_monitor
```

In production (the live links above) these run as separate systemd units on
a small Azure VM behind Caddy for automatic HTTPS: four always-on services
(chat, ops, bot, outbound scheduler) plus the eval monitor as a `systemd`
**timer** (`businessflow-eval-monitor.timer`, `OnCalendar=*-*-* 21:00:00
UTC`) firing a one-shot service, not a 5th resident process. Found live: a
resident 5th process holding the full torch/ASR/TTS/RAG stack in memory
24/7, on a VM with only 3.8GB total RAM, was a direct contributor to the
kernel OOM-killing the (unrelated, borrower-facing) Telegram bot mid-
conversation -- a one-shot timer holds that memory for a few minutes a
night instead of all day. GitHub Actions deploys on every push to `main`
that passes CI (a forced-command SSH key on the VM only ever runs its own
fixed pull-and-restart sequence, so a leaked deploy key can't run arbitrary
commands on the box).

The Telegram bot's own session state is otherwise still in-memory and not
immune to being restarted (a deploy, or that same OOM class of event) --
but it's no longer catastrophic: a borrower who already verified is
recognized again from the durable `telegram_chat_id` column
(`accounts.get_account_by_telegram_chat_id`) on their very next message,
with a "Welcome back" reply, instead of being asked to re-verify from
scratch.

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
Recall@k/MRR, ASR WER, end-to-end turn latency) aren't part of this suite --
run each directly, e.g. `python -m eval.retrieval_benchmark` or
`python -m eval.latency_benchmark`.

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
- **Scanned-document OCR reads English, not Hindi.** Docling's bundled OCR
  returned garbage for a clean Devanagari scan (confidence 0.50-0.57 against
  0.93-0.99 for English), so an upload like that is refused with a message
  asking for the text version rather than ingested. Phone photos (jpg/png)
  are not accepted at all -- only PDF, Word and Markdown. A heavily degraded
  English scan can also come back empty (also refused), and a moderately
  degraded two-page one missed one figure in six, which no confidence check
  catches. Tested on scans rendered here, not on real-world ones.
- **Voice accent and quality.** English is spoken by Piper `en_US-lessac` (an American voice; Piper has no
  Indian-English voice, and it reads as a little mechanical) and Hindi by Piper's `hi_IN` priyamvada voice, with Meta
  MMS-TTS as the automatic fallback (`TTS_HINDI_ENGINE=mms` makes it the first choice; `PIPER_HINDI_VOICE` picks
  rohan, pratham or priyamvada; the voice file is fetched when a service starts). The Hindi choice was benchmarked
  before it was merged (`python -m eval.tts_benchmark`, 40 sentences x 3 repeats per engine, 32 in Devanagari scored for clarity and 8 in Roman script for
  robustness, pass criteria fixed in advance, results in `eval/results/tts_benchmark.md`): read back by Whisper large-v3, Piper's character
  error is 7.3% against MMS's 12.8% (paired per-sentence difference -5.5 points, 95% interval -9.2 to -1.9; the gain
  is in numbers and acronyms, plain sentences are close at 7.0% vs 7.7%), Piper is 7x faster than MMS on 2 CPU
  threads (real-time factor 0.10 vs 0.74), its length varies less from one repeat to the next (2.4% vs 4.6%), and in
  a fresh process it grows by 138 MB against MMS's 175 MB. The same benchmark found that Piper's output peaked at
  full scale and overshot after the Opus step; replies are now kept 1 dB below it. One listener ranked priyamvada
  above MMS. Kokoro (82M) ranked first by ear and clearest, but costs ~0.7-0.9 GB and runs at about half
  real time on those 2 threads, so it was not adopted for a 3.8 GB VM. The only Indian-accented open model found,
  AI4Bharat's Indic Parler-TTS, needs a GPU; true Indian-accent English in production means a GPU server or a
  hosted engine, and none is in place. A Hindi reply that contains Latin script (English words, acronyms, or a whole
  reply in Roman script) is spoken in two voices (Devanagari runs by the Hindi voice, Latin runs by the English
  one) instead of losing those words or producing no voice; a spoken reply that still cannot be made is announced
  in the chat. The benchmark measures intelligibility as heard by one speech recogniser, speed and memory; naturalness and
  accent were judged by one listener's ears, not by a metric.
- **"Naturalise tone" (web chat only).** A bar under the chat header looks for a natural-sounding voice in the browser for the
  chat's language (a voice whose name says Natural, Neural, Premium or Enhanced: Edge's "... Online (Natural)" voices,
  Windows Natural voices, macOS Premium/Enhanced; Chrome's Google voices and the old Windows ones are not counted). None:
  it says "Sorry, not available". Found: it switches to it and says a short sample, and the speaker icon on a reply then
  reads the words from `POST /conversations/{id}/speech-text` (the same `verbalize()` step the server voice uses, so ids and
  amounts are spoken the same way) with that voice. If the voice cannot be used (none for this language, the text could
  not be fetched, or it never starts within 5 s) the server voice plays instead, so a tap never ends in silence. Off by
  default; only the on/off choice is remembered, and the voice is looked up again on every load. **Privacy:** Edge's natural voices are
  online, so the reply text (which can contain the borrower's amounts, dates and account id) is sent to the browser's provider to
  be spoken; the page says so when it switches on, and the server voice keeps everything on this machine. Telegram voice
  notes are unchanged (they are audio files made on the server). A reply can also be stopped by tapping its speaker
  again, and only one reply plays at a time. **Not verified:** how any natural voice sounds (the development browser has
  none, and audio cannot be judged by a tool; the option was exercised with one real voice renamed to look natural,
  covering the switch, the sample, the spoken text, stop, the fallback to server audio, the Hindi case and a stale saved
  setting). Tests: `tests/js/naturalvoice.test.cjs` (22, run with node by `tests/test_naturalvoice_js.py`) and five
  `speech-text` endpoint tests in `tests/test_browser_api.py`.
- **"What I can help with" buttons (web chat and Telegram).** A short list of tappable suggestions so a borrower who does not
  know what to ask is not left with a blank box: in the web chat under the greeting (and again from the "What can you help
  with?" button in the bar under the header), and in Telegram on `/start`, on `/help`, and right after a chat verifies. One list
  in `channels/abilities.py` feeds both (the web page reads it from `GET /conversations/{id}/abilities`). A tap is the same as
  typing the button's sentence: it goes through the normal assistant turn with the same checks, except "How do I verify my
  account?" and "More questions or doubts", which are answered at once with no assistant turn. Kept modest on purpose: eight
  entries for a verified chat, each for something the assistant has a tool for, worded "ask about" where the tool only reviews
  (more time to pay, loan closure); the last button is always "More questions or doubts", and its reply promises only that the
  assistant will say so if it cannot answer. An unverified chat is offered only the general entries and how to verify, so it never
  sees a button the assistant would have to refuse. Telegram's `/menu` (buttons that run a command directly) is unchanged.
  **Limits:** the Hindi wording was written without a Hindi speaker's review; each tap in either channel is a full assistant turn
  (the same cost and Groq quota as typing the question); the buttons can only say what the assistant is built to do, and the
  assistant's own free-text answer to "what can you help with?" (which the model writes, and which can say "everything related to
  your loan account") is not changed by this. **Not verified:** a real tap in Telegram (needs a linked chat and a running bot):
  Telegram is covered by tests that drive the handlers with minimal stand-ins for the message and query objects. The web page was
  exercised in a browser, with the assistant's reply stubbed so no model quota was spent. Tests: `tests/test_abilities.py` (17),
  `tests/test_telegram_abilities.py` (19), four `abilities` endpoint tests in `tests/test_browser_api.py`.
- **Voice on the ops dashboard.** A microphone, a question box and a menu button at the bottom right of the ops
  dashboard (also over an open profile, so "her" and "this account" mean the profile on screen). English or Hindi.
  - *Ask* about the portfolio, the inbox, the escalation rate or one borrower. An answer is a short spoken line plus rows
    on screen you can tap (a borrower opens their profile, a figure opens its list, a phone number dials) and the next
    steps that make sense from there.
  - *Work through* a queue, one item at a time (`ops/voice_workflows.py`): **Start my day** (calls to make, the inbox by
    kind, payment claims, cases waiting over 3 days, broken promises); the **call list** (overdue borrowers, most overdue
    first; an account with an open dispute or payment claim is left out, as its reminders are); **work the inbox** (or
    only payment claims). Each step reads its item fresh and offers the actions that fit it -- for a call, the outcomes
    ("no answer", "reached, she'll pay 8,000 by Friday" logs the call and the promise from one card); for a payment claim,
    confirm it, ask for the UTR reference (the case card's own wording), or close it; for a request for more time, approve
    or reject. Confirming an action on the item moves the run on; "next", "skip", "back", "stop" work by voice or button.
    A case a colleague closed meanwhile is passed over, never acted on twice. The run lives in the browser tab.
  - *Do* one thing: record a payment, confirm a payment claim, close a case, log a promise or a call, message a borrower
    or a group, approve or reject a request, hand an account to a person, send reminders, fix contact details.

  One model call (`ops/voice.py`, `OPS_VOICE_MODEL`, a different model from the borrower chat's so they never share a
  daily Groq allowance) picks one of 26 abilities from a fixed menu and fills in the details; it sees no account data.
  Answers are written by code from the same data the cards show (`ops/voice_reply.py`) and spoken in the chat's English
  Piper voice. An action becomes a confirmation card -- what will happen, what was heard, what the borrower will be sent
  -- and only the operator's Confirm click sends it, through the same endpoints the dashboard's own forms use; no voice
  endpoint writes anything. An amount, a phone number, or a figure in a message the model drafted must be found in what
  the operator actually said (`ops/spoken_numbers.py`: "12,500", "twelve and a half thousand", "1.2 lakh", "साढ़े बारह
  हज़ार", "१२५००", ...) or the card is not shown; "the full amount" comes from the database. Hindi number words outside
  one to twenty and the round tens and fives (तेईस, say) are not read, so such a figure is asked for again. Two close
  names are asked about, never guessed. Where the model's reading was found wrong, code corrects it rather than the
  prompt: confirming a claim nobody filed becomes recording a payment; "close it" on a request closes the request.
  Speech runs on the VM: the ops service sends the recording to the chat service over loopback (`ops/speech_client.py`),
  which already holds Whisper and Piper, instead of loading a second copy (+125 MB for Whisper, measured; +138 MB for
  Piper); that internal endpoint needs the ops key and refuses anything that came through the public proxy (checked live:
  403). Replies are in English; no spoken "yes" -- Confirm is a click. Picking the ability, with the real model and the
  26-ability menu (`python -m eval.ops_voice_benchmark`, all 54 in one run): 51 of 54 (95% interval 85-98%) -- 36 of 38
  single requests and 15 of 16 workflow, in-run and Hindi requests; each miss a wrong ability, never a wrong detail; no
  figure invented; median 1.3 s. The misses: "who should I call first" started the call list (whose first step names
  that borrower); "she paid the full amount" was read as confirming a claim (code makes it a payment when none was
  filed); "what should I focus on this morning" got the overview (which offers Start my day) -- it passed the run
  before, so the same sentence can get a different reading on a second try. About 2,800 tokens a request (measured:
  2,796 in, 28 out; roughly 70 a day on a free-tier allowance of 200k). **Verified live (2026-10-15):** the
  microphone and the spoken reply on the deployed dashboard, a Hindi question understood. **Not verified:** the
  workflows on the live dashboard (exercised in a preview with made-up accounts, the real model, real Whisper and Piper,
  and Confirm requests recorded instead of sent); exact timings on the VM. Tests: `tests/test_ops_voice.py`,
  `tests/test_ops_voice_workflows.py`, `tests/test_ops_voice_api.py`, `tests/test_spoken_numbers.py`.
- A production-grade frontend design pass beyond the current functional
  dashboards, and horizontal scaling, are out of scope for a project at
  this size. (End-to-end latency IS measured, see `eval.latency_benchmark`
  in Testing above -- it times the full text/LLM/tool-loop path against
  real Groq, Postgres, and RAG, separating clean runs from ones that hit a
  mid-turn key rotation.)
