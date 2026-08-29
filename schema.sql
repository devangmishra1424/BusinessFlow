-- BusinessFlow's Postgres schema (Supabase), reverse-engineered from the
-- live database via information_schema and cross-checked against the SQL
-- in src/businessflow/accounts/store.py and scripts/seed_accounts.py --
-- this file previously didn't exist anywhere; the table shapes were only
-- inferable from scattered code. Not wired into an automated migration
-- tool (no Alembic/etc) -- this is a reference snapshot, not a source of
-- truth the app runs against. Keep it in sync by hand if the schema changes.
--
-- Note: Supabase also provisions its own `auth` schema (auth.users,
-- auth.refresh_tokens, etc, for Supabase's built-in user-auth system) --
-- unrelated to anything below, not used by this project, and not
-- reproduced here.

create table accounts (
    account_id           text primary key,
    borrower_name        text not null,
    business_name        text not null,
    phone_number         text not null,  -- E.164
    language_preference  text not null,  -- 'hi' | 'en' | 'hinglish'

    loan_type            text not null,
    principal_amount     numeric not null,
    emi_amount           numeric not null,
    tenure_months        integer not null,
    months_remaining     integer not null,
    emi_due_date         date not null,  -- due date of the current, unpaid EMI cycle

    nach_mandate_active  boolean not null default true,
    dispute_open         boolean not null default false,
    risk_tier            text not null,  -- 'low' | 'medium' | 'high'

    -- Extracted from an uploaded, signed loan agreement (see
    -- rag/extraction.py's extract_loan_terms, wired into ops/api.py's
    -- upload_account_document) -- nullable because most accounts won't
    -- have this until their agreement is uploaded and successfully
    -- parsed; NULL means "not extracted yet," not zero.
    interest_rate_pct    numeric,

    -- Fixed PIN assigned at seed time (no real sign-up flow exists) --
    -- verified once at conversation start via
    -- accounts.store.verify_account_key, gating access to this account's
    -- data for that session.
    access_key           text,

    -- The Telegram chat_id this account last verified from (see
    -- channels/telegram_bot.py's handle_incoming_message), so a decision
    -- made later on the ops dashboard -- approving/rejecting a
    -- restructuring request -- can actually reach the borrower without
    -- them being mid-conversation. NULL for accounts never verified over
    -- Telegram (e.g. browser-chat-only). Last-verified-chat-wins: this is
    -- a single link, not a history of every chat_id ever used.
    telegram_chat_id     bigint,

    created_at           timestamptz not null default now(),
    updated_at           timestamptz not null default now()
);

create table payment_history (
    id            bigint primary key generated always as identity,
    account_id    text not null references accounts(account_id),
    payment_date  date not null,
    amount        numeric not null,
    on_time       boolean not null,
    created_at    timestamptz not null default now()
);

create table promises (
    id               bigint primary key generated always as identity,
    account_id       text not null references accounts(account_id),
    made_on          date not null,
    promised_date    date not null,
    promised_amount  numeric not null,
    -- NULL until promised_date has passed and is checked against
    -- payment_history; true/false once evaluated.
    kept             boolean,
    created_at       timestamptz not null default now()
);

create table disputes (
    id           bigint primary key generated always as identity,
    account_id   text not null references accounts(account_id),
    reason       text not null,
    status       text not null default 'open',
    opened_at    timestamptz not null default now(),
    resolved_at  timestamptz
);

-- escalation_id (not a bigint id) is the primary key -- store.create_escalation
-- formats it as "ESC-0001" style from this sequence, since that's the id
-- shown to a human operator, not an internal row number.
create sequence escalation_seq;

create table escalations (
    escalation_id  text primary key,  -- e.g. 'ESC-0001', from escalation_seq
    account_id     text not null references accounts(account_id),
    reason         text not null,
    status         text not null default 'queued_for_human',  -- | 'approved' | 'rejected'
    created_at     timestamptz not null default now(),
    resolved_at    timestamptz,

    -- Structured terms a human can actually apply with one click, for the
    -- escalation types that propose a concrete account change (currently
    -- just extend_tenure -- see tools/escalation_tools.py's
    -- propose_restructuring) rather than just flagging something for a
    -- free-form human conversation. NULL for every other escalation kind
    -- (escalate_to_human, request_closure_certificate), same as before
    -- this column existed.
    proposed_changes  jsonb,

    -- Optional, ops-entered explanation shown back to the borrower when a
    -- proposal is rejected (see ops/api.py's POST /escalations/{id}/reject).
    -- NULL is a valid, expected value -- the reason box is optional.
    resolution_reason text
);

-- A single-use, expiring link minted for one specific payment (the
-- borrower-facing "Get a payment link" quick action, and the proactive
-- outbound reminder's own "Pay now" button) -- the token itself, not the
-- account_id, is what's embedded in the URL, so a guessable/edited link
-- can't be used to "pay" a different account or amount (see
-- accounts/store.py's redeem_payment_token).
create table payment_tokens (
    token       text primary key,
    account_id  text not null references accounts(account_id),
    amount      numeric not null,
    created_at  timestamptz not null default now(),
    expires_at  timestamptz not null,
    used_at     timestamptz
);

-- Generic audit/telemetry log -- every tool call (successful or failed)
-- and every conversation turn (see memory/conversation_memory.py) is
-- logged here as one event_type or another, rather than each concern
-- getting its own table. account_id is nullable: some events (e.g. a
-- check_policy call with no account_id) aren't tied to a specific borrower.
create table events (
    id          bigint primary key generated always as identity,
    account_id  text references accounts(account_id),
    event_type  text not null,
    details     jsonb not null default '{}',
    created_at  timestamptz not null default now()
);

-- RAG vector store: every ingested chunk (general policy KB, or a
-- specific borrower's uploaded loan agreement/KYC/etc), on the same
-- Postgres project as everything above rather than a separate local
-- ChromaDB file -- that file was a per-machine artifact nothing in the
-- deploy process ever rebuilt, so a fresh VM deploy silently ran with an
-- empty index instead of failing loudly. See rag/store.py, rag/ingest.py,
-- rag/retriever.py.
create extension if not exists vector;

create table document_chunks (
    id               text primary key,  -- sha1(source_document :: chunk_index :: ingested_at), see ingest.py
    source_document  text not null,
    document_type    text not null,  -- 'policy' | 'loan_agreement' | 'kyc' | 'regulatory' | 'other'

    -- 'general' sentinel for documents that apply to every borrower (the
    -- hand-written policy KB); a real account_id scopes a document to one
    -- specific borrower's own upload. Not an FK to accounts(account_id) --
    -- 'general' isn't a real account, and a chunk must survive its
    -- account being deleted (it never is today, but nothing here should
    -- assume that).
    account_id       text not null default 'general',

    chunk_index      integer not null,
    headings         text not null default '',  -- e.g. "Restructuring options > One-time settlement"
    document_text    text not null,
    embedding        vector(384) not null,  -- intfloat/multilingual-e5-small's real output size

    -- NULL means active. A corrected re-upload of the same source_document
    -- marks the previous generation's chunks superseded (never deletes
    -- them, so a compliance/history look-back can still see what a
    -- document said before) -- see ingest.py's _supersede_existing_chunks.
    -- Retrieval only ever considers superseded_at is null.
    superseded_at    timestamptz,

    created_at       timestamptz not null default now()
);

create index document_chunks_embedding_idx on document_chunks using hnsw (embedding vector_cosine_ops);
create index document_chunks_account_active_idx on document_chunks (account_id) where superseded_at is null;
create index document_chunks_source_document_idx on document_chunks (source_document);
