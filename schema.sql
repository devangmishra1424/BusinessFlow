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

    -- Fixed PIN assigned at seed time (no real sign-up flow exists) --
    -- verified once at conversation start via
    -- accounts.store.verify_account_key, gating access to this account's
    -- data for that session.
    access_key           text,

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
    status         text not null default 'queued_for_human',
    created_at     timestamptz not null default now(),
    resolved_at    timestamptz
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
