-- Bank-statement reconciliation (cash application): every credit line ops upload from the lender's bank statement,
-- what it was matched to, and what became of it. accounts/bank_statement.py reads and matches; accounts/store.py
-- posts (record_payment's rules) and links the payment here.
--
--   status      'posted' (a payment was recorded from it), 'suggested' (a best guess waiting for a person),
--               'unmatched' (nothing identified a borrower), 'ignored' (a person said it is not a loan payment)
--   line_hash   the same bank line uploaded again hashes the same, so re-uploading a statement never posts twice
--   dispute_id  the open payment claim the line settled, when it matched the reference the borrower gave
--
-- Additive: a new table only, safe while the old code runs. Re-runnable.
-- Apply with: psql "$DATABASE_URL" -f migrations/2026-10-24_bank_lines.sql, then re-run `python -m scripts.setup_eval_schema`.

create table if not exists bank_lines (
    id            bigint primary key generated always as identity,
    statement     text not null,
    line_no       integer not null,
    line_date     date not null,
    amount        numeric not null check (amount > 0),
    narration     text not null default '',
    reference     text not null default '',
    line_hash     text not null unique,
    status        text not null check (status in ('posted', 'suggested', 'unmatched', 'ignored')),
    account_id    text references accounts(account_id) on delete set null,
    match_reason  text,
    payment_id    bigint references payment_history(id) on delete set null,
    dispute_id    bigint references disputes(id) on delete set null,
    decided_by    text,
    decided_note  text,
    decided_at    timestamptz,
    created_at    timestamptz not null default now()
);
create index if not exists bank_lines_open_idx on bank_lines (created_at desc) where status in ('suggested', 'unmatched');
