-- The late fee becomes a real charge on the books, instead of a figure every screen worked out for itself.
--
-- charges: one row per fee actually assessed. A late fee is assessed once per missed instalment (for_due_date is that
-- instalment's due date) when the grace period has passed, never twice for the same instalment (the unique key), never
-- compounded and never charged interest. Its status says what happened to it: still due, paid (by which payment, how
-- much so far), or waived (by whom, why). Nothing is ever deleted: a waiver is a recorded decision.
--
-- payment_history.charges_paid: how much of a payment went to charges, so a payment of EMI + fee reads as exactly that.
-- payment_history.kind gains 'charge' (a payment that only paid charges, e.g. the fee left over after the EMI).
--
-- Additive and safe to run while the old code is live: the old code never names the new table or column. Re-runnable.
-- Apply with: psql "$DATABASE_URL" -f migrations/2026-10-22_charges.sql
-- schema.sql already has all of this (CI builds its database from schema.sql, not from here).
-- After applying it to the live database, re-run `python -m scripts.setup_eval_schema`: the nightly eval refuses to run
-- on a copy that lacks a live table.

begin;

create table if not exists charges (
    id                  bigint primary key generated always as identity,
    account_id          text not null references accounts(account_id) on delete cascade,
    kind                text not null check (kind in ('late_fee')),
    for_due_date        date not null,
    amount              numeric not null check (amount > 0),
    amount_paid         numeric not null default 0 check (amount_paid >= 0),
    assessed_on         date not null,
    status              text not null default 'due' check (status in ('due', 'paid', 'waived')),
    paid_on             date,
    paid_by_payment_id  bigint references payment_history(id) on delete set null,
    waived_on           date,
    waived_by           text,
    waived_reason       text,
    created_at          timestamptz not null default now(),
    unique (account_id, kind, for_due_date)
);
create index if not exists charges_account_due_idx on charges (account_id) where status = 'due';

alter table payment_history add column if not exists charges_paid numeric not null default 0;

-- Replace the kind check, whatever it is called on this database (schema.sql's inline check is auto-named, so the name
-- is looked up rather than assumed: dropping a guessed name that doesn't exist would leave the old check in force and
-- every 'charge' payment refused).
do $$
declare c record;
begin
    for c in
        select conname from pg_constraint
        where conrelid = 'payment_history'::regclass and contype = 'c' and pg_get_constraintdef(oid) ilike '%kind%'
    loop
        execute format('alter table payment_history drop constraint %I', c.conname);
    end loop;
    alter table payment_history add constraint payment_history_kind_check check (kind in (
        'regular', 'extra_unapplied', 'extra_applied', 'overpayment_applied',
        'principal_prepayment_reduce_emi', 'principal_prepayment_reduce_tenure', 'charge'));
end
$$;

commit;
