-- One case, one conversation: a dispute row is now one of two kinds.
--   'dispute'        the borrower says an amount, fee or due date is wrong (every existing row)
--   'payment_claim'  the borrower says they already paid; ops checks the bank record, then
--                    confirms it (records the payment) or closes it
--
-- Additive and safe to run while the old code is live: the old code never names this column,
-- and every existing and newly inserted row gets 'dispute' from the default. Re-runnable.
-- Apply with: psql "$DATABASE_URL" -f migrations/2026-10-15_dispute_kind.sql
-- schema.sql already has the column (CI builds its database from schema.sql, not from here).

begin;

alter table disputes add column if not exists kind text not null default 'dispute';

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'disputes_kind_check') then
        alter table disputes add constraint disputes_kind_check check (kind in ('dispute', 'payment_claim'));
    end if;
end
$$;

commit;
