-- A promise to pay becomes a plan in the borrower's own words: one or more parts, each an amount by a date, judged
-- exactly by those words (owner's decision, 2026-10-22: "if the words of the promise are broken, the promise is broken").
--
-- promises gains:
--   plan_id      the plan a row is a part of ('P-<id of its first part>'); every existing row is a one-part plan
--   part_no      1-based position in the plan; parts_total how many parts it has
--   terms_text   the terms as they were read back to the borrower and confirmed ("₹10,000 by 25 Oct, then ...")
--   made_via     'chat' (the assistant, after a read-back), 'reminder_button', 'ops', or 'seed'
--   resolution   'kept' | 'broken' | 'void' (void: a later part of a broken plan, or an open part of a plan the
--                borrower replaced with a new promise); NULL while open. kept stays as before for every reader:
--                true for kept, false for broken, NULL for open or void
--   resolved_on  when the resolution was written
--
-- Additive and safe to run while the old code is live: it never names the new columns, and new rows it writes get the
-- defaults (a one-part plan). Re-runnable. Apply with: psql "$DATABASE_URL" -f migrations/2026-10-23_promise_plans.sql
-- schema.sql already has all of this. Afterwards re-run `python -m scripts.setup_eval_schema`.

begin;

alter table promises add column if not exists plan_id text;
alter table promises add column if not exists part_no integer not null default 1;
alter table promises add column if not exists parts_total integer not null default 1;
alter table promises add column if not exists terms_text text;
alter table promises add column if not exists made_via text;
alter table promises add column if not exists resolution text;
alter table promises add column if not exists resolved_on date;

-- Every existing row is a one-part plan, resolved as its kept value already says.
update promises set plan_id = 'P-' || id where plan_id is null;
update promises set resolution = case when kept then 'kept' else 'broken' end where kept is not null and resolution is null;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'promises_resolution_check') then
        alter table promises add constraint promises_resolution_check check (resolution in ('kept', 'broken', 'void'));
    end if;
end
$$;

create index if not exists promises_account_open_idx on promises (account_id) where resolution is null;

commit;
