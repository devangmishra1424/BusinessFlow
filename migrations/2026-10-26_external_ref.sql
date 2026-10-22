-- A loan imported from the lender's loan book keeps the lender's own reference (ops/loan_book.py), so importing the
-- same book again skips loans already on the books instead of opening them twice. Additive; null for every existing
-- account. Re-runnable.

alter table accounts add column if not exists external_ref text;
create unique index if not exists accounts_external_ref_key on accounts (external_ref);
